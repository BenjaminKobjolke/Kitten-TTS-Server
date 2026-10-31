"""ONNX inference routine for KittenTTS speech generation."""

from typing import Optional, Tuple

import numpy as np


def synthesize(
    engine, text: str, voice: str, speed: float = 1.0, *, raise_on_error: bool = False
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Synthesizes audio from text using the loaded KittenTTS model.

    Args:
        text: The text to synthesize.
        voice: Voice identifier (model-dependent).
        speed: Speech speed factor (1.0 is normal speed).

    Returns:
        A tuple containing the audio waveform (numpy array) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    logger = engine.logger
    np = engine.np
    ort = engine.ort
    MODEL_LOADED = engine.MODEL_LOADED
    onnx_session = engine.onnx_session
    voices_data = engine.voices_data
    phonemizer_backend = engine.phonemizer_backend
    _voice_aliases = engine._voice_aliases
    _speed_priors = engine._speed_priors
    get_all_accepted_voices = engine.get_all_accepted_voices
    basic_english_tokenize = engine.basic_english_tokenize
    _phoneme_token_chunks = engine._phoneme_token_chunks

    if not MODEL_LOADED or onnx_session is None:
        logger.error("KittenTTS model is not loaded. Cannot synthesize audio.")
        return None, None

    # Validate voice against all accepted identifiers (named + internal)
    accepted = get_all_accepted_voices()
    if voice not in accepted:
        logger.error(
            f"Voice '{voice}' not available for current model. "
            f"Available voices: {accepted}"
        )
        return None, None

    # Resolve voice aliases (named voices -> internal expr-voice keys)
    internal_voice = _voice_aliases.get(voice, voice)
    if internal_voice != voice:
        logger.debug(f"Resolved voice alias '{voice}' -> '{internal_voice}'")

    # Apply speed prior for this voice if available (from model config)
    prior = _speed_priors.get(internal_voice, 1.0)
    if prior != 1.0:
        speed = speed * prior
        logger.debug(
            f"Applied speed prior {prior} for voice '{internal_voice}', "
            f"effective speed: {speed}"
        )

    try:
        logger.debug(
            f"Synthesizing with voice='{voice}' (internal='{internal_voice}'), "
            f"speed={speed}"
        )

        # Get voice embedding and ensure correct shape [1, 256]
        voice_embedding = voices_data[internal_voice]
        if voice_embedding.ndim == 2 and voice_embedding.shape[0] > 1:
            # ONNX2: multiple reference embeddings, select based on text length
            # for varied prosody across different text lengths
            text_length = len(text)
            ref_id = min(text_length, voice_embedding.shape[0] - 1)
            voice_embedding = voice_embedding[ref_id:ref_id + 1]
            logger.debug(
                f"Selected ONNX2 embedding row {ref_id} (text_len={text_length}), "
                f"shape {voice_embedding.shape}"
            )
        elif voice_embedding.ndim == 1:
            voice_embedding = voice_embedding.reshape(1, -1)
        voice_embedding = voice_embedding.astype(np.float32)
        logger.debug(f"Input text (first 100 chars): '{text[:100]}...'")

        # Phonemize the input text
        import logging as log_module
        phonemizer_logger = log_module.getLogger("phonemizer")
        original_level = phonemizer_logger.level
        phonemizer_logger.setLevel(log_module.ERROR)

        phonemes_list = phonemizer_backend.phonemize([text])
        phonemizer_logger.setLevel(original_level)

        # Process phonemes to get token IDs
        phonemes = " ".join(basic_english_tokenize(phonemes_list[0]))
        token_chunks = _phoneme_token_chunks(phonemes)

        # Determine the execution device
        provider = onnx_session.get_providers()[0]
        audio_segments = []

        for tokens in token_chunks:
            if provider == "CUDAExecutionProvider":
                # GPU inference with I/O Binding
                input_ids_np = np.array([tokens], dtype=np.int64)
                ref_s_np = voice_embedding
                speed_array_np = np.array([speed], dtype=np.float32)

                input_ids_ort = ort.OrtValue.ortvalue_from_numpy(
                    input_ids_np, "cuda", 0
                )
                ref_s_ort = ort.OrtValue.ortvalue_from_numpy(ref_s_np, "cuda", 0)
                speed_array_ort = ort.OrtValue.ortvalue_from_numpy(
                    speed_array_np, "cuda", 0
                )

                io_binding = onnx_session.io_binding()
                io_binding.bind_ortvalue_input("input_ids", input_ids_ort)
                io_binding.bind_ortvalue_input("style", ref_s_ort)
                io_binding.bind_ortvalue_input("speed", speed_array_ort)

                output_name = onnx_session.get_outputs()[0].name
                io_binding.bind_output(output_name, "cuda")

                onnx_session.run_with_iobinding(io_binding)
                output_ortvalue = io_binding.get_outputs()[0]
                audio_segments.append(output_ortvalue.numpy())

            else:
                # CPU inference
                input_ids = np.array([tokens], dtype=np.int64)
                ref_s = voice_embedding
                speed_array = np.array([speed], dtype=np.float32)

                onnx_inputs = {
                    "input_ids": input_ids,
                    "style": ref_s,
                    "speed": speed_array,
                }
                outputs = onnx_session.run(None, onnx_inputs)
                audio_segments.append(outputs[0])

        if not audio_segments:
            raise ValueError("Text produced no model tokens")
        audio = np.concatenate(audio_segments, axis=-1)

        sample_rate = 24000

        logger.info(
            f"Successfully generated {len(audio)} audio samples at {sample_rate}Hz"
        )
        return audio, sample_rate

    except Exception as e:
        logger.error(f"Error during KittenTTS synthesis: {e}", exc_info=True)
        if raise_on_error:
            raise
        return None, None



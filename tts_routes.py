"""Text-to-speech API routes."""

import io
import logging
import time
import asyncio
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import engine
import utils
from config import (
    config_manager,
    get_audio_output_format,
    get_audio_sample_rate,
    get_gen_default_speed,
)
from models import CustomTTSRequest, ErrorResponse

logger = logging.getLogger("server")
router = APIRouter()


@router.post(
    "/tts",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/opus": {}},
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks
) -> StreamingResponse:
    """
    Generates speech audio from text using specified parameters.
    Returns audio as a stream (WAV or Opus).
    """
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("TTS request received")

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received /tts request: voice='{request.voice}', "
        f"format='{request.output_format}'"
    )
    logger.debug(
        f"TTS params: speed={request.speed}, split={request.split_text}, "
        f"chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")

    perf_monitor.record("Parameters resolved")

    all_audio_segments_np: List[np.ndarray] = []
    final_output_sample_rate = get_audio_sample_rate()
    engine_output_sample_rate: Optional[int] = None

    if request.split_text and len(request.text) > (
        request.chunk_size * 1.5 if request.chunk_size else 120 * 1.5
    ):
        chunk_size_to_use = (
            request.chunk_size if request.chunk_size is not None else 120
        )
        logger.info(f"Splitting text into chunks of size ~{chunk_size_to_use}.")
        text_chunks = utils.chunk_text_by_sentences(request.text, chunk_size_to_use)
        perf_monitor.record(f"Text split into {len(text_chunks)} chunks")
    else:
        text_chunks = [request.text]
        logger.info(
            "Processing text as a single chunk "
            "(splitting not enabled or text too short)."
        )

    if not text_chunks:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable chunks."
        )

    for i, chunk in enumerate(text_chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
        try:
            chunk_audio_np, chunk_sr_from_engine = await asyncio.to_thread(
                engine.synthesize,
                text=chunk,
                voice=request.voice,
                speed=(
                    request.speed
                    if request.speed is not None
                    else get_gen_default_speed()
                ),
            )
            perf_monitor.record(f"Engine synthesized chunk {i+1}")

            if chunk_audio_np is None or chunk_sr_from_engine is None:
                error_detail = f"TTS engine failed to synthesize audio for chunk {i+1}."
                logger.error(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)

            if engine_output_sample_rate is None:
                engine_output_sample_rate = chunk_sr_from_engine
            elif engine_output_sample_rate != chunk_sr_from_engine:
                logger.warning(
                    f"Inconsistent sample rate from engine: chunk {i+1} "
                    f"({chunk_sr_from_engine}Hz) differs from previous "
                    f"({engine_output_sample_rate}Hz). Using first chunk's SR."
                )

            # The speed factor is handled by the engine, so no post-processing
            # is needed here.

            all_audio_segments_np.append(chunk_audio_np)

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e_chunk:
            logger.error(
                "Error processing audio chunk %s: %s",
                i + 1,
                e_chunk,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error processing audio chunk {i + 1}.",
            ) from e_chunk

    if not all_audio_segments_np:
        logger.error("No audio segments were successfully generated.")
        raise HTTPException(
            status_code=500, detail="Audio generation resulted in no output."
        )

    if engine_output_sample_rate is None:
        logger.error("Engine output sample rate could not be determined.")
        raise HTTPException(
            status_code=500, detail="Failed to determine engine sample rate."
        )

    try:
        if len(all_audio_segments_np) > 1:
            # Add silence between chunks for natural pauses
            silence_duration_ms = 200  # silence between chunks
            silence_samples = int(
                silence_duration_ms / 1000 * engine_output_sample_rate
            )
            silence_array = np.zeros(silence_samples, dtype=np.float32)

            # Apply crossfade and add silence between chunks
            crossfade_samples = int(0.01 * engine_output_sample_rate)  # 10ms crossfade

            merged_audio = []
            for i, chunk in enumerate(all_audio_segments_np):
                if i == 0:
                    merged_audio.append(chunk)
                else:
                    # Add silence gap between chunks
                    merged_audio.append(silence_array)

                    # Then add the next chunk with optional crossfade
                    if (
                        len(merged_audio[-2]) >= crossfade_samples
                        and len(chunk) >= crossfade_samples
                    ):
                        # Apply fade out to end of previous audio (before silence)
                        fade_out = np.linspace(1, 0, crossfade_samples)
                        merged_audio[-2][-crossfade_samples:] *= fade_out

                        # Apply fade in to start of current chunk
                        fade_in = np.linspace(0, 1, crossfade_samples)
                        chunk_copy = chunk.copy()
                        chunk_copy[:crossfade_samples] *= fade_in
                        merged_audio.append(chunk_copy)
                    else:
                        merged_audio.append(chunk)

            final_audio_np = await asyncio.to_thread(np.concatenate, merged_audio)
            logger.debug(
                f"Added {silence_duration_ms}ms silence between "
                f"{len(all_audio_segments_np)} chunks"
            )
        else:
            final_audio_np = all_audio_segments_np[0]

        perf_monitor.record("All audio chunks processed and concatenated")

    except ValueError as e_concat:
        logger.error(f"Audio concatenation failed: {e_concat}", exc_info=True)
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail="Audio concatenation failed."
        ) from e_concat

    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
    )

    encoded_audio_bytes = await asyncio.to_thread(
        utils.encode_audio,
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
    )
    perf_monitor.record(
        f"Final audio encoded to {output_format_str} "
        f"(target SR: {final_output_sample_rate}Hz from engine SR: "
        f"{engine_output_sample_rate}Hz)"
    )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format_str} "
            f"or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to encode audio to {output_format_str} "
                "or generated invalid audio."
            ),
        )

    media_type = f"audio/{output_format_str}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    suggested_filename_base = f"tts_output_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    logger.info(
        f"Successfully generated audio: {download_filename}, "
        f"{len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    logger.debug(perf_monitor.report())

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
    )



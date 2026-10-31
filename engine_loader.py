"""Model loading routine for the KittenTTS engine."""

_STATE_NAMES = (
    "onnx_session",
    "voices_data",
    "MODEL_LOADED",
    "loaded_model_selector",
    "loaded_model_repo_id",
    "loaded_model_device",
    "_voice_aliases",
    "_speed_priors",
)


def load_model(engine) -> bool:
    """Load a model while keeping state in the public engine module."""
    global onnx_session, voices_data, MODEL_LOADED
    global loaded_model_selector, loaded_model_repo_id, loaded_model_device
    global _voice_aliases, _speed_priors

    for name in _STATE_NAMES:
        globals()[name] = getattr(engine, name)
    try:
        return _load_model(engine)
    finally:
        for name in _STATE_NAMES:
            setattr(engine, name, globals()[name])


def _load_model(engine) -> bool:
    """
    Loads the KittenTTS model from Hugging Face Hub and initializes ONNX session.
    Automatically downloads the model if not already cached.
    Updates global variables for model components.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global onnx_session, voices_data, MODEL_LOADED
    global loaded_model_selector, loaded_model_repo_id, loaded_model_device

    logger = engine.logger
    config_manager = engine.config_manager
    resolve_selector = engine.resolve_selector
    MODEL_REGISTRY = engine.MODEL_REGISTRY
    hf_hub_download = engine.hf_hub_download
    json = engine.json
    np = engine.np
    ort = engine.ort
    _V01_VOICE_ALIASES = engine._V01_VOICE_ALIASES
    _check_cancelled = engine._check_cancelled
    _update_download_status = engine._update_download_status
    _init_espeak = engine._init_espeak

    if MODEL_LOADED:
        logger.info("KittenTTS model is already loaded.")
        return True

    try:
        # Resolve the model selector from config
        config_value = config_manager.get_string(
            "model.repo_id", "KittenML/kitten-tts-nano-0.1"
        )
        selector = resolve_selector(config_value)
        reg = MODEL_REGISTRY[selector]
        model_repo_id = reg["repo_id"]

        model_cache_path = config_manager.get_path(
            "paths.model_cache", "./model_cache", ensure_absolute=True
        )

        logger.info(f"Loading KittenTTS model: {reg['display_name']} ({model_repo_id})")
        logger.info(f"Using cache directory: {model_cache_path}")

        # Ensure cache directory exists
        model_cache_path.mkdir(parents=True, exist_ok=True)

        # Phase 1: Download config.json
        _check_cancelled()
        _update_download_status(
            "downloading", f"Downloading config for {reg['display_name']}...", 10
        )

        config_path = hf_hub_download(
            repo_id=model_repo_id,
            filename="config.json",
            cache_dir=str(model_cache_path),
        )

        # Load config to get model filenames
        with open(config_path, "r") as f:
            model_config = json.load(f)

        model_type = model_config.get("type", "")
        if model_type not in ("ONNX1", "ONNX2"):
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                "Expected ONNX1 or ONNX2."
            )

        # Phase 2: Download model file (can be large - check cancellation)
        _check_cancelled()
        model_filename = model_config["model_file"]
        _update_download_status(
            "downloading",
            f"Downloading model file ({reg['size']})...",
            30,
        )

        model_path = hf_hub_download(
            repo_id=model_repo_id,
            filename=model_filename,
            cache_dir=str(model_cache_path),
        )

        # Phase 3: Download voices file
        _check_cancelled()
        _update_download_status("downloading", "Downloading voice data...", 60)

        voices_path = hf_hub_download(
            repo_id=model_repo_id,
            filename=model_config["voices"],
            cache_dir=str(model_cache_path),
        )

        # Phase 4: Load voices data and voice aliases
        _update_download_status("loading", "Loading voice embeddings...", 70)
        voices_data = np.load(voices_path)
        logger.info(f"Loaded voices data with keys: {list(voices_data.keys())}")

        # Load voice aliases - v0.8 models have these in config.json,
        # v0.1/v0.2 models use our own named aliases
        global _voice_aliases, _speed_priors
        _voice_aliases = {}
        _speed_priors = {}
        if "voice_aliases" in model_config:
            _voice_aliases = dict(model_config["voice_aliases"])
            logger.info(f"Loaded voice aliases from model config: {_voice_aliases}")
        elif reg["version"] in ("0.1", "0.2"):
            _voice_aliases = dict(_V01_VOICE_ALIASES)
            logger.info(f"Applied v0.1 named voice aliases: {_voice_aliases}")

        # Load speed priors (per-voice speed correction factors from model config)
        if "speed_priors" in model_config:
            _speed_priors = dict(model_config["speed_priors"])
            logger.info(f"Loaded speed priors: {_speed_priors}")

        # Phase 5: Initialize ONNX session
        _check_cancelled()
        _update_download_status("loading", "Initializing ONNX inference session...", 80)

        device_setting = config_manager.get_string("tts_engine.device", "auto").lower()
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")

        sess_options = ort.SessionOptions()
        providers = []
        provider_options = []

        attempt_gpu = device_setting in ["auto", "cuda", "gpu"]
        is_gpu_available = "CUDAExecutionProvider" in available_providers

        if attempt_gpu and is_gpu_available:
            logger.info(
                f"'{device_setting}' mode selected and "
                "CUDAExecutionProvider is available."
            )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"device_id": "0"}, {}]
            loaded_model_device = "cuda"
        else:
            if device_setting in ["cuda", "gpu"] and not is_gpu_available:
                logger.warning(
                    f"Configuration explicitly requests GPU ('{device_setting}'), "
                    "but CUDAExecutionProvider is NOT available."
                )
            logger.info("Defaulting to CPUExecutionProvider.")
            providers = ["CPUExecutionProvider"]
            loaded_model_device = "cpu"

        logger.info(f"Initializing ONNX InferenceSession with providers: {providers}")

        if "CUDAExecutionProvider" in providers:
            onnx_session = ort.InferenceSession(
                str(model_path), sess_options,
                providers=providers, provider_options=provider_options,
            )
        else:
            onnx_session = ort.InferenceSession(
                str(model_path), sess_options, providers=providers,
            )

        # Phase 6: Initialize phonemizer (shared across models, only once)
        _check_cancelled()
        _update_download_status("loading", "Initializing phonemizer...", 90)
        _init_espeak()

        # Done
        loaded_model_selector = selector
        loaded_model_repo_id = model_repo_id
        MODEL_LOADED = True

        _update_download_status(
            "complete", f"{reg['display_name']} loaded successfully!", 100
        )
        logger.info(f"KittenTTS model loaded successfully: {reg['display_name']}")
        return True

    except RuntimeError as e:
        if "cancelled" in str(e).lower():
            logger.info(f"Model loading cancelled: {e}")
            onnx_session = None
            voices_data = None
            MODEL_LOADED = False
            loaded_model_selector = None
            loaded_model_repo_id = None
            loaded_model_device = None
            _update_download_status("cancelled", "Model loading was cancelled.", 0)
            return False
        else:
            logger.error(f"Error loading KittenTTS model: {e}", exc_info=True)
            onnx_session = None
            voices_data = None
            MODEL_LOADED = False
            loaded_model_selector = None
            loaded_model_repo_id = None
            loaded_model_device = None
            _update_download_status("error", "", 0, str(e))
            return False
    except Exception as e:
        logger.error(f"Error loading KittenTTS model: {e}", exc_info=True)
        onnx_session = None
        voices_data = None
        MODEL_LOADED = False
        loaded_model_selector = None
        loaded_model_repo_id = None
        loaded_model_device = None
        _update_download_status("error", "", 0, str(e))
        return False



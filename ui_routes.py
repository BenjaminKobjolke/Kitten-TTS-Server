"""HTTP routes that serve the browser UI and its initial data."""

import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

import engine
from config import get_full_config_for_template

logger = logging.getLogger("server")
router = APIRouter()
ui_static_path = Path(__file__).parent / "ui"
@router.get("/styles.css", include_in_schema=False)
async def get_main_styles() -> FileResponse:
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")


@router.get("/script.js", include_in_schema=False)
async def get_main_script() -> FileResponse:
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request) -> Response:
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    try:
        index_path = ui_static_path / "index.html"
        if index_path.is_file():
            return FileResponse(index_path, media_type="text/html")
        return HTMLResponse(
            "<html><body><h1>Not Found</h1><p>index.html not found.</p></body></html>",
            status_code=404,
        )
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1>"
            "<p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )


@router.get("/api/model-info", tags=["Model Information"])
async def get_model_info_endpoint() -> dict[str, object]:
    """Returns detailed information about the currently loaded TTS model."""
    logger.debug("Request received for /api/model-info")
    try:
        return engine.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve model information"
        )


@router.get("/api/model-registry", tags=["Model Information"])
async def get_model_registry_endpoint() -> dict[str, object]:
    """Returns the full list of available models for the UI dropdown."""
    return engine.get_model_registry()


@router.get("/api/model-status", tags=["Model Information"])
async def get_model_status_endpoint() -> dict[str, object]:
    """Returns the current download/loading progress for model switching."""
    return engine.get_download_status()


@router.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data() -> dict[str, object]:
    """
    Provides all necessary initial data for the UI to render,
    including configuration, file lists, presets, and model information.
    """
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()

        # Get model information for UI
        model_info = engine.get_model_info()
        model_registry = engine.get_model_registry()

        loaded_presets = []
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(
                        f"Invalid format in {presets_file}. Expected a list, "
                        f"got {type(yaml_content)}."
                    )
        else:
            logger.info(
                f"Presets file not found: {presets_file}. "
                "No presets will be loaded for initial data."
            )

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoice": None,
        }

        return {
            "config": full_config,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
            "model_info": model_info,
            "model_registry": model_registry,
            "available_voices": engine.get_available_voices(),
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to load initial data for UI."
        )

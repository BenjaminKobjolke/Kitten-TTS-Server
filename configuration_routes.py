"""Configuration and model lifecycle API routes."""

import logging

from fastapi import APIRouter, HTTPException, Request

import engine
from config import config_manager
from models import UpdateStatusResponse

logger = logging.getLogger("server")
router = APIRouter()


@router.post(
    "/save_settings", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def save_settings_endpoint(request: Request) -> UpdateStatusResponse:
    """
    Saves partial configuration updates to the config.yaml file.
    Merges the update with the current configuration.
    """
    logger.info("Request received for /save_settings.")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object for /save_settings.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            restart_needed = any(
                key in partial_update
                for key in ["server", "tts_engine", "paths", "model"]
            )
            message = "Settings saved successfully."
            if restart_needed:
                message += (
                    " A server restart may be required for some changes "
                    "to take full effect."
                )
            return UpdateStatusResponse(
                message=message, restart_needed=restart_needed
            )
        else:
            logger.error(
                "Failed to save configuration via config_manager.update_and_save."
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration file due to an internal error.",
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for /save_settings: {ve}")
        raise HTTPException(status_code=400, detail="Invalid request data.") from ve
    except Exception as e:
        logger.error(f"Error processing /save_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during settings save.",
        ) from e


@router.post(
    "/reset_settings", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def reset_settings_endpoint() -> UpdateStatusResponse:
    """Resets the configuration in config.yaml back to hardcoded defaults."""
    logger.warning("Request received to reset all configurations to default values.")
    try:
        if config_manager.reset_and_save():
            logger.info("Configuration successfully reset to defaults and saved.")
            return UpdateStatusResponse(
                message=(
                    "Configuration reset to defaults. Please reload the page. "
                    "A server restart may be beneficial."
                ),
                restart_needed=True,
            )
        else:
            logger.error("Failed to reset and save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to reset and save configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing /reset_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during settings reset.",
        ) from e


@router.post(
    "/restart_server", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def restart_server_endpoint() -> UpdateStatusResponse:
    """
    Triggers an async hot-swap of the TTS model engine.
    Returns immediately while the model downloads and loads in the background.
    The UI polls /api/model-status to track progress.
    """
    logger.info("Request received for /restart_server (Async Model Hot-Swap).")

    try:
        engine.reload_model_async()
        return UpdateStatusResponse(
            message=(
                "Model reload initiated in background. "
                "Poll /api/model-status for progress."
            ),
            restart_needed=False,
        )
    except Exception as e:
        logger.error(f"Error initiating model hot-swap: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate model reload.",
        ) from e


@router.post("/api/cancel-loading", tags=["Configuration"])
async def cancel_loading_endpoint() -> dict[str, str]:
    """Cancels any in-progress model loading."""
    logger.info("Request received for /api/cancel-loading.")
    cancelled = engine.cancel_loading()
    if cancelled:
        return {"message": "Model loading cancellation requested."}
    return {"message": "No model loading in progress."}


@router.post("/api/unload", tags=["Configuration"])
async def unload_model_endpoint() -> dict[str, str]:
    """
    Unloads the TTS model and releases all resources.
    The model must be reloaded via /restart_server before TTS requests can be
    processed.
    """
    logger.info("Request received for /api/unload (Model Unload).")
    try:
        success = engine.unload_model()
        if success:
            return {"message": "Model unloaded successfully. Resources released."}
        else:
            raise HTTPException(status_code=500, detail="Failed to unload model.")
    except Exception as e:
        logger.error(f"Error during model unload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to unload model.") from e

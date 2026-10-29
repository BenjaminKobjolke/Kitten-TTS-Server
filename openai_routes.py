"""OpenAI-compatible speech API routes."""

import io
import logging
import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import engine
import utils
from config import get_audio_sample_rate

logger = logging.getLogger("server")
router = APIRouter()


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"  # Add "mp3"
    speed: float = 1.0
    seed: Optional[int] = None


@router.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(
    request: OpenAISpeechRequest,
) -> StreamingResponse:
    # Check if the TTS model is loaded
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    try:
        # Synthesize the audio
        audio_np, sr = await asyncio.to_thread(
            engine.synthesize,
            text=request.input_,
            voice=request.voice,
            speed=request.speed,
        )

        if audio_np is None or sr is None:
            raise HTTPException(
                status_code=500, detail="TTS engine failed to synthesize audio."
            )

        # Ensure it's 1D
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze()

        # Encode the audio to the requested format
        encoded_audio = await asyncio.to_thread(
            utils.encode_audio,
            audio_array=audio_np,
            sample_rate=sr,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )

        if encoded_audio is None:
            raise HTTPException(status_code=500, detail="Failed to encode audio.")

        # Determine the media type
        media_type = f"audio/{request.response_format}"

        # Return the streaming response
        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Speech generation failed.") from e

"""Headless CLI so ai-chat-data-server can run KittenTTS as an external tool.

Modes (see dataserver-tool.json and README "Use as ai-chat-data-server tool"):
  --check                                              load the model, exit 0/1
  --text T --out DIR [--lang en] [--voice V] [--speed S]  write DIR/speech.mp3 once
  --serve                                              keep the model loaded, JSON lines:
      stdout {"ready": true} once, then per stdin line {"text", "out", "lang", "voice", "speed"}
      answers {"ok": true} or {"ok": false, "error": "..."}
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

import engine
import utils

DEFAULT_VOICE = "Bruno"
DEFAULT_SPEED = 1.1  # matches config.yaml
CHUNK_SIZE = 200


def synth(text, out, lang="", voice="", speed="", **_unused):
    # ponytail: serve mode gets every placeholder (e.g. "url"); ignore the ones TTS does not use.
    lang = lang or "en"
    if lang != "en":
        raise ValueError(f"unsupported language {lang!r}, only 'en'")
    if not text or not text.strip():
        raise ValueError("text is empty")
    if not out:
        raise ValueError("out is empty")
    voice = voice or DEFAULT_VOICE
    voices = engine.get_available_voices()
    if voice not in voices:
        raise ValueError(f"unknown voice {voice!r}, available: {', '.join(voices)}")
    speed = float(speed) if speed not in ("", None) else DEFAULT_SPEED
    if not 0.5 <= speed <= 2.0:
        raise ValueError(f"speed {speed} out of range 0.5-2.0")

    parts = []
    sample_rate = None
    for chunk in utils.chunk_text_by_sentences(text, CHUNK_SIZE):
        audio, sample_rate = engine.synthesize(chunk, voice, speed)
        if audio is None:
            raise RuntimeError(f"synthesis failed for chunk {chunk[:40]!r}")
        parts.append(audio)
    if not parts:
        raise ValueError("text produced no chunks")
    os.makedirs(out, exist_ok=True)
    sf.write(os.path.join(out, "speech.mp3"), np.concatenate(parts), sample_rate, format="MP3")


def serve():
    proto = sys.stdout
    sys.stdout = sys.stderr  # library prints / HF warnings must not corrupt the protocol
    if not engine.load_model():
        return 1
    proto.write(json.dumps({"ready": True}) + "\n")
    proto.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            synth(**json.loads(line))
            reply = {"ok": True}
        except Exception as e:  # noqa: BLE001 - worker must survive any bad request
            reply = {"ok": False, "error": str(e)}
        proto.write(json.dumps(reply) + "\n")
        proto.flush()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--text", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--lang", default="")
    parser.add_argument("--voice", default="")
    parser.add_argument("--speed", default="")
    args = parser.parse_args()

    if args.serve:
        return serve()
    if not engine.load_model():
        print("model load failed", file=sys.stderr)
        return 1
    if args.check:
        return 0
    try:
        synth(args.text, args.out, args.lang, args.voice, args.speed)
    except Exception as e:  # noqa: BLE001 - report any failure as exit 1
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

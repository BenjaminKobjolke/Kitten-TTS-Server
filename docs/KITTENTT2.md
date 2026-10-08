# KittenTTS 2 support (not implemented)

Status: researched 2026-10-08, deliberately postponed. This file records what was found and how
to add it later.

## What KittenTTS 2 is

- Repo: <https://github.com/KittenML/KittenTTS>, weights: `KittenML/kitten-tts-2` on Hugging Face (not gated).
- A 1.7B-parameter 1-bit speech language model. **Not ONNX**: safetensors + PyTorch.
- Shipped through a new pip package `kittenml` (0.9.4 as of 2026-10-06, Python >=3.10, pure-Python wheel).
  The same `KittenTTS(...)` constructor also loads the 0.8 ONNX models.
- 47 built-in voices (the eight 0.8 names carry over: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo),
  20 languages, 24 kHz output.
- Extras: voice cloning (`reference=` 5-30 s clip), `preset="expressive"`, expression markup
  (`[joyful]`, `<laugh>`, `(((emphasis)))`), `generate_stream`, optional vLLM backend (Linux + CUDA only).

### Weight variants

| `weights=` | Size | Notes |
|---|---|---|
| default (`packed`) | ~950 MB | lossless ternary |
| `emb4` | ~506 MB | quantized token embedding, small quality cost |
| `full` | ~3.5 GB | plain bf16 |

### Costs

- CPU needs ~6 GB RAM, GPU ~8 GB free. Upstream publishes no real-time factor; expect it to be far
  slower than the ONNX models on CPU.
- `kittenml` dependencies: `torch>=2.6`, `torchaudio>=2.6`, `transformers>=4.55`, `safetensors`, `diffusers`,
  `librosa`, `einops`, `omegaconf`, `hydra-core`, `s3tokenizer`, `resemble-perth`, `pyloudnorm`,
  `kitten-text-processing`, `espeakng_loader`, `phonemizer`, `onnxruntime`, `brotli`, `scipy`, `soundfile`,
  `numpy`, `huggingface_hub`.
- Licence: weights are under the Stellon Labs Community License. Free for research/non-commercial use and
  for commercial use below USD 1M annual revenue or total funding; above that a separate licence is
  required. Attribution required when distributing. The `kittenml` code is Apache 2.0.

### Upstream usage

```python
from kittenml import KittenTTS
import soundfile as sf

m = KittenTTS("KittenML/kitten-tts-2")            # or weights="emb4"
audio = m.generate("Hello, world.", voice="Bruno")
sf.write("output.wav", audio, m.sample_rate)
print(m.available_voices)
```

## Why it is not a one-line registry change

`engine.py` reimplements ONNX inference itself and does not use the upstream package. `load_model()`
rejects any model whose `config.json` `type` is not `ONNX1`/`ONNX2`, and `synthesize()` feeds a fixed
`input_ids` / `style` / `speed` ONNX session. v2 fits none of that.

## Implementation plan

Do not reimplement v2 inference; delegate to `kittenml` behind a `backend` flag in the registry.
Hot-swap, UI dropdown, `/tts`, `/v1/audio/speech`, chunking and encoding all read from the registry
and need no changes.

1. **Dependencies**: add `kittenml>=0.9.4` to `pyproject.toml`, `requirements.txt`, `requirements-nvidia.txt`;
   `uv lock`. Install into both `venv/` (web UI) and `.venv/` (uv/CLI). Import lazily inside the v2 branch so
   the ONNX models keep working if the install fails.
2. **`MODEL_REGISTRY`** in `engine.py`: add `kitten-tts-2` (`weights: None`, ~950MB) and `kitten-tts-2-emb4`
   (`weights: "emb4"`, ~506MB), both `repo_id: KittenML/kitten-tts-2`, `"backend": "kittenml"`, version `2`,
   params `1.7B`, `default_voice: "Bruno"`, `voices` seeded with `_V08_VOICES`. Both share one repo id, so make
   sure `_REPO_TO_SELECTOR` maps the full repo id to `kitten-tts-2`.
3. **`load_model()`**: after the cache directory is created, branch on `reg.get("backend") == "kittenml"`:
   construct `KittenTTS(repo_id, weights=...)` into a new global `kitten2_model`, set
   `reg["voices"] = list(kitten2_model.available_voices)`, set the `loaded_model_*` globals and `MODEL_LOADED`,
   report status `complete`, return. Skip the `config.json` / ONNX / npz / espeak phases. Reset
   `kitten2_model` in the existing `except` blocks.
4. **`unload_model()`**: set `kitten2_model = None` (existing `gc.collect()` and `torch.cuda.empty_cache()` do the rest).
5. **`synthesize()`**: loaded-check becomes "ONNX session or v2 model present". After voice validation, if
   `kitten2_model` is set, return `(np.asarray(kitten2_model.generate(text, voice=voice), dtype=np.float32),
   kitten2_model.sample_rate)`. Voice-alias and speed-prior code is ONNX-only.
6. **Text**: README model table and "all 7 models" wording; `ui/index.html` static "8 high-quality voices" /
   "ultra-lightweight (<25MB)" help text.

### Unknowns to check first

Upstream docs do not say. Inspect the installed package (`inspect.signature` on `KittenTTS.__init__` and `.generate`):

- **Speed**: does `generate` accept `speed`? If not, use `librosa.effects.time_stretch(audio, rate=speed)`
  (librosa is already a dependency) or ignore speed for v2.
- **Cache directory**: does the constructor accept `cache_dir`? If not, weights land in the default HF cache
  instead of `./model_cache`.
- **Device**: is there a device argument, and how does it map to `tts_engine.device`?
- Whether the README's "506 MiB" figure or the model card's "469 MB" is right for `emb4`.

### Out of scope for a first pass

Voice cloning, preset selector, streaming, vLLM backend, non-English in `dataserver_cli.py`
(`lang == "en"` gate, and `normalize=False` for non-English text). Expression tags work without any
change because text is passed through untouched.

### Verification

1. `config.yaml` `model.repo_id: kitten-tts-2-emb4`, then `uv run python dataserver_cli.py --check` exits 0.
2. UI dropdown lists both v2 entries; voice list shows 47 after load.
3. Synthesize with `Bruno` via UI and `POST /v1/audio/speech`; note generation time on CPU.
4. Speed 1.3 gives shorter audio than speed 1.0.
5. Hot-swap back to `kitten-nano-0.8-fp32` and synthesize, confirming the ONNX path and unload still work.

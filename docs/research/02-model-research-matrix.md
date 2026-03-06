# Model Research Matrix

## Purpose

This matrix exists to pressure-test the platform architecture across task classes, provider pressure, scheduler classes, and host surfaces. It is no longer acceptable to let LTX alone define “universal.”

## Matrix

| Family | Workload class | Provider or source pressure | Scheduler class | Capability pressure | Artifact or output pressure | Memory pressure | Likely native hotspots | Host integration pressure | Validation priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `LTX-2.3 Fast` | video generation | Hugging Face primary, large multi-asset source set, local bundles needed for trusted offline paths | `media_video_dit` | strict resolution and frame rules, image conditioning, LoRA, audio-aware outputs | `mp4`, `mov`, `wav`, staged artifacts, upscalers | very high | VAE decode, latent pack or unpack, output encode path | very high: desktop and Comfy | Phase B |
| image diffusion (`SDXL` or `FLUX`) | image generation | Hugging Face first, later OCI or artifact-registry pressure possible | `image_diffusion` | size constraints, LoRA, inpaint or conditioning variants | `png`, `jpg`, optional intermediate latents | high | attention helpers, VAE decode, tiled decode | medium | Phase C |
| `Whisper` or equivalent | ASR or non-generation speech | Hugging Face plus local files; processor-heavy preflight | `speech_streaming` or `speech_batch` | chunking, timestamps, language hints | text, json, subtitle outputs | medium | feature extraction, chunk merge | medium | Phase C |
| `F5-TTS` or `mlx-audio` TTS family | speech generation | Hugging Face plus community-converted models | `speech_tts_audio` | speaker or voice conditioning, streaming chunks, voice design | `wav`, `flac`, streaming segments | high | spectrogram or mel preprocessing, vocoder path | medium | Phase C or D |
| `Qwen2.5-VL`, `Gemma 3n`, or similar | VLM or omni understanding | Hugging Face plus possible `trust_remote_code` pressure | `vlm_interactive` | multi-image or audio processor state, prompt templates, long-context behavior | text or json outputs, optional media refs | medium | processor glue more than kernels at first | medium to high | Phase C |
| pure text family via `mlx-lm` | text generation pressure test | Hugging Face and local bundles; batching and compatibility-facade pressure | `text_batch_or_stream` | streaming deltas, prompt cache, model switching | text, structured JSON, tool-style outputs | medium | KV and cache policy, sampler efficiency | high because compatibility-facade pressure is strongest here | Phase C or explicit scope exclusion |
| second video family (`Wan`, `HunyuanVideo`, or similar) | video generation validation | provider and conversion pressure beyond LTX naming and metadata shapes | `media_video_dit` | different shape rules and pipeline decomposition | video outputs, possibly different upscaler strategy | very high | model-family-specific kernels and decode paths | medium | Phase E or later |

## Shared Abstractions The Platform Must Support

- provider inspection before blind full downloads
- portable artifact conversion and manifesting
- machine-local build cache separation
- truthful capability declarations
- handle-based input and output flows
- stage-aware scheduling
- provenance and policy propagation

## Why A Text Family Is In The Matrix

The project is still centered on local generative and multimodal workloads, not generic text serving. A text family is included here because it is the fastest way to expose hidden assumptions in:

- capability schema shape
- scheduler taxonomy
- compatibility facade pressure
- caching policy

If the platform decides not to support text in the stable v1 scope, that choice must be explicit and benchmark-backed, not accidental.

## Validation Basket

The smallest honest architecture-validation basket is now:

1. `LTX-2.3 Fast` T2V
2. `LTX-2.3 Fast` I2V
3. one image diffusion family
4. one non-generation family
5. one text family or a written exclusion

Anything smaller risks freezing an LTX-shaped architecture by mistake.

# Qwen-Image Family Adapter

This package carries the first-class `MLXR` family contract for `Qwen-Image`.

Current truthful scope:

- source inspection for official diffusers-style `Qwen-Image` bundles
- componentized portable-artifact conversion for the released primary rows
- workflow planning for:
  - `image.generate` on `Qwen-Image-2512`
  - `image.edit` on the current edit rows
- owned `Qwen-Image-2512` prompt encoding on top of a local `Qwen2.5-VL`
  text-only MLX substrate and the official fixed prompt scaffold
- explicit runtime stage order of `prompt_encode -> generate -> encode_output`
- first owned `Qwen-Image-2512` `image.generate` receipt through the canonical
  `mlxr generate` path
- first owned `Qwen-Image-2512 Lightning` LoRA-backed receipt through the same
  canonical `mlxr generate` path, using the generic `--lora` CLI surface plus a
  family-local `qwen_image.scheduler_preset=lightning` extension
- first owned `Wuli-Art/Qwen-Image-2512-Turbo-LoRA` `V3.0` receipt through the
  same runtime path, after normalizing its diffusers-style
  `diffusion_model.*` + `lora_A` / `lora_B` naming onto the owned transformer
  seams and pairing it with a family-local `qwen_image.scheduler_preset=turbo_wuli`
  extension
- first owned `Qwen-Image-Edit-2511` backend slice:
  multimodal prompt encoding through the official `Qwen2.5-VL` processor path,
  edit-aware transformer modulation for `zero_cond_t`, VAE encode plus decode,
  and conditioned latent concatenation in the owned MLX runtime
- first owned `Qwen-Image-Edit-2511 Lightning` receipt through the canonical
  `mlxr generate` path, using the generic `--lora` CLI surface plus the
  family-local `qwen_image.scheduler_preset=lightning` extension
- first owned multi-image `Qwen-Image-Edit-2511` Lightning receipt through the
  same canonical runtime path, proving that the practical current `EditPlus`
  row is not limited to one image reference
- first heavyweight official-recipe `Qwen-Image-2512` generation receipts at
  `1664x928`, now clean across base `50-step`, LightX `8-step`, and Wuli
  `4-step` after fixing the shared large-image decode path with upstream-style
  Qwen VAE tiling
- first official-recipe `Qwen-Image-Edit-2511` base receipt through the
  canonical `mlxr generate --task image.edit` path at `1344x768 / 40 steps`
- first official-recipe `Qwen-Image-Edit-2511 Lightning` edit receipt through
  the same canonical path at `1344x768 / 4 steps`
- first owned generation backend for `Qwen-Image-2512`:
  scheduler contract, latent packing or unpacking helpers, config loaders, real
  prompt-conditioned generation, and runtime-managed output encoding
- default runtime-registry enablement
- runtime-managed `png` and `jpg` output artifacts
- trusted-local image and LoRA inputs through the canonical job/CLI surface
- family-local extension parsing for optional prompt-enhancement intent
- streamed trusted-local file import for large LoRA payloads on the runtime path

What is intentionally not claimed yet:

- promoted Gemini-reviewed quality receipts for every current generate and edit
  row; the real runtime receipts now exist, but not every promoted rung has a
  saved review sidecar yet
- first-class installed adapter identities for reusable LoRAs; current reuse is
  still the generic trusted-local `--lora` path
- `Qwen-Image-Layered`
- `Qwen-Image-2.0`
- hosted DashScope prompt enhancement as a required runtime dependency
- promoted real-run validation receipts for every supported row and profile

Primary rows to target next:

- `Qwen-Image-2512`
- `Qwen-Image-Edit-2511`

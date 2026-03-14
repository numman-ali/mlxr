# Qwen-Image Family Notes

Status: companion family note for rationale, pressure, and bring-up context

For the current repo truth by row, read
[qwen-image-capability-matrix.md](./qwen-image-capability-matrix.md)
together with this note.

## Recommendation

Current recommendation:

- treat `Qwen-Image-2512` as the primary current text-to-image row
- treat `Qwen-Image-Edit-2511` as the primary current editing row
- keep older `Qwen-Image` and `Qwen-Image-Edit-2509` as legacy-compatible
  rows, not the promoted defaults
- document `Qwen-Image-Layered` as a real released layered-output row, but keep
  it out of the first `MLXR` runtime slice because its RGBA layer output
  contract is materially different from plain `image.generate` / `image.edit`

## Why This Family Matters

`Qwen-Image` is the strongest current official family pressure for:

- bilingual text-heavy image generation
- image-to-image editing
- multi-image editing
- character-preserving variation workflows that can generate related keyframes
  before LTX video generation

That makes it the first serious answer to the current “generate related
start/end/keyframe stills” problem without pretending LTX image-conditioning is
an editing model.

## Truthful Upstream Surface

Current official rows with concrete local quick starts:

- `Qwen-Image-2512`
  - text-to-image
  - current best open-weight text-to-image row in the family
- `Qwen-Image`
  - earlier text-to-image row
- `Qwen-Image-Edit-2511`
  - image editing
  - multi-image editing through `QwenImageEditPlusPipeline`
  - additive edit LoRAs such as `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`
    belong on this same `image.edit` row rather than on a new task
- `Qwen-Image-Edit-2509`
  - earlier multi-image edit row
- `Qwen-Image-Edit`
  - older single-image edit row

Released but not yet concretely specified in the local repo surface:

- `Qwen-Image-Layered`

Released with a real upstream pipeline contract, but better treated as a later
`MLXR` tranche:

- `Qwen-Image-Layered`
  - layered RGBA decomposition output
  - different output contract from plain image generation or editing

Announced but not a current local bring-up target from this repo alone:

- `Qwen-Image-2.0`

## First Truthful MLXR Slice

The first truthful `MLXR` slice should be split in two:

1. `Qwen-Image-2512`
   - `image.generate`
   - prompt, negative prompt, width, height, steps, and the current shared
     generic guidance control
   - runtime-managed `png` and `jpg` outputs
2. `Qwen-Image-Edit-2511`
   - `image.edit`
   - one or more image references
   - prompt plus negative prompt
   - runtime-managed `png` and `jpg` outputs

What should stay out of the first promoted slice:

- `Layered`
- `2.0`
- README-only understanding or control claims without a concrete local runtime
  recipe
- upstream-specific scale knobs such as `true_cfg_scale` until they are mapped
  deliberately onto the shared or family-local runtime surface
- DashScope-backed prompt enhancement as a required inference dependency
- additive edit LoRAs such as `Multiple-Angles` until the base edit row is
  already promotion-strong and the LoRA has its own real receipts

## Current Repo Truth

What is real in the repo now:

- family inspection, conversion, workflow planning, and fail-closed execution
  boundaries are in place
- the first owned `Qwen-Image-2512` prompt-conditioning seam is real
- that prompt seam uses the official fixed system/user/assistant scaffold, a
  fixed `34`-token prefix drop, and an owned local `Qwen2.5-VL` text-only MLX
  substrate
- the family now advertises an explicit `prompt_encode -> generate ->
  encode_output` runtime stage order instead of hiding prompt work inside
  `generate`
- the first owned generation backend is now present under
  `qwen_image/_generation_backend/` with config parsing, scheduler behavior,
  latent pack or unpack helpers, real prompt-conditioned generation, and
  runtime-managed output encoding
- a real local prompt-encode smoke against the live `Qwen-Image-2512` text
  bundle produced `3584`-wide prompt embeddings as expected
- a real canonical `mlxr generate` smoke now produces a coherent `Qwen-Image-2512`
  PNG through the shared runtime and CLI path:
  `tmp/manual-runs/qwen2512-cli-smoke-sigmafix.png`
- a larger canonical `512x512` receipt is now also real through the same path:
  `tmp/manual-runs/qwen2512-cli-512.png`, with the `generate` stage taking about
  `266390 ms`
- the first heavyweight official-recipe base receipt is now real too:
  `tmp/showcase-runs/qwen-official-eval-20260311T160313Z/official_anime_base50_1664x928.png`
  at `1664x928 / 50 steps / guidance=4.0`; it completed through the canonical
  `mlxr` path with `generate` taking about `1335348 ms` and peak active memory
  about `85.64 GB`, which proves the owned base path really runs at the full
  official rung
- the shared full-resolution corruption bug is now fixed:
  `floating_city_base50_1664x928.png` at about `1253912 ms`,
  `floating_city_lightx8_1664x928.png` at about `151001 ms`,
  and `floating_city_wuli4_1664x928.png` at about `108543 ms`
  now all complete cleanly through the canonical `mlxr` path after adding
  upstream-style Qwen VAE tiled encode/decode behavior and proving the fix with
  a same-latent owned-vs-diffusers decode oracle
- the first owned decode oracle receipt is now
  `tmp/manual-runs/qwen-decode-oracle-20260311T190307Z/`; it saves the final
  latents plus owned, diffusers, and diffusers-tiled decodes for one shared
  comparison case and is the canonical way to debug future Qwen large-image
  regressions
- the first owned Lightning-backed receipt is now real too:
  `tmp/manual-runs/qwen2512-lightning-cli-512.png`, using the generic `--lora`
  path with the `lightx2v/Qwen-Image-2512-Lightning` `4`-step adapter and a
  family-local `scheduler_preset=lightning`; its `generate` stage took about
  `97186 ms`
- the first owned `Wuli-Art/Qwen-Image-2512-Turbo-LoRA` `V3.0` receipt is now
  also real on the same base model through the canonical runtime path:
  `tmp/manual-runs/qwen2512-wuli-v3-4step-wide-512x288.png`, using the generic
  LoRA reference surface plus `scheduler_preset=turbo_wuli`; its `generate`
  stage took about `85529 ms`
- the heavyweight Wuli generation lane is now also real and clean on the
  official wide rung:
  `tmp/showcase-runs/qwen-official-eval-20260311T190800Z/floating_city_wuli4_1664x928.png`
  completed at `1664x928 / 4 steps`, with `generate` taking about `108543 ms`
- the first owned `Qwen-Image-Edit-2511` backend slice is now real:
  multimodal prompt encoding through the processor path, edit-aware
  `zero_cond_t` transformer modulation, VAE encode plus decode, and
  conditioned latent concatenation all run through the owned MLX backend
- the first direct base edit receipt is now real too:
  `tmp/manual-runs/qwen-edit-direct-1step.png`; it proves the owned base edit
  path returns, but at `256x256 / 1 step` it took about `148289 ms` and
  produced a black frame, which only tells us the probe rung is too weak; it is
  not a fair official-recipe base-model verdict
- the first official-recipe base edit receipt is now real:
  `tmp/showcase-runs/qwen-official-eval-20260311T221605Z/kenji_edit_base40_1344x768.png`
  completed through the canonical `mlxr generate --task image.edit` path at
  `1344x768 / 40 steps / guidance=4.0`, with `generate` taking about
  `1574707 ms`; it is slow, but it is a real clean base-model edit receipt
- the first practical owned edit receipts are Lightning-backed:
  `tmp/manual-runs/qwen-edit-lightning-4step-wide.png` at about `79324 ms` and
  `tmp/manual-runs/qwen-edit-lightning-8step-wide.png` at about `117765 ms`,
  both at `512x288`
- the first canonical runtime and CLI edit receipt is now real:
  `tmp/manual-runs/qwen-edit-cli-lightning-4step-wide.png`, completed as job
  `job_7cb8a1fc7a6a4882a47589f7e03c46ab` with `generate` taking about
  `72566 ms`, prompt encode about `2388 ms`, and peak active memory about
  `54.18 GB`
- the first canonical multi-image edit receipt is now real too:
  `tmp/manual-runs/qwen-edit-cli-lightning-4step-multiref.png`, completed as
  job `job_8357e3b17a614d8089ce32752e7faa17` with `generate` taking about
  `124552 ms`, prompt encode about `1914 ms`, and peak active memory about
  `64.04 GB`; the output is semantically plausible rather than a collapsed
  failure, so the practical current `EditPlus` row is genuinely multi-image
- the first heavyweight Lightning edit comparison receipt is now also real:
  `tmp/showcase-runs/qwen-official-eval-20260311T221605Z/kenji_edit_lightning4_1344x768.png`
  completed at `1344x768 / 4 steps`, with `generate` taking about `96917 ms`
- `Wuli V3.0` required a real checkpoint-compatibility fix rather than a new
  backend: its LoRA uses diffusers-style `diffusion_model.*` keys with
  `lora_A` / `lora_B`, and it also targets modulation layers like
  `img_mod.1` / `txt_mod.1` that needed to alias onto the owned transformer
  names
- the most important bring-up fixes so far were:
  exact Qwen-image transformer RoPE parity, Qwen2.5-VL text-decoder parity,
  and correcting the owned scheduler update to operate in sigma space rather
  than `1000`-scaled timesteps
- reusable large LoRAs are now a clear product-shape pressure: the generic
  `--lora` surface is correct, but large safetensors adapters should move
  toward installed or registered adapter identities over time rather than
  living forever as repeated transient uploads

What is not yet true:

- complete review-backed promotion coverage for every current generate and edit
  row
- a separately defended promoted base multi-image edit row
- `Layered`
- `2.0`

So the repo is now past “contract only” for `Qwen-Image`, through the
full-resolution generation repair, and into real official-recipe edit
execution. The honest current boundary is:

- full-resolution base, LightX, and Wuli generation are now real and clean on
  the canonical `1664x928` wide rung
- base `image.edit` is now also real on the official recipe, but it is much
  slower than the practical Lightning lane
- Lightning-backed edit remains the fastest practical path we have actually
  demonstrated so far
- `qwen_official_eval.py` is the canonical repo-owned runner for heavy
  base-first Qwen evaluation through the real `mlxr` CLI path
- `qwen_decode_oracle.py` is the canonical debug oracle when the large-image
  decode path looks suspicious again
- wider Gemini review coverage, promoted base multi-image edit, and Layered
  remain open

## Important Platform Pressure

`Qwen-Image` pressures the platform differently from `Z-Image`:

- edit tasks need `image.edit` to be a real shared runtime task
- multi-image editing needs repeatable image references in one job
- layered decomposition would require a structured multi-artifact image output
  contract instead of a single flat raster artifact
- prompt enhancement exists, but as an optional hosted sidecar, not core model
  execution
- the family leans on Qwen multimodal text/processor components, which is a
  good proving case for the broader multimodal substrate plan

## Prompt Enhancement Truth

The official prompt-enhancement utilities are real, but they are not a core
checkpoint capability:

- text-to-image rewriting uses hosted DashScope text models
- edit-prompt polishing uses hosted DashScope multimodal models

`MLXR` should treat prompt enhancement as optional orchestration or host UX,
not as part of the native family runtime contract.

## Product Role

This family is the strongest current fit for:

- creating related stills from an existing reference image
- generating alternative keyframes while preserving subject identity better than
  plain text-only generation
- supplying edited or fused inputs into later LTX image-to-video or
  interpolation flows

## Sources

- `references/official/Qwen-Image/README.md`
- `references/official/Qwen-Image/src/examples/edit_demo.py`
- `references/official/Qwen-Image/src/examples/generate_w_prompt_enhance.py`
- `references/official/Qwen-Image/src/examples/tools/prompt_utils.py`
- `tmp/hf-qwen-image-2512-meta/model_index.json`
- `tmp/hf-qwen-image-edit-2511-meta/model_index.json`

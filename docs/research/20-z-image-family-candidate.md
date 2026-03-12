# Z-Image Family Candidate

Status: working research note for the first explicit image family now that the
`Z-Image-Turbo` runtime slice is real and the base `Z-Image` row is next.

## Purpose

This note exists to turn the current vague "one image diffusion family" slot
into a concrete candidate with official sources, real product pressure, and a
truthful first-slice plan.

Current recommendation:

- treat `Z-Image` and `Z-Image-Turbo` as the leading candidate for the first
  `MLXR` image family
- use that family to satisfy both the Phase C image-family validation basket
  and the official `ltx-desktop` product pressure for text-to-image support
- keep the promoted slice narrow: prompt-only text-to-image generation first

## Current Runtime Status

What is now real in `MLXR` for `Z-Image-Turbo`:

- source inspection for official diffusers-style bundles
- componentized portable-artifact conversion
- native Apple Silicon / MLX prompt-only still-image generation
- canonical `mlxr generate` support through the shared runtime job path
- runtime-managed `png` and `jpg` outputs
- shared runtime job telemetry plus env-gated family trace metadata for deeper
  denoise/decode benchmarking

What is now real for the released base `Z-Image` row:

- the same runtime-owned source registration and artifact conversion contract
- a first canonical `mlxr generate` smoke with negative prompt and nonzero
  guidance through the shared runtime job path
- family-local `cfg_normalization` and `cfg_truncation` controls through
  `extensions.z_image`, while keeping them out of the shared public CLI surface
- a first balanced quality rung with a coherent `768x768 / 28-step / cfg=4.0`
  receipt at
  `tmp/manual-runs/20260309T195155Z-zimage-base-balanced-rung/zimage_base_balanced.png`

What is still missing for the base row:

- any shared `cfg_normalization` surface
- benchmark-backed quality claims at the recommended higher-step rungs
- batch prompt generation in one runtime job

What is still missing even for `Turbo`:

- prompt-list batch inference in one job
- `num_images_per_prompt`
- latent-output mode
- product-surface compile / attention-backend / offload tuning knobs
- any claim that README-only Prompt Enhancer messaging is a covered runtime feature

The Prompt Enhancer note in the official README should currently be treated as a
project-level capability/story signal, not as a separately exposed open-source
runtime feature we can claim to have wired.

## Why Z-Image Is A Strong Candidate

Three things make `Z-Image` more useful than leaving the image slot abstract:

1. It is now a real official upstream family under
   `references/official/Z-Image/`.
2. The official desktop product already exposes `Z Image Turbo` as a separate
   text-to-image feature in API mode.
3. The family gives `MLXR` a concrete non-LTX generative workload that still
   connects directly to the LTX product story by generating stills for
   image-to-video workflows.

That combination is unusually valuable:

- platform validation wants a real image family
- desktop compatibility wants an image-generation story
- LTX product workflows benefit from a strong still-image source

## Official Family Surface

Current official upstream variants:

- `Z-Image-Turbo`
  - few-step distilled generation variant
  - text-to-image
  - optimized for very fast inference
- `Z-Image`
  - higher-quality foundation generation model
  - text-to-image
  - supports negative prompts and broader controllability
- `Z-Image-Omni-Base`
  - foundation checkpoint spanning generation and editing tasks
  - intended as a rawer fine-tuning and downstream base
- `Z-Image-Edit`
  - editing-oriented variant

Current official ecosystem signal:

- upstream says `diffusers` support has been merged and current examples use
  `ZImagePipeline`

## What MLXR Should And Should Not Claim

The first truthful promoted `MLXR` image-family slice should claim only:

- text-to-image generation
- runtime-managed `png` and `jpg` outputs
- prompt plus standard inference parameters
- provenance-preserving source registration and artifact conversion
- shared runtime stage telemetry and family-local trace metadata for validation

It should not claim yet:

- image editing
- ControlNet-style controls
- omni-base editing support
- benchmark-backed Apple-Silicon performance
- desktop parity beyond the specific rows we actually wire
- README-only Prompt Enhancer behavior as a distinct runtime feature
- family-local base-model knobs as shared cross-family product-surface controls

## First-Slice Product Role

The first user-visible product role for `Z-Image` in `MLXR` should be:

- generate still images locally
- export them as runtime-managed artifacts
- allow those artifacts to feed later `video.condition.image` workflows

That is a clean story because it avoids inventing a fake shared "image assist"
concept. It is simply:

- one real image family
- one real image artifact output
- one real downstream use in LTX image-to-video

## Why This Helps The Platform

`Z-Image` pressures different seams than LTX:

- image-only scheduler class
- still-image encode and export
- image-family capability schema
- different checkpoint and artifact assumptions
- potential negative-prompt handling on a non-LTX family

That makes it a better Phase C validation family than continuing to stretch the
platform only around LTX-specific shapes.

## Desktop Implication

The official desktop repo currently treats `Z Image Turbo` as an API-backed
text-to-image feature, separate from the local LTX video-generation path.

Current `MLXR` implication:

- the first local desktop image-generation story does not need to clone the
  exact current API dependency
- but it should acknowledge that the desktop product already has an image
  generation surface users will expect to keep
- a local `Z-Image` family would let `MLXR` eventually replace that API-backed
  image feature with a runtime-owned path

## Recommended Bring-Up Order

1. document the truthful `Z-Image-Turbo` landed slice and explicit missing
   parity items
2. bring up the released base `Z-Image` row on the same runtime-owned surface
3. validate negative-prompt and higher-step base-model behavior honestly
4. connect exported images into downstream `video.condition.image` workflows
5. decide whether prompt-list batch generation becomes a shared runtime feature
   or stays a host/client wrapper
6. only then evaluate editing-oriented rows such as `Z-Image-Edit`

Current measured caution:

- the balanced `768x768 / 28-step` base rung is now development-loop viable
  enough to use for real receipts
- the attempted `1024x1024 / 36-step` base rung stayed outside a quick local
  validation budget on this machine, so larger recommended-profile claims still
  need deliberate benchmark work instead of optimism

## Source References

Primary references:

- `references/official/Z-Image/README.md`
- `references/official/ltx-desktop/README.md`
- `references/official/ltx-desktop/backend/_routes/image_gen.py`
- `references/official/ltx-desktop/frontend/hooks/use-generation.ts`

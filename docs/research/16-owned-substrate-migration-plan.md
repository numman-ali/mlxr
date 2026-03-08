# Owned Substrate Migration Plan

## Purpose

This document is the canonical recovery and migration plan for the `LTX` family
engine.

Its job is to keep three truths aligned:

1. what `MLXR` currently and truthfully supports
2. which unofficial donor runtimes still sit in the execution path
3. the order in which those donors get removed without breaking the current
   promoted slice

This is not a speculative design note. It is the implementation order for
bringing `LTX` back onto a repo-owned substrate.

Use it together with:

- [18-ltx-reference-map.md](18-ltx-reference-map.md)
- [19-ltx-compatibility-checklist.md](19-ltx-compatibility-checklist.md)
- [17-multimodal-inference-engine.md](17-multimodal-inference-engine.md)

## Current Truth

Today the only promoted real engine is the distilled proving slice:

- distilled two-stage text-to-video
- distilled two-stage image-to-video
- audio-bearing output on the AV path
- a narrow `video.condition.audio` preserved-reference slice

That promoted slice is real, but it still depends on unofficial donor code at
runtime:

- the generation engine still imports `mlx_video.*` through the current helper
  stack
- the prompt-encode path still depends on `mlx-vlm` / `mlx-lm` for the current
  Gemma runtime surface

Those dependencies are not compatible with the long-term repo direction. The
next engine work must reduce and then remove them instead of layering more
capability claims on top.

## Progress Snapshot

The donor-removal program is underway, but still in progress.

Already moved onto repo-owned code in the promoted distilled path:

- latent upsampling helper
- audio decoder loader path
- VAE encoder loader path
- x2 spatial upsampler class and loader path
- video unpatchify helper
- decoder-side temporal/spatial tiling config and tile blending logic
- decoder-side convolution, timestep embedding, residual block-group, and
  depth-to-space upsampling blocks
- audio-VAE and vocoder weight sanitizers
- RMS norm helper used by the patched transformer bridge

Still donor-backed in the promoted distilled path:

- transformer construction and patched bridge
- VAE encoder class and remaining video-VAE loader substrate
- remaining audio runtime classes and shared helper imports
- donor checkout import indirection through the current helper stack

This means the current promoted slice is healthier than before, but it is not
yet donor-free.

## Migration Rules

- Do not build new promoted capability rows on top of donor runtime code you
  already intend to delete.
- Do not promote non-distilled standard or HQ support until the code is green,
  the engine path is real, and the runtime does not silently fall back to
  preview.
- Keep the current distilled showcase and capability story truthful until the
  replacement path clears the same validation bar.
- Preserve research and useful code structure from donor-based experiments, but
  do not preserve broken or over-promoted implementation states.

## Required Order

### 1. Restore and protect the green baseline

- Keep the current distilled slice green under the narrow runtime tests and full
  `verify`.
- Keep the docs honest about what is promoted versus merely planned.
- Archive failed experiments outside repo-tracked state if they contain useful
  research or code ideas.

### 2. Remove `mlx-video` from the current distilled execution path

This is the first donor-removal wave.

Move the current promoted slice onto repo-owned modules under
`ltx/_generation_backend/` by replacing donor runtime imports in this order:

1. runtime utilities and simple math:
   - sigma schedules
   - latent conditioning state
   - denoise-mask application
   - position grids
   - image loading and image-prep helpers
   - velocity-to-denoised conversion
   - video unpatchify and decoder-side tiling helpers
2. model loaders and runtime wrappers:
   - transformer loader and bridge
   - VAE encoder/decoder loaders
   - x2 upsampler loader
   - audio decoder / vocoder / BWE loader
3. backend identity and receipts:
   - no runtime backend names should continue to say `mlx_video_*`
   - no runtime helper should import the donor checkout through `sys.path`

The reference checkout may stay under `references/ecosystem/` as a frozen code
reference, but it must stop being part of the execution path.

### 3. Re-land non-distilled standard two-stage on the owned engine

Only after the distilled slice is donor-free:

- support the `dev` checkpoint explicitly
- require the distilled LoRA contract for non-distilled rows
- make standard two-stage the default non-distilled path
- keep HQ opt-in and unpromoted until it is validated
- fail hard if a requested non-distilled path is unavailable; do not preview
  fallback

### 4. Re-land HQ on the owned engine

HQ is the second non-distilled tranche, not a metadata toggle.

It depends on:

- a real owned scheduler for the full checkpoint path
- a correct owned second-order sampler
- owned guidance substrate
- correct noise-key evolution
- explicit variant selection

### 5. Remove `mlx-vlm` / `mlx-lm` from the Gemma prompt path

This is the second donor-removal wave.

Keep the current Gemma contract and LTX prompt V2 semantics, but replace the
runtime helper dependency with a repo-owned minimal Gemma path that serves LTX
only.

The architecture and layering for that replacement should follow
[17-multimodal-inference-engine.md](17-multimodal-inference-engine.md):

- reusable LM and multimodal execution logic belongs in `packages/core/`
- family adapters should keep family truth, not become the hidden home of a
  shared multimodal engine
- donor libraries remain references and research inputs, not runtime
  dependencies or copy-paste implementation targets

This phase is about substrate ownership, not about changing the text-encoder
model.

## What Counts As Done

### Distilled donor-removal tranche

- no runtime import of `mlx_video.*` remains in the promoted distilled path
- current distilled smoke and showcase receipts still pass
- backend receipts use `MLXR`-owned names
- full `verify` remains green

### Non-distilled standard tranche

- dev checkpoint detection is real
- distilled LoRA is required where the official path requires it
- standard two-stage is reachable through the actual request path
- no preview fallback exists for that row
- safe-rung and coherence-rung receipts exist

### HQ tranche

- HQ is explicitly selectable
- HQ validation uses the real engine and not preview
- HQ docs stay unpromoted until real receipts exist

### Prompt-substrate tranche

- no runtime dependency on `mlx-vlm` / `mlx-lm` remains
- current prompt-encode outputs and negative-prompt behavior stay intact
- current real distilled receipts still pass after the swap

## Relationship To The Capability Matrix

The capability matrix remains the user-facing truth map.

This migration plan is the engine-ownership plan underneath it.

Until these migration phases are complete:

- the distilled slice remains the only promoted generation engine
- non-distilled standard and HQ remain planned or in-progress, not promoted
- showcase replacement work does not outrank donor-runtime removal

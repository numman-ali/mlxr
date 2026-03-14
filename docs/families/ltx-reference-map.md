# LTX Reference Map

## Purpose

This document is the canonical map from each `LTX-2.3` subsystem in `MLXR` to
the highest-authority external reference we should use when implementing or
validating it.

Its job is simple:

- stop important upstream features from being missed
- stop donor repos from quietly becoming runtime dependencies again
- make it obvious which reference is authoritative for each part of the stack

Use this together with:

- [11-ltx-capability-matrix.md](11-ltx-capability-matrix.md)
- [16-owned-substrate-migration-plan.md](16-owned-substrate-migration-plan.md)
- [17-multimodal-inference-engine.md](17-multimodal-inference-engine.md)
- [19-ltx-compatibility-checklist.md](19-ltx-compatibility-checklist.md)

## Reference precedence

When references disagree, use this precedence:

1. current `MLXR` code and promoted repo truth
2. official `LTX-2` references under `references/official/LTX-2/`
3. official product-facing LTX repos such as `ltx-desktop`
4. official training and utility docs in `LTX-2`
5. ecosystem repos under `references/ecosystem/`

Important rule:

- `references/ecosystem/mlx-video/` and `references/ecosystem/mlx-vlm/` are
  research inputs only
- they are not the desired runtime architecture
- they are not templates to copy mechanically

## Subsystem map

### Family surface and pipeline canon

Authoritative references:

- `references/official/LTX-2/README.md`
- `references/official/LTX-2/packages/ltx-pipelines/README.md`

Use these for:

- the official list of public pipeline families
- the intended production-default versus speed-first paths
- the official meaning of standard two-stage, HQ, one-stage, distilled,
  audio-to-video, interpolation, retake, and IC-LoRA
- the current public asset set and checkpoint names

Current repo implication:

- the capability matrix must not omit any pipeline row from these docs
- `MLXR` should continue to treat `TI2VidTwoStagesPipeline` as the next
  production-quality target after the promoted distilled slice

### Prompting and simple-product guidance

Authoritative references:

- `references/official/LTX-2/README.md`
- `references/official/ltx-desktop/frontend/views/GenSpace.tsx`
- `references/official/ltx-desktop/frontend/components/SettingsModal.tsx`
- `references/official/ComfyUI-LTXVideo/README.md`

Use these for:

- text-first prompting rules
- product-facing prompt enhancement posture
- default UX assumptions about one prompt plus optional refs
- where the official surfaces separate text-first from stronger conditioned paths

Current repo implication:

- the default `MLXR` story remains one main prompt plus optional references
- internal intent splitting is allowed, but the default UI should not expose a
  compulsory audio-prompt and video-prompt split

### Desktop compatibility contract

Authoritative references:

- `references/official/ltx-desktop/backend/_routes/generation.py`
- `references/official/ltx-desktop/backend/api_types.py`
- `references/official/ltx-desktop/frontend/hooks/use-generation.ts`
- `references/official/ltx-desktop/backend/runtime_config/runtime_policy.py`

Use these for:

- the current desktop backend HTTP contract
- snake_case versus camelCase response-shape expectations
- trusted-local path expectations at the host seam
- the current Darwin API-only policy the runtime is meant to replace

Current repo implication:

- `packages/adapters/ltx-desktop/` should preserve desktop compatibility as a
  thin host adapter
- desktop path-passing should remain a host-local compatibility seam, not the
  generic runtime HTTP contract
- local desktop support should be described separately from the broader `LTX`
  capability backlog

### Scheduler and sigma schedules

Authoritative references:

- `references/official/LTX-2/packages/ltx-core/src/ltx_core/components/schedulers.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/distilled.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/retake.py`

Use these for:

- `LTX2Scheduler`
- the distilled fixed-sigma path
- token-count-aware schedules for the full checkpoint family
- which rows use which scheduler

Current repo implication:

- a full-checkpoint row is not real until it uses the owned non-distilled
  scheduler path rather than the current distilled fixed-sigma shortcut

### Samplers and denoising loops

Authoritative references:

- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/res2s.py`
- `references/official/LTX-2/packages/ltx-core/src/ltx_core/components/diffusion_steps.py`

Use these for:

- standard guided Euler
- HQ `res_2s`
- SDE noise handling
- evolving noise-key expectations
- gradient-estimation loops if we choose to adopt them later

Current repo implication:

- HQ cannot be promoted until the owned sampler path is correct and validated
- repeated or fixed noise-key reuse should be treated as an algorithmic bug, not
  as an implementation detail

### Guidance and perturbations

Authoritative references:

- `references/official/LTX-2/packages/ltx-core/src/ltx_core/components/guiders.py`
- `references/official/LTX-2/packages/ltx-core/src/ltx_core/guidance/perturbations.py`
- `references/official/LTX-2/packages/ltx-pipelines/README.md`

Use these for:

- CFG
- STG
- modality isolation
- rescaling
- guider factories and sigma-dependent guidance policy

Current repo implication:

- advanced guidance controls should not be promoted until they are real on the
  owned engine
- user-facing STG claims should follow real request-path wiring, not internal
  experiments

### Conditioning semantics

Authoritative references:

- `references/official/LTX-2/packages/ltx-pipelines/README.md`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/retake.py`
- `references/official/LTX-2/packages/ltx-core/README.md`

Use these for:

- image conditioning by latent replacement
- keyframe-guided conditioning
- reference-video conditioning
- audio-driven conditioning
- retake / temporal masking semantics
- what start-frame and end-frame ideas map to upstream rows versus what would be
  a made-up product layer

Current repo implication:

- combined text + image + audio is in scope only as an explicit upstream-aligned
  row
- start/end frame support should map truthfully to interpolation, retake, or
  keyframe semantics instead of being invented as a vague extra feature
- `video.condition.video` should be treated as the official `ICLoraPipeline`
  reference-video row
- `video.retake` should be treated as the separate official `RetakePipeline`
  row

### Asset contract

Authoritative references:

- `references/official/LTX-2/README.md`
- `references/official/LTX-2/packages/ltx-pipelines/README.md`

Use these for:

- distilled checkpoint
- dev checkpoint
- Gemma text encoder
- x2 upsampler
- x1.5 upsampler
- temporal upscaler
- distilled LoRA
- IC-LoRA family

Current repo implication:

- standard and HQ rows are not truthful unless the required distilled LoRA is
  present
- x1.5 and temporal upscaler remain planned until the corresponding upstream
  rows are actually implemented

### Audio stack

Authoritative references:

- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py`
- `references/official/LTX-2/packages/ltx-core/README.md`
- official checkpoint asset metadata

Use these for:

- text-to-audio-video behavior
- audio-to-video behavior
- guidance expectations on the audio branch
- the role of preserved reference audio versus generated audio
- current checkpoint-backed BWE behavior

Current repo implication:

- `video.condition.audio` should be described as a real preserved-reference
  slice until conditioned-scene quality clears the promotion bar
- audio review must remain part of semantic validation, not an afterthought

### LoRA support

Authoritative references:

- `references/official/LTX-2/README.md`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py`
- `references/official/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py`

Use these for:

- required internal distilled LoRA staging for full-checkpoint two-stage rows
- future user-facing IC-LoRA and control-LoRA behavior

Current repo implication:

- required internal distilled LoRA and user-facing LoRA control are different
  things and must not be conflated
- user LoRA refs should continue to fail closed until the actual row exists

### Prompt-substrate ownership

Authoritative references:

- `docs/research/17-multimodal-inference-engine.md`
- `references/official/mlx-lm/`
- `references/ecosystem/mlx-vlm/`

Use these for:

- how to remove `mlx-vlm` / `mlx-lm` from runtime execution
- what reusable LM/multimodal engine logic should live in `packages/core/`
- what should stay family-local in LTX

Current repo implication:

- do not transplant `mlx-vlm` or `mlx-lm` structure into the LTX family
- build a repo-owned shared MLX substrate in core and let LTX consume it

### Host UX and workflow boundaries

Authoritative references:

- `references/official/ltx-desktop/`
- `references/official/ComfyUI-LTXVideo/`
- `docs/workflow-orchestration-design.md`
- `docs/research/17-multimodal-inference-engine.md`

Use these for:

- what belongs in the simple first-party UX
- what belongs in thin host adapters
- what belongs in workflow planning
- what remains family truth

Current repo implication:

- `mlxr` stays the first trusted product surface
- workflow remains documented as planning until core truly owns stage execution
- desktop and Comfy should stay thin over the shared runtime instead of
  re-embedding inference logic

## Practical implementation rules

When implementing against these references:

- prefer official LTX references for feature truth
- use ecosystem repos only as algorithm and code-comparison input
- do not copy donor structure mechanically
- improve the code to match `MLXR`’s layering, typing, and validation standard
- update the capability matrix and compatibility checklist in the same tranche if
  repo truth changed

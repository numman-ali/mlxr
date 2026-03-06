# LTX Integration Seams

## Purpose

This file records where LTX is pressuring the platform and where those seams are universal versus LTX-specific.

The goal is not to clone the upstream architecture. The goal is to understand which parts should become:

- provider logic
- family-adapter logic
- scheduler stages
- output-path work
- host-adapter work

## LTX Runtime Repo Seams

### `packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py`

What it shows:

- LTX is a coordinated multi-model load, not a single checkpoint load
- the stack includes transformer, video VAE, audio encoder or decoder, vocoder, embeddings processor, text encoder, and optional upsampler
- builders can share a registry, but model objects are not cached by default

Universal seam:

- a family adapter must be able to describe multi-component artifacts and staged load or unload behavior

LTX-specific pressure:

- audio and video branches share one family but not one simple memory profile

### `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py`

What it shows:

- metadata preflight from safetensors is already part of upstream loading
- registry-backed state-dict reuse exists
- LoRA loading uses CPU as a deliberate pressure valve before device transfer

Universal seam:

- conversion and load plans should support metadata-first inspection, multi-file manifests, and staged load devices

### `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`

What it shows:

- prompt encoding is explicitly stage-scoped and followed by cleanup
- image conditioning still assumes path-based local file inputs
- stage cleanup is already part of the upstream mental model

Universal seam:

- the runtime should preserve prompt-encode and conditioning as separate stages
- generic HTTP clients need handle-based imports in place of raw file paths

### `packages/ltx-pipelines/src/ltx_pipelines/distilled.py`

What it shows:

- the fast path is explicitly two-stage
- stage 1 and stage 2 have different sigma schedules
- stage 2 depends on stage 1 outputs and upsampler assets
- audio decode already exists in the fast path return shape

Universal seam:

- stage graphs need first-class scheduler support
- portable artifacts need to declare related assets and optional stages

### `packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py`

What it shows:

- current encode or mux path is PyAV and ndarray-heavy
- output encode is a real stage, not a footnote

Universal seam:

- output encode time must be benchmarked separately
- Apple-native output-path work can create runtime value outside the denoise loop itself

## LTX Constraint Propagation

LTX is useful because it forces the runtime to surface constraints honestly.

Current upstream signals include:

- width and height as multiples of `32`
- frame count formulas such as `8k+1`
- staged fast-path defaults around `768x512` stage-1 and `121` frames at `24 fps`
- image-conditioning frame indices and strengths
- audio duration tied to the video duration in audio-aware flows

Platform implication:

- these must live in the capability schema and request validation path
- hosts should not hardcode them from README text

## LTX Desktop Seams

### `backend/runtime_config/runtime_policy.py`

The current desktop product still forces API generations on Darwin. That is the clearest product gap this runtime is meant to close.

### `backend/services/fast_video_pipeline/fast_video_pipeline.py`

The current local pipeline protocol is path-based:

- `checkpoint_path`
- `gemma_root`
- `upsampler_path`
- local image paths
- local output path

That is fine for a trusted local implementation. It is not the right generic HTTP contract.

### `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`

Current behavior:

- wraps the upstream `DistilledPipeline`
- maps local paths directly into the pipeline
- uses a local file output path
- still contains a `torch.compile` helper even though desktop skips it on MPS

Platform implication:

- the adapter should move from path-passing to source registration, imported handles, and runtime-managed artifacts

### `backend/handlers/pipelines_handler.py`

Current behavior:

- desktop manages local warm and cold pipeline state itself
- `torch.compile` is skipped on MPS
- image generation (`ZIT`) and video generation have separate lifecycle behavior

Platform implication:

- the shared runtime should own pipeline lifecycle and admission logic
- the desktop adapter should stop being the scheduler

### `backend/runtime_config/model_download_specs.py`

Current behavior:

- explicit download specs already exist for checkpoint, upsampler, Gemma text encoder, and `ZIT`
- text encoder requirement changes depending on API versus local policy

Platform implication:

- provider and provenance logic should absorb this into a general source model instead of leaving it as desktop-local policy

## Current First-Slice Runtime Artifact Shape

The current runtime-backed LTX slice now includes truthful artifact conversion plus the first real execution stage, but it is still not generation-complete.

What the repo now treats as canonical for the first truthful fast-path slice:

- required source roles are `checkpoint`, `spatial_upsampler`, and `text_encoder`
- the checkpoint and x2 spatial upsampler come from `Lightricks/LTX-2.3`
- the text encoder comes from `google/gemma-3-12b-it-qat-q4_0-unquantized`
- a single local bundle is still allowed, but the real family contract also accepts explicit multi-source role bindings

Portable artifact implications:

- portable artifact payloads are copied under `payload/<role>/...`
- each artifact records typed components with role, kind, relative path, source id, resolved ref, component digest, size, and provenance
- artifact identity comes from copied component contents plus conversion settings, not provider-cache paths
- load-time validation now checks that the required payload roles are present and internally consistent before stage execution begins

Execution implications:

- `prompt_encode` now runs through a repo-owned strict-local MLX Gemma path using the artifactized `checkpoint` and `text_encoder` payloads
- prompt context stays worker-local and becomes the honest seam into later denoise/generate work
- negative-prompt support is still intentionally excluded for the fast path instead of being silently ignored

Still intentionally out of scope for this slice:

- temporal upsamplers, x1.5 upsampler, LoRAs, audio-aware branches, and prompt enhancement
- real denoise/video generation
- claiming that the LTX artifact shape has already validated the platform across families

## Comfy Seams

### Current baseline reality

ComfyUI now has LTX support in core. `ComfyUI-LTXVideo` is now the advanced-node and workflow layer.

### `ComfyUI-LTXVideo/__init__.py`

What it shows:

- the add-on exposes advanced nodes such as tiled sampling, low-VRAM loaders, prompt enhancement, STG, IC-LoRA helpers, conditioning save or load, and prompt encoder helpers

Platform implication:

- a runtime-backed Comfy integration should target advanced control and long-running execution value, not just “basic T2V and I2V are available”

### `ComfyUI-LTXVideo/nodes_registry.py`

What it shows:

- the add-on is heavily node-schema-driven and expects rich node-level behaviors

Platform implication:

- the best runtime-backed surface may be an execution bridge plus advanced nodes that map cleanly onto runtime handles, conditioning artifacts, and job state

## Universalizable Seams Vs LTX-Only Seams

| Seam | Universalizable? | Why |
| --- | --- | --- |
| provider resolution and provenance | yes | all families need it |
| portable artifact manifests | yes | all families need it |
| stage-aware scheduler | yes | LTX pressures it hardest, but it is not LTX-only |
| handle-based image or audio imports | yes | applies beyond LTX |
| PyAV output replacement with Apple-native path | mostly yes | value extends to other media families |
| Gemma-specific prompt encoding | no | LTX-specific or family-specific |
| IC-LoRA and LTX guider semantics | no | family extensions belong in the extension namespace |

## Recommended First Runtime-Backed LTX Targets

### 1. Desktop fast path on macOS

Replace Darwin forced API mode with a thin runtime client that:

- resolves or downloads LTX assets through the shared provider model
- submits runtime-native jobs
- consumes runtime-managed artifacts

### 2. LTX artifact conversion and registration

Support the current LTX asset bundle as a family-aware portable artifact with explicit related assets and policy state.

### 3. Advanced Comfy execution bridge

Target advanced value:

- conditioning save or load handles
- long-running job orchestration
- advanced IC-LoRA or control workflows
- clearer scheduler and artifact ownership

## Biggest Risks

- local Gemma text encoder footprint is large enough to distort naive desktop assumptions
- audio branch parity cannot be ignored just because T2V and I2V land first
- current output encode path likely leaves Apple-specific performance on the table
- advanced Comfy flows can easily bypass provenance and artifact rules if the bridge is too thin

# Upstream Baseline

Last refreshed: 2026-03-06

This file records the current upstream reality the platform docs are built around. It is not a marketing overview. It exists to stop the design from drifting away from what the official repos and docs actually say today.

## Pinned Local Repos

| Repo | Local head | Last upstream commit seen locally | Why it matters |
| --- | --- | --- | --- |
| `references/official/mlx` | `be872ebd` | `2026-03-05` | official MLX core and docs baseline |
| `references/official/mlx-c` | `1370f59` | `2026-03-05` | official C bridge and language-extension seam |
| `references/official/mlx-swift` | `a2f0d76` | `2026-02-23` | official Swift host-side surface |
| `references/official/LTX-2` | `9e8a28e` | `2026-03-05` | current official LTX runtime/inference monorepo |
| `references/official/ltx-desktop` | `32589e6` | `2026-03-05` | current official desktop product and macOS gap |
| `references/official/ComfyUI-LTXVideo` | `82bd963` | `2026-02-11` | official advanced Comfy nodes and workflows |
| `references/ecosystem/mlx-vlm` | `1ad8937` | `2026-03-05` | active VLM/omni serving reference |
| `references/ecosystem/mlx-audio` | `1da513b` | `2026-03-05` | active audio/TTS/STT serving reference |
| `references/ecosystem/vllm-metal` | `e0f97be` | `2026-03-05` | text-serving and Metal-oriented serving contrast |

Additional remote-only references verified on 2026-03-06:

- `ml-explore/mlx-swift-examples` at remote `HEAD c6844888`
- `ml-explore/mlx-swift-lm` at remote `HEAD 6bb84aac`

Those are not cloned locally today, but they are current inputs to the language and embedding story.

## MLX Core Reality

Freshness note:

- the published docs are currently at `MLX 0.31.0`
- the latest tagged core release is still `v0.30.6`

Design consequence:

- architecture decisions should be pinned to tested versions, not just the most recent docs snapshot

### Compile and export

Current official MLX docs still justify caution:

- `mx.compile` helps on stable inner loops, but shape changes can still trigger recompilation.
- shapeless compilation exists, but the docs still warn about graphs whose behavior depends on shape.
- compile assumes purity; orchestration with side effects is not the right unit to compile.
- exporter/importer remains explicitly experimental, so exported compiled state is not a safe portability layer.

Architecture consequences:

- compile belongs in the worker execution and machine-local build-cache story
- exported compiled state does not belong in the portable artifact contract
- the runtime must measure compile count and cache hit rate instead of assuming compile is a free win

### Extensions and custom Metal

Official MLX docs still expose the right low-level seams:

- custom extensions
- custom primitives
- custom Metal kernels

That means the runtime should reserve an explicit hotspot seam now instead of pretending native work is only a late emergency response.

### Streams and memory telemetry

MLX streams are real execution primitives, but they do not remove the need for a scheduler.

Current docs and APIs support:

- explicit stream usage
- Metal memory telemetry APIs such as active memory and memory limits
- Apple Silicon unified memory rather than a discrete VRAM mental model

Recent release notes also show the substrate is still moving, including newer accelerator support on recent Apple silicon generations. That is another reason not to freeze the architecture around one static reading of MLX internals.

Architecture consequences:

- streams stay in the worker design
- admission control must use live telemetry, not just adapter guesses

## MLX Language Surfaces

### Python

Python is still the real family bring-up center of gravity.

That remains true across:

- official MLX examples
- most active MLX ecosystem packages
- current conversion and provider tooling

### C

`mlx-c` is an official bridge layer. Its README still describes it as a bridge for other languages and explicitly notes that MLX Swift uses it.

Architecture consequence:

- C is a good bridge and ABI seam
- it is not a reason to move the runtime core out of Python

### Swift

The old docs understated Swift.

Current evidence says:

- `mlx-swift` is active
- the README says all MLX Python capabilities should be available in Swift
- the repo documents framework-style embedding concerns and Xcode-first build flows
- the README explicitly warns that command-line SwiftPM cannot build the Metal shaders, so real production builds still route through Xcode
- the examples ecosystem and `mlx-swift-lm` prove there is a meaningful Apple-native host path already

Architecture consequence:

- Swift should move earlier for host SDK, launcher, and embedded first-party access
- Swift still should not own v1 family bring-up

## LTX 2.3 Current Reality

### Official runtime repo

`LTX-2` is now the official runtime/inference repo. Its current README points directly to `LTX-2.3` weights and documents the current pipelines, including:

- two-stage T2V and I2V
- one-stage generation
- distilled fast path
- IC-LoRA
- keyframe interpolation
- audio-to-video
- retake

That is materially broader than the earlier docs captured.

### Official model card and files

As of 2026-03-06:

- `Lightricks/LTX-2.3` is live on Hugging Face
- the model card still describes `Diffusers` support as “coming soon”
- the repo uses an `ltx-2-community-license-agreement`, not a generic OSS label
- the main artifacts include `22b dev`, `22b distilled`, spatial upscalers, temporal upscaler, and the distilled LoRA
- the `fp8` repo exists, but current public evidence only clearly shows the `22b-dev-fp8` weight; the distilled `fp8` path should not be treated as available without re-checking

Architecture consequences:

- source and policy docs must preserve the actual license and access state
- LTX artifact planning must model multiple related assets, not a single checkpoint file
- the desktop and Comfy docs must stop assuming Diffusers is already the official path

## LTX Desktop Current Reality

`ltx-desktop` is important because it captures the current product gap.

Current repo reality:

- the backend still has `decide_force_api_generations(system="Darwin", ...) -> True`
- the app treats macOS as API-only
- local desktop pipelines are still direct-path and local-process shaped
- current download specs target `LTX-2.3` assets and local Gemma encoding when API mode is not used

Architecture consequences:

- the desktop adapter should replace the Darwin `force API` branch with local runtime support
- desktop path-based assumptions should move into trusted local import helpers, not the universal HTTP contract

## Comfy And LTX Current Reality

The earlier Comfy strategy was stale.

Current reality on 2026-03-06:

- `ComfyUI-LTXVideo` README says `LTX-2` is built into ComfyUI core
- the extension repo is now an advanced-node and workflow pack, not the only route to LTX support
- current Comfy docs describe native LTX support in core
- current Comfy stable releases include LTX2-specific fixes for audio and vocoder paths

Architecture consequences:

- baseline “runtime-backed T2V/I2V proxy nodes” are not the best first Comfy value
- the better value surface is advanced execution, constraint-aware flows, conditioning handles, and long-running local job management

## Active MLX Ecosystem Packages

Current community packages are useful reference inputs, not architecture authorities.

### `mlx-vlm`

- active and broad
- useful for multimodal processor packaging and serving ergonomics
- still strongly shaped by text-serving and OpenAI-compatible API assumptions

### `mlx-audio`

- active and broad across TTS, STT, and STS
- useful for audio model packaging, quantization, and server UX
- also shaped by a text-style serving surface and should not define the core multimedia API

### `vllm-metal`

- useful contrast for text-serving performance and Metal integration work
- not a direct template for video or multimodal runtime design

## Stale Or Still-Uncertain Areas

- exact MLX exporter/importer compatibility across versions should still be treated as unstable
- exact LTX 2.3 low-memory behavior on Apple Silicon remains unmeasured
- exact Comfy advanced-node value surface for a local runtime should be benchmarked with real workflows
- exact Swift embedding shape for a first-party host is still a design choice, not a measured result

## Immediate Design Consequences

- keep Python first for workers and family bring-up
- move Swift earlier at the host boundary
- split portable artifacts from machine-local build cache
- move provider and provenance into first-class docs
- refresh Comfy strategy around advanced value, not baseline support
- keep LTX first, but stop letting it freeze the universal runtime before cross-family validation

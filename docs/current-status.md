# MLXR Current Status

Last updated: 2026-03-14

## Snapshot

`MLXR` is already a real local runtime on Apple Silicon, but it is not yet a
fully polished public product.

What is real today:

- one shared runtime with provider, provenance, artifact, and job models
- a real first-party CLI over the local daemon
- real LTX, Qwen-Image, FLUX.2, and Z-Image family slices
- a growing cross-family capability story
- an early buildable first-party Mac app package over the same runtime

What is not true yet:

- frozen public API
- finished desktop adapter support
- a released polished first-party Mac app
- a fully green open-source release posture

## Surface Status

| Surface | Current status | Notes |
| --- | --- | --- |
| Shared runtime | real | canonical source of truth |
| Local daemon API | real | UDS-first, handle-based |
| `mlxr` CLI | real | first-party thin client with `generate`, `serve`, `doctor`, `models`, and `feedback` |
| `ltx-desktop` adapter | planned | compatibility seam, not yet implemented |
| Comfy adapter | planned | thin runtime-backed adapter, not yet implemented |
| First-party Mac app | early implementation | buildable native Swift package over the shared runtime, now with a persistent runtime-led `Studio`, reusable asset-first `Library`, rail-based `Activity`, guided model installs, and a real dev `.app` bundle path for macOS/Peekaboo testing |
| Embedded first-party access | planned later | Swift-side host boundary work |

## Family Truth

| Family | Current promoted truth | Current supported but unpromoted truth | Still blocked or later |
| --- | --- | --- | --- |
| LTX-2.3 | distilled `video.generate`, distilled `video.condition.image`, conditioned-audio, audio-bearing outputs | `one_stage`, `two_stage`, `two_stage_hq`, retake, interpolation, broader IC-LoRA control rows | fully promoted `dev` family closure, desktop compatibility adapter |
| Z-Image | prompt-only text-to-image generation | higher-rung quality and broader product polish | editing, ControlNet-style claims, README-only prompt enhancement |
| Qwen-Image | base generate, LightX fast generate, Wuli fast generate | base edit, Lightning edit, older rows | Layered, 2.0, additive edit LoRA lanes, prompt enhancement as runtime feature |
| FLUX.2 | `klein-4b` and `klein-9b` `image.generate`, single-reference `image.edit` | multi-reference edit, base rows, KV row, broader review coverage | `dev`, LoRA loading, prompt upsampling, quantized owned execution |

## Recommended Current Defaults

If someone asks what to use today, the clean answers are:

- LTX fast for local text-to-video and image-to-video
- LTX conditioned-audio for the current promoted audio-conditioned slice
- Qwen-Image base or Lightning for serious local image generation and editing
- FLUX.2 `klein-9b` for straightforward local still-image generation and
  single-reference edit
- Z-Image for prompt-only still-image generation when a narrower image-only
  slice is enough

## What We Should Not Overclaim

- full desktop parity
- full LTX `dev` quality parity
- prompt enhancement as a built-in runtime feature
- FLUX.2 KV speedups
- Layered Qwen support
- every implemented row as a promoted product default

## Current Product Direction

The project direction is now:

1. make the runtime and CLI public, legible, and trustworthy
2. harden the simple native Mac app, especially the new studio, library, and model-management flows, into the first user-facing product
3. keep desktop and Comfy as thin client or compatibility tracks
4. treat a future `MLXR Studio` as a separate longer-term product

## Current Release-Readiness Gaps

The main blockers between “strong internal repo” and “clean public OSS project”
are:

- repo-wide green-up and release hygiene
- a clearer public status and roadmap story
- explicit OSS files such as license, contributing, support, and security docs
- a more public-facing CLI story
- app hardening, polish, and release packaging for the first-party Mac client

## Pointers

- [roadmap.md](/Users/numman/Repos/mlxr/docs/roadmap.md)
- [open-source-release-checklist.md](/Users/numman/Repos/mlxr/docs/open-source-release-checklist.md)
- [cli-ergonomics.md](/Users/numman/Repos/mlxr/docs/cli-ergonomics.md)
- [mac-app-v1-product-spec.md](/Users/numman/Repos/mlxr/docs/mac-app-v1-product-spec.md)
- [plans/mac-app-runtime-parity-plan.md](/Users/numman/Repos/mlxr/docs/plans/mac-app-runtime-parity-plan.md)

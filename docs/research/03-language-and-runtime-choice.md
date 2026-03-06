# Language And Runtime Choice

## Executive Recommendation

The decision is no longer framed as “Python or Rust.”

The split that survives current evidence is:

- Python for family adapters, conversion, and workers
- Swift earlier for first-party host integration, launcher, SDK, and Apple-native embedding
- C++ and Metal as the explicit hotspot and MLX-extension seam
- Rust optional for tooling or supervision, not the v1 inference core

## What Python Must Own

Python should still own v1 family bring-up because that is where the current MLX ecosystem actually lives.

Python responsibilities:

- model-family adapters
- provider-driven inspection and conversion glue
- processor and tokenizer integration
- semantic parity work
- most worker execution logic in v1

Why this still holds:

- official MLX examples and docs remain Python-first for bring-up
- active MLX ecosystem packages are Python-first
- most current model conversion and provider flows are already Python-native

## Why Rust Should Remain Non-Core

Rust is not rejected because it is bad. It is rejected as the core inference runtime because it would solve the wrong problem first.

### What Rust would slow down

- family bring-up velocity
- parity work against current Python-based upstreams
- provider and conversion integration
- rapid adaptation to changing LTX and multimodal codepaths

### What Rust may still do later

- optional supervisor processes
- content-addressable storage helpers
- packaging or service tooling
- narrow infra-side bottlenecks that do not own family semantics

Current conclusion:

- Rust is still optional and non-core
- it is not the default runtime language for workers

## Why Swift Matters Earlier Than The Old Docs Said

The earlier docs were too dismissive of Swift.

Current upstream signals:

- `mlx-swift` is active
- its README explicitly positions it as a Swift API for MLX
- it documents framework-style embedding concerns for Apple apps
- it warns that command-line SwiftPM cannot build the Metal shaders, so real Metal-backed app builds still need Xcode or an Xcode-driven path
- it links to `mlx-swift-examples` and `mlx-swift-lm`, which prove there is already a meaningful Apple-native host path

Swift should move earlier for:

- launcher and service wrapper work
- first-party host SDK design
- embedded access mode for Apple apps
- Apple-native media and output integration

Swift should also own the “one MLX copy in process” problem at the host boundary. The current official README already warns about duplicate MLX copies in one process when frameworks and apps link MLX incorrectly.

Swift should still not own:

- v1 family bring-up
- core provider inspection and conversion
- fast-changing model semantics

## Why C++ And Metal Must Be Reserved Now

Low-level work should still be benchmark-driven, but the seam must be explicit now.

Candidate areas:

- VAE decode and tile decode
- patchify and unpatchify
- latent layout transforms
- spectrogram and audio preprocessing
- vocoder-critical helpers
- Apple-native output pipelines and no-copy-ish media handoff

This does not mean rewriting the platform in C++. It means creating a clean boundary for profiled hotspots.

## Better MLX: Local Extension Path And Upstream Roadmap

The project can create value in four layers:

1. use stock MLX correctly
2. add local MLX extensions for profiled hotspots
3. maintain a small fork or targeted upstream backlog for media-specific pain
4. build Apple-native output plumbing around MLX

That means “better MLX” is a real track, not a vague escalation bullet.

See [ADR-0005](../adr/0005-mlx-extension-and-upstream-roadmap.md).

## Revised Ownership Table

| Area | Preferred owner | Why |
| --- | --- | --- |
| Provider adapters | Python | matches current tooling and provider SDKs |
| Family adapters | Python | fastest bring-up and parity work |
| Conversion and inspection | Python | best alignment with source and metadata tooling |
| Runtime workers | Python first | simplest MLX integration path today |
| Host SDK and launcher | Swift | best Apple-native fit |
| Embedded first-party access | Swift plus shared core | avoids daemon-only rigidity |
| Native hotspots | C++ and Metal | best place for profiled MLX extensions |
| Optional supervisor or tooling | Rust | acceptable but non-core |

## Final Decision

- Keep Python as the v1 worker and family language.
- Move Swift earlier at the host boundary.
- Reserve C++ and Metal as a first-class hotspot seam.
- Keep Rust optional and non-core.

This is a better fit for the actual state of the MLX ecosystem on 2026-03-06 than the older Python-vs-Rust framing.

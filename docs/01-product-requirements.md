# MLXR Product Requirements

Status: provisional product requirements for platform and first product track. These requirements are concrete enough to guide implementation, but they do not freeze the public API or the final worker topology.

## Vision

Build the best local-first generative and multimodal runtime for Apple Silicon.

That means:

- users point the runtime at model sources they are allowed to use
- the runtime preserves provenance, access state, and policy
- the runtime fetches, validates, converts, caches, schedules, and serves those models locally
- every host surface sees one truthful capability model instead of inventing its own inference contract

The current scope is universal across host surfaces and model-family adapters for local generative and multimodal workloads. It is not yet a claim of support for every possible MLX workload class.

## Product Thesis

The product is not “an MLX port of LTX.” It is:

- a reusable Apple Silicon runtime
- with LTX as the first proving workload
- designed so additional image, video, audio, speech, VLM, and selected text families can land without rewriting host integrations

The runtime should create value in two layers:

1. As a platform: one source/provenance model, one artifact model, one scheduler, one native API, one operations model.
2. As a product: a clean local experience for difficult workloads that official upstream tools do not support well on macOS today.

## Scope We Can Defend

### Universal across

- host surfaces: CLI, local daemon, desktop adapters, Comfy adapters, embedded first-party use
- model-family architecture: video, image, audio, speech, VLM, and at least one text family as a validation pressure test
- provider architecture: the runtime models multiple providers even if v1 only implements a small subset

### Not yet universal across

- every provider in implementation
- every MLX workload class
- every public API shape
- every hardware tier

## Primary Users

### 1. Local creators

They want one installation, one runtime, and predictable model behavior across tools. They should not need to understand provider cache layouts, artifact conversion internals, or MLX compile behavior.

### 2. Product integrators

They want to build a desktop app, service wrapper, or adapter without re-implementing model loading, scheduler policy, or provenance handling.

### 3. Model bring-up engineers

They want clear adapter seams, reproducible conversion paths, and honest capability declarations.

### 4. Performance engineers

They want a runtime that makes it obvious where Python is enough, where MLX compile helps, and where native MLX extensions or upstream work are justified.

## Non-Negotiable Product Principles

- Local-first: no cloud dependency for the core runtime path.
- Provenance-first: every source must retain provider, revision, license, access state, and remote-code policy.
- Truthful capabilities: hosts must not guess what a model can do.
- Safe local serving: browser-reachable HTTP is a security surface, not a convenience feature.
- Apple-specific honesty: unified memory, media engines, build/JIT caches, and launch behavior are core requirements.
- Thin adapters: host packages stay thin and defer inference ownership to the shared runtime.

## v1 Provider Scope

Architecture scope and implementation scope are intentionally different.

### Required in the architecture now

- Hugging Face
- local filesystem bundles
- GitHub release or repo sources
- OCI or object-store style registries as future provider classes

### Required in the first implementation slice

- Hugging Face as the primary provider
- local filesystem bundles for trusted offline workflows

Anything beyond that remains provisional until the provider abstraction and benchmark plan are validated.

## Required v1 User Experience

### Model lifecycle

Users should be able to:

1. reference a model source
2. inspect requirements before a full download
3. fetch with policy-aware auth
4. convert into a portable runtime artifact
5. run jobs through one native contract
6. export outputs without needing raw daemon-side filesystem paths in the generic HTTP API

### Host surfaces

The runtime must support:

- a trusted local CLI
- a native daemon API with event streaming
- first-party Apple embedding or shared-core access for future native apps
- thin adapters for desktop and Comfy-style workflows

Compatibility APIs such as OpenAI-style facades may exist later, but they are adapters, not the source of truth.

### Install and service UX

The runtime must define:

- install location and runtime home
- LaunchAgent or equivalent service lifecycle
- upgrade and migration behavior
- disk quota and garbage collection policy
- crash recovery behavior
- sleep and wake behavior on macOS

Those are product requirements, not implementation afterthoughts.

## Required Proving Workloads

These are required before the native API can be called stable.

### Platform validation basket

- `LTX-2.3 Fast` text-to-video
- `LTX-2.3 Fast` image-to-video
- one image diffusion family
- one non-generation family such as Whisper ASR or a VLM understanding family
- one text family or an explicit documented decision to keep text outside the stable scope

### Product-first basket

- LTX `Fast` local inference on Apple Silicon
- truthful capability reporting for LTX constraints, profiles, outputs, and policy state
- one desktop integration path
- one Comfy integration path with value beyond baseline “proxy the same buttons”

## Source, License, And Policy Requirements

Weights and processors come from hub-hosted or local sources under explicit license and access state.

The runtime must preserve:

- provider
- revision or digest
- declared license
- access state such as public, gated, private, or local-only
- remote-code requirement and approval state
- blob digests where available

Converted artifacts are not assumed to be redistributable, even when the source is accessible to the local user.

The runtime must treat `trust_remote_code` as a policy boundary, not a convenience flag.

## Security Requirements

- Default external transport on macOS is Unix domain socket, not loopback HTTP.
- Loopback HTTP is opt-in and requires mandatory auth.
- Mutating HTTP routes require browser-origin protections when HTTP is enabled.
- The canonical HTTP API uses input handles and export flows, not raw absolute file paths.
- Trusted direct-path access may exist for the local CLI or embedded callers, but it is a separate trusted surface.

## Performance Requirements

Performance claims are benchmark-gated. Demo-level “it ran once” is not a success metric.

The runtime must measure:

- source fetch and preflight time
- cold convert, cold load, warm load, and warm run
- per-stage timings
- compile counts and compile-cache hit rates
- active and peak memory
- output encode time separate from model execution
- cancellation latency
- failure reason taxonomy

See [Benchmark matrix](benchmark-matrix.md) for the exact gates.

## Hardware Tiers

Hardware tiers remain provisional until measured.

### Working assumptions

- `64 GB+` is the likely recommended tier for full LTX fast-path targets
- `32 GB` may be a degraded tier for narrower profiles, but this is not yet a promise
- `128 GB` class systems should be treated as a separate planning tier, not just “more headroom”

The runtime must surface truthful hardware requirements per capability profile instead of one vague “recommended machine” note.

## v1 Non-Goals

- training and fine-tuning
- cloud inference as part of the core runtime path
- treating build cache as a portable artifact
- declaring the public API fixed before cross-family validation
- pretending Comfy or desktop adapters should own inference logic

## Success Criteria

Product success is achieved only if all of the following are true:

- the runtime can serve one difficult video family and additional non-video families through one native lifecycle model
- host adapters remain thin
- provenance and policy are preserved end to end
- local service lifecycle is usable on macOS
- the benchmark matrix is good enough to freeze decisions with evidence rather than optimism

The stable-freeze gates live in [Benchmark matrix](benchmark-matrix.md) and [Open questions and validation plan](research/09-open-questions-and-validation-plan.md).

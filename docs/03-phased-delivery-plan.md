# Phased Delivery Plan

Status: revised plan. The architecture and public API are not marked fixed in Phase 0 anymore.

## Summary

The work is split into two tracks:

- platform track: the reusable runtime and its contracts
- product track: the LTX-first path that proves the platform under a hard workload

The stable-freeze checkpoint happens only after the platform survives more than one family and more than one task shape.

## Phase A: Platform Skeleton And Risk Closure

Goal: build the minimum platform backbone without freezing the API too early.

### Deliverables

- provider abstraction and provenance manifests
- portability classes for source refs, portable artifacts, and machine-local build cache
- control-plane daemon scaffold
- worker-process scaffold
- capability schema `v0`
- handle-based import/export model
- security baseline: UDS default, HTTP opt-in, mandatory auth rules
- operations baseline: runtime home, logs, quotas, launch strategy

### Exit criteria

- ADRs `0001` through `0005` are in place
- one provider can resolve and inspect without downloading everything blindly
- one worker can execute through the new control-plane path
- no document still claims the API is fixed

## Phase B: LTX Fast Proof Path

Goal: prove the platform on `LTX-2.3 Fast` without pretending the result generalizes automatically.

### Deliverables

- Hugging Face source resolution for LTX
- portable artifact conversion for LTX fast-path assets
- local Gemma text encoding path
- `video.generate` and `video.condition.image` job support
- truthful LTX capability reporting
- runtime-managed output artifacts

### Exit criteria

- cold pull, convert, load, and warm run metrics exist
- constraints such as resolution and frame-count rules are enforced by the platform, not only by an adapter README
- the scheduler records stage timings and memory telemetry
- LTX desktop can call the runtime in a basic thin-client mode

## Phase C: Cross-Family Platform Validation

Goal: pressure-test the platform with non-LTX families before any API freeze.

### Required validation families

- one image diffusion family
- one non-generation family such as Whisper or a VLM understanding family
- one text family or a documented decision to leave text outside the stable v1 scope

### Deliverables

- scheduler-class validation beyond video
- capability schema updates driven by cross-family pressure
- artifact and build-cache behavior validated on multiple families
- provider and provenance model exercised beyond a single LTX-shaped download path

### Exit criteria

- benchmark matrix coverage exists across the minimum family basket
- one provider limitation or second-provider path is documented honestly
- architecture changes triggered by Phase C are applied before freeze

## Phase D: Product Surfaces

Goal: turn the validated platform path into usable product integrations.

### Deliverables

- desktop adapter aligned to the shared runtime
- Comfy strategy aligned to current core-plus-advanced-node reality
- trusted CLI surface
- first-party embedded or SDK access design for future Apple apps

### Exit criteria

- desktop and Comfy integrations are thin clients over the shared runtime
- any remaining host-specific direct-path behavior is isolated to trusted local surfaces
- host parity scenarios are added to the benchmark matrix

## Phase E: Better MLX And Native Hotspots

Goal: improve the substrate where profiling proves it matters.

### Deliverables

- hotspot shortlist with benchmark evidence
- first `runtime-mlx-ext` or `runtime-kernels` package when justified
- Apple-native output-path experiments
- explicit upstream or fork backlog for MLX pain points

### Exit criteria

- stock MLX vs extension-assisted benchmarks exist
- low-level work is justified by real profile traces, not by guesswork

## Phase F: Freeze Review

Goal: decide what can actually be called stable.

### Freeze candidates

- capability schema
- native job lifecycle
- provider and provenance contract
- security baseline
- artifact and build-cache contract

### Freeze rules

Do not freeze if any of the following remain unresolved:

- benchmark matrix gaps on required families
- provider abstraction only works for one narrow source flow
- handle-based import/export still leaks raw-path assumptions into the generic HTTP API
- worker-process topology still changes fundamental lifecycle semantics
- operations model is missing crash, upgrade, or quota behavior

## Cross-Cutting Workstreams

### Security and policy

Runs from Phase A through Freeze Review. It is not a post-beta cleanup item.

### Benchmarking

Begins in Phase B and becomes mandatory in Phase C. See [Benchmark matrix](benchmark-matrix.md).

### Operations and packaging

Begins in Phase A because launch, cache, and upgrade behavior influence the runtime shape.

### MLX extension and upstream track

Runs in parallel once hotspots are proven. See [ADR-0005](adr/0005-mlx-extension-and-upstream-roadmap.md).

## Default Execution Order

1. Phase A
2. Phase B
3. Phase C
4. Phase D
5. Phase E
6. Phase F

The sequence is deliberate. The platform must survive cross-family validation before it earns a stable public contract.

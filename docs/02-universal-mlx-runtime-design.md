# MLXR Technical Design

Status: provisional architecture. This is the working design to implement against now, but it is not the final frozen public contract. API freeze is gated by the benchmark matrix and cross-family validation.

## Summary

The runtime is split into a platform track and a product track.

- The platform track defines the source model, artifact model, provenance model, security posture, scheduler, worker topology, and host surfaces.
- The product track uses `LTX-2.3 Fast` to pressure-test those decisions on a hard Apple Silicon workload.

The architecture that survives review is:

- one canonical local runtime
- a native job-oriented API as the source of truth
- provider adapters and family adapters as separate seams
- a control-plane daemon plus execution workers
- portability classes separated into source references, portable artifacts, and machine-local build cache
- Python-first workers, Swift earlier at the host boundary, C++/Metal as an explicit hotspot seam

## Architecture Status

### Strong decisions

- one runtime, many surfaces
- native API first, compatibility facades second
- Python-first model-family bring-up
- thin host adapters
- LTX as the first proving workload

### Provisional decisions

- exact worker granularity
- exact capability schema version `1`
- exact handle import/export flows
- final scheduler heuristics and reservation formulas
- exact MLX extension backlog

Those decisions stay provisional until the validation plan upgrades them.

## Top-Level Architecture

```text
Hosts
  CLI / SDK / Desktop / Comfy / Embedded First-Party Apps
                  |
                  v
        Control-Plane Daemon
  - transport + auth
  - provider resolution
  - provenance + policy store
  - artifact index
  - scheduler
  - capability service
  - job store + event stream
                  |
                  +--> Family worker: LTX
                  +--> Family worker: image diffusion
                  +--> Family worker: speech or VLM
                  +--> Family worker: optional text family
```

The daemon is the canonical external surface. Embedded first-party apps may use a shared core or SDK path, but they still need to honor the same lifecycle, provenance, and scheduler rules.

## Provider Model

The previous design was family-broad but provider-narrow. That is fixed here.

### Provider seam

`SourceProviderAdapter` is parallel to `ModelFamilyAdapter`. It is responsible for resolving sources, auth requirements, inspection, fetch, and provenance.

```python
class SourceProviderAdapter(Protocol):
    provider_id: str

    def resolve(self, source_ref: "SourceRef") -> "ResolvedSource": ...
    def inspect(self, resolved: "ResolvedSource") -> "SourceInspection": ...
    def auth_requirements(self, source_ref: "SourceRef") -> "AuthRequirements": ...
    def fetch(self, resolved: "ResolvedSource", policy: "FetchPolicy") -> "SourceMaterialization": ...
    def provenance(self, resolved: "ResolvedSource") -> "ProvenanceRecord": ...
```

### v1 provider scope

- implemented now: Hugging Face and trusted local filesystem bundles
- designed now, not yet promised in v1: GitHub releases/repos, OCI-style registries, private internal stores

See [Provider and provenance model](provider-and-provenance-model.md) and [ADR-0001](adr/0001-source-provider-adapter.md).

## Portability Classes

The old design incorrectly mixed portable artifacts and machine-local compile state. That is replaced with three explicit classes.

### 1. Source references and provenance

These are provider-backed references plus manifests that say what was resolved, under what auth and policy, at what revision.

Examples:

- Hugging Face repo + pinned revision
- local bundle path + digest manifest
- GitHub release asset + tag and digest

### 2. Portable converted artifacts

These are runtime-usable converted artifacts that can move between machines if the source license allows it.

They may contain:

- converted weights
- processors and tokenizers
- family metadata
- capability declarations
- dependency manifests

They do not contain machine-local MLX compile state or JIT output.

### 3. Machine-local build cache

This holds per-machine, per-version, per-shape, and per-native-module state such as:

- compiled function traces
- kernel libraries
- shape-bucketed build outputs
- other non-portable local acceleration state

See [ADR-0002](adr/0002-artifact-vs-build-cache.md).

## Storage Layout

The storage layout remains provisional, but the portability-class split is not.

```text
$MLX_RUNTIME_HOME/
  config/
  logs/
  jobs/
  temp/
  sources-ref/
    huggingface/
    local/
    github/
    oci/
  artifacts-portable/
    <family>/<model_id>/<artifact_digest>/
  build-cache-local/
    <mlx_version>/<macos_build>/<chip_family>/<adapter_version>/<shape_bucket>/<precision>/
```

Internally the implementation may move toward content-addressable storage for deduplication and safer garbage collection. That is a strong internal direction, but not yet a user-visible contract.

## Worker And Process Topology

The control plane and execution plane are separated on purpose.

### Control-plane daemon responsibilities

- transport and auth
- capability and registry API
- provider resolution and policy checks
- artifact lookup
- scheduling and admission control
- job state and event streaming

### Worker responsibilities

- family-specific load and execution
- access to MLX, Python model glue, and optional native extensions
- stage-level telemetry back to the scheduler
- local reclamation and crash isolation

### v1 default

- one heavy media generation job per worker process
- light auxiliary work may share workers if benchmarked safe
- warm worker reuse is allowed only when it does not violate isolation or memory goals

See [ADR-0004](adr/0004-worker-process-topology.md).

## Family Adapter Contract

Family adapters stay in Python for v1 bring-up, but they must expose enough structure for the platform to stay honest.

```python
class ModelFamilyAdapter(Protocol):
    family_id: str

    def inspect_source(self, source: "SourceMaterialization") -> "FamilyInspection": ...
    def convert(self, source: "SourceMaterialization", plan: "ConversionPlan") -> "PortableArtifact": ...
    def load(self, artifact: "PortableArtifact", profile: "ExecutionProfile") -> "LoadedModelHandle": ...
    def capabilities(self, artifact: "PortableArtifact") -> "CapabilityDescriptor": ...
    def run_stage(self, loaded: "LoadedModelHandle", stage: "ExecutionStage") -> "StageResult": ...
    def unload(self, loaded: "LoadedModelHandle") -> None: ...
```

The adapter contract is intentionally stage-aware so the scheduler can reason about load, prompt encoding, denoise, decode, encode, and teardown as separate resource phases.

## Capability Model

The earlier capability descriptor was too thin. The runtime now assumes a richer schema.

Required fields include:

- identity: model ID, family, variant, artifact digest
- tasks and modalities
- constraints
- profiles by task
- artifacts out
- scheduler class
- hardware tiers
- dependencies
- policy state such as license, access state, and remote-code requirement
- extension namespace for family-specific fields

The full contract lives in [Capability schema](capability-schema.md).

## Access Modes

There are three access modes, and they are intentionally different.

### 1. Native daemon mode

Default external surface for CLI, SDK, desktop adapters, and third-party tools.

### 2. Embedded first-party mode

Reserved for first-party Apple apps that should not pay daemon-only integration costs. Embedded callers still use the same lifecycle and scheduler model.

### 3. Trusted local mode

A local-only direct-path convenience surface for the CLI or trusted embedding. This is where explicit local file paths may remain acceptable.

The canonical HTTP contract is not the trusted direct-path surface.

## File, Input, And Artifact Handle Model

The universal HTTP contract no longer accepts raw absolute file paths by default.

### Generic client flow

1. Import user inputs through an import route or SDK helper.
2. Receive input handles.
3. Reference those handles in job payloads.
4. Let the runtime produce runtime-managed artifacts.
5. Export or stream results through explicit output flows.

### Trusted local flow

The CLI or embedded first-party callers may use a direct-path mode when the runtime can treat the caller as a trusted local process. That path is not the universal default and should not bleed into adapter contracts.

## Transport And Security Model

The previous `127.0.0.1` plus optional token design was not good enough.

### Default transport

- macOS default: Unix domain socket
- loopback HTTP: opt-in only
- remote network bind: out of scope for v1

### HTTP rules when enabled

- mandatory auth for all mutating routes
- no wildcard CORS
- explicit browser-origin protections for requests carrying `Origin`, `Referer`, or `Sec-Fetch-*`
- reject cross-site browser requests
- prefer token delivery that is not ambient browser credential state

### Remote code policy

- `trust_remote_code` is off by default
- approval is source-specific and revision-specific
- remote-code approval must be visible in provenance and capability metadata
- remote-code execution belongs in worker isolation boundaries, not the control plane

See [ADR-0003](adr/0003-local-security-model.md) and [Provider and provenance model](provider-and-provenance-model.md).

## Native API Shape

The runtime keeps a native job-oriented API as the source of truth, but that API is still provisional.

### Stable design principles

- job-oriented
- handle-based inputs and outputs
- event streaming for progress and stage updates
- capability discovery before job submission
- explicit provenance and policy propagation

### Not yet frozen

- exact REST route naming
- exact SDK surface
- the final balance between synchronous convenience calls and async job handles

The transport and contract details live in [Serving architecture and API](research/04-serving-architecture-and-api.md).

## Scheduler Resource Model

The earlier `media mode` vs `batch mode` split was directionally right but too shallow. The scheduler is now defined as a resource graph.

### Scheduler classes

Examples:

- `media_video_dit`
- `image_diffusion`
- `speech_streaming`
- `vlm_interactive`
- `text_batch_or_stream`

### Core resources

- unified memory budget
- MLX/Metal compute budget
- CPU thread budget
- media encode/decode engines
- disk and network I/O
- worker slots
- build-cache activity

### Stage reservations

Jobs reserve resources per stage, not just per job:

- source fetch
- inspect
- convert
- load
- conditioning / prompt encode
- denoise or generate
- decode
- encode or mux
- export

### Live telemetry inputs

The scheduler uses:

- adapter-declared estimates
- live memory telemetry from MLX and Metal where available, including platform signals equivalent to current allocated size and recommended working set where exposed
- build-cache hit or miss data
- stage timings from prior runs
- cancellation and failure history

### v1 safety defaults

- one heavy generation stage at a time per worker
- conservative admission when memory pressure is uncertain
- cooperative cancellation points between stages and inside long loops where practical

The old `media` vs `batch` taxonomy survives only as a coarse policy hint above this actual resource model.

## Native Hotspot And Better-MLX Seam

Low-level acceleration is not a vague future escape hatch anymore. It is a first-class seam.

### Reserved packages

- `packages/runtime-mlx-ext/`
- `packages/runtime-kernels/`

### What belongs there

- profiled hotspots only
- custom MLX extensions
- custom Metal kernels
- native helpers for layout transforms or output pipelines

### What does not belong there

- speculative kernels without profiling evidence
- generic rewrites of family logic that still belongs in Python

See [ADR-0005](adr/0005-mlx-extension-and-upstream-roadmap.md).

## Language Ownership

- Python owns family adapters, conversion, and workers in v1.
- Swift moves earlier for first-party host SDKs, service launch, embedding, and Apple-native media integration.
- C++/Metal is the native hotspot seam.
- Rust remains optional and non-core.

The detailed rationale lives in [Language and runtime choice](research/03-language-and-runtime-choice.md).

## Product Track: LTX First

LTX remains the first proving workload because it forces the runtime to handle:

- large video generation
- text and image conditioning
- multi-stage execution
- audio-aware outputs
- upscalers and LoRAs
- strict input constraints
- desktop and Comfy host pressure

### First supported LTX slice

- `LTX-2.3 Fast` text-to-video
- `LTX-2.3 Fast` image-to-video
- local Gemma text encoding
- runtime-managed output artifacts

### Designed now, not promised in the first slice

- full audio parity
- retake
- full advanced IC-LoRA flows
- broader Comfy graph coverage

The LTX-specific integration surfaces live in [LTX integration seams](research/07-ltx-integration-seams.md).

## Freeze Criteria

This document becomes stable only when the same lifecycle and capability model survives:

1. LTX Fast
2. one image diffusion family
3. one non-generation family such as Whisper or a VLM understanding family
4. the benchmark and operations matrix
5. at least the currently declared provider scope

Until then, this design is the implementation baseline, not a frozen architecture claim.

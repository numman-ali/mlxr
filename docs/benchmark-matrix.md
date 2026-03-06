# Benchmark Matrix

Status: required before the architecture or public API can be called stable.

## Purpose

This matrix replaces demo-level acceptance criteria with freeze-grade evidence.

The runtime is not considered validated because it can “run once.” It is validated only when it survives the matrix below with recorded timings, memory, and failure behavior.

## Required Axes

| Axis | Minimum coverage |
| --- | --- |
| Families | `LTX-2.3 Fast` T2V, `LTX-2.3 Fast` I2V, one image diffusion family, one non-generation family, one TTS or audio family, one VLM, and one text family or an explicit freeze-blocking exclusion |
| Hardware | at least one `32 GB`, one `64 GB`, and one `128 GB` Apple Silicon tier; include one older Max-class machine and one newer Max-class machine if available |
| Run state | cold inspect, cold fetch, cold convert, cold load, warm load, warm run, model switch |
| Concurrency | isolated heavy job, heavy plus light, multiple light jobs, cancellation under load |
| Output path | model-only time, decode time, encode or mux time, export time |
| Precision and profile | baseline float path, q8, lower-bit profiles where meaningful, selective quantization where applicable |
| Shapes | common bucketed shapes and deliberately off-bucket shapes |

## Required Workload Basket

| Family | Why it is required |
| --- | --- |
| `LTX-2.3 Fast` T2V | hardest current product pressure: staged video generation, strict constraints, big memory, output muxing |
| `LTX-2.3 Fast` I2V | forces conditioning and artifact handle flow |
| Image diffusion | validates non-LTX media generation without LTX-specific assumptions |
| Whisper or equivalent | validates non-generation pipeline stages and different scheduler behavior |
| TTS or audio family | validates audio outputs, vocoder or chunking pressure |
| VLM | validates mixed-modality processors and interactive generation patterns |
| Text family | validates whether the platform can honestly support text-serving pressure or must scope it out |

## Mandatory Metrics

These must be recorded per run:

- source resolve time
- preflight inspect time
- fetch time
- convert time
- cold load time
- warm load time
- run wall time
- per-stage timings
- active memory
- peak memory
- build-cache hit rate
- compile count
- compile time
- source-cache hit rate
- artifact-cache hit rate
- output encode or mux time
- cancellation latency
- failure reason

When available, also record:

- Metal `recommendedMaxWorkingSetSize`
- Metal or MLX active memory telemetry
- Metal `currentAllocatedSize`
- CPU, GPU, and media-engine utilization

## Failure Taxonomy

Every failed run must be categorized instead of disappearing into “error.”

Minimum categories:

- provider resolution failure
- auth or access-state failure
- license or policy failure
- remote-code policy failure
- source corruption or digest mismatch
- conversion failure
- load failure
- compile failure
- admission rejection
- runtime OOM or memory pressure
- output encode or export failure
- host-adapter parity failure

## Host Parity Scenarios

These scenarios are required before the host contract can be called stable:

- same model through CLI and native daemon API
- same runtime-managed artifact flow through desktop adapter
- same core capability and policy state exposed through a Comfy adapter or execution bridge
- same rejection reasons surfaced consistently for invalid constraints

## Freeze Gates

The architecture and native API remain provisional until all of the following are true:

1. `LTX-2.3 Fast` T2V and I2V both pass with benchmark traces.
2. One image diffusion family passes through the same lifecycle model.
3. One non-generation family passes through the same source, artifact, and scheduler framework.
4. One text family either passes or is explicitly excluded from stable scope with written rationale.
5. The security and operations scenarios are exercised under the same runtime.

## Provisional Acceptance Targets

Targets here are structural, not performance promises to users.

- the runtime records every mandatory metric
- capability declarations match actual constraint enforcement
- build cache and portable artifacts stay separated in implementation and reporting
- cancellation works at stage boundaries and inside long-running workloads where supported
- host adapters do not bypass provenance or policy handling

## Reporting Rules

- Record machine details, macOS version, MLX version, and adapter version for every benchmark set.
- Record source revision and artifact digest for every result.
- Keep benchmark outputs versioned under `benchmarks/` or a structured export store.
- Do not upgrade a provisional decision to stable based on one machine or one family.

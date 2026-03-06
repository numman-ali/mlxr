# Open Questions And Validation Plan

## Purpose

This document defines what is still unknown, what must be measured, and what conditions upgrade a provisional decision into a stable one.

## What Is Strong Enough To Keep

- one canonical runtime
- native job API as the source of truth
- Python-first family bring-up and workers
- thin host adapters
- LTX first as a proving workload

Everything else below remains provisional until the listed measurements exist.

## Open Questions

| Question | Why it is still open | Validation needed | Freeze impact |
| --- | --- | --- | --- |
| Which provider set is enough for stable v1? | architecture now models more than the implementation does | benchmark Hugging Face plus local bundles; decide whether second provider is required or explicitly deferred | blocks provider-scope freeze |
| What exactly is portable versus machine-local in the storage model? | exporter and compile behavior still make portability easy to overclaim | artifact portability tests across machines and MLX versions | blocks storage-contract freeze |
| Which worker topology is best? | control-plane plus workers is chosen, but worker granularity is still open | compare per-family warm workers against more isolated per-job workers on memory, latency, and failure recovery | blocks execution-topology freeze |
| What is the canonical Apple-native output path? | current PyAV path is clearly not the final Apple-optimized answer | benchmark generic path versus `VideoToolbox`-oriented path and copy counts | blocks media-output guidance freeze |
| How much Swift should ship in the first product slice? | Swift matters earlier, but exact packaging and SDK boundaries are still open | spike launcher or embedded host path against daemon-only path | blocks host-SDK freeze |
| Which MLX extension or upstream changes are worth carrying? | low-level work should be benchmark-driven, not hand-wavy | hotspot traces plus stock-versus-extension benchmarks | blocks low-level roadmap freeze |
| What are the real LTX memory envelopes on Apple Silicon? | current tiers are still educated guesses | benchmark matrix on `32 GB`, `64 GB`, and `128 GB` machines | blocks hardware-tier freeze |
| Which compile bucket strategy is best for LTX and video? | shape churn can erase gains | profile common and off-bucket shapes | blocks compile-policy freeze |
| Which acceleration papers survive contact with MLX and LTX? | paper wins do not transfer automatically | controlled benchmark tracks with quality checks | blocks research-optimization freeze |
| Does the native contract honestly support text families, or should they stay outside stable scope? | text pressure exposes different batching and compatibility assumptions | run one text family through the same lifecycle model or explicitly exclude it | blocks “universal” wording freeze |

## Validation Plan

### Track 1: Source and provenance

- implement provider inspection before full download
- verify revision pinning and manifesting
- validate license and access-state propagation
- validate remote-code policy recording

### Track 2: Artifact and build cache

- convert one family into a portable artifact
- invalidate build cache without invalidating the artifact
- compare warm behavior with and without build-cache reuse

### Track 3: Execution topology

- compare in-process bring-up versus worker process execution
- measure crash isolation and recovery
- measure memory reclamation after worker teardown

### Track 4: Scheduler and memory

- record live telemetry during load, generation, decode, and encode
- test conservative admission and rejection reasons
- test cancellation during heavy jobs

### Track 5: Host parity

- CLI
- native daemon API
- desktop adapter
- Comfy execution bridge or adapter

Host parity is not “same feature buttons.” It is same lifecycle, same policy, and same truthful failures.

## Freeze Criteria

The architecture and native API can only be called stable when:

1. `LTX-2.3 Fast` T2V and I2V both pass the benchmark matrix.
2. One image diffusion family passes through the same source, artifact, and scheduler model.
3. One non-generation family passes through the same lifecycle.
4. One text family either passes or is explicitly excluded from stable scope with written reasons.
5. The security and operations model has been exercised under the same runtime.

## Failure-Mode Tracking

Every validation run should log:

- family and task
- source revision
- artifact digest
- machine tier
- failure category
- recovery outcome

Failure categories must at least include:

- provider or auth failure
- policy failure
- conversion failure
- load failure
- compile failure
- scheduler admission rejection
- runtime memory failure
- output encode or export failure
- host-adapter mismatch

## Rule For Promoting Decisions

Use these labels consistently:

- `strong`
- `reasonable but under-validated`
- `incomplete`
- `likely wrong`
- `clearly wrong`

No decision graduates from provisional to stable without:

- benchmark evidence
- cross-family validation
- updated docs

If a decision still depends on one family, one machine, or one provider path, it is not stable yet.

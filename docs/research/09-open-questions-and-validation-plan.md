# Open Questions And Validation Plan

## Purpose

This document defines what is still unknown, what must be measured, and what conditions upgrade a provisional decision into a stable one.

## What Is Strong Enough To Keep

- one canonical runtime
- native job API as the source of truth
- core-owned workflow planning between scheduler, family adapters, and hosts
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
| What is the minimum workflow-template contract that survives more than LTX? | LTX proves the need for a core workflow-planning layer, but the reusable template shape is still only exercised on one hard media family | run one image diffusion family and one non-generation family through the same workflow-planning split without host-owned request shaping | blocks workflow-contract freeze |
| What is the canonical Apple-native output path? | current PyAV path is clearly not the final Apple-optimized answer | benchmark generic path versus `VideoToolbox`-oriented path and copy counts | blocks media-output guidance freeze |
| How much Swift should ship in the first product slice? | Swift matters earlier, but exact packaging and SDK boundaries are still open | spike launcher or embedded host path against daemon-only path | blocks host-SDK freeze |
| Which MLX extension or upstream changes are worth carrying? | low-level work should be benchmark-driven, not hand-wavy | hotspot traces plus stock-versus-extension benchmarks | blocks low-level roadmap freeze |
| What are the real LTX memory envelopes on Apple Silicon? | current tiers are still educated guesses | benchmark matrix on `32 GB`, `64 GB`, and `128 GB` machines | blocks hardware-tier freeze |
| Which compile bucket strategy is best for LTX and video? | shape churn can erase gains | profile common and off-bucket shapes | blocks compile-policy freeze |
| Which acceleration papers survive contact with MLX and LTX? | paper wins do not transfer automatically | controlled benchmark tracks with quality checks | blocks research-optimization freeze |
| Does the native contract honestly support text families, or should they stay outside stable scope? | text pressure exposes different batching and compatibility assumptions | run one text family through the same lifecycle model or explicitly exclude it | blocks “universal” wording freeze |

## Latest Measured Blocker

As of the current `2026-03-07` validation pass:

- provider auth, gated Gemma access, Unix-socket transport, large-file download, and portable-artifact conversion all succeeded on the current harness
- the repo-owned MLX prompt path now handles the current upstream `LTX-2.3` 22B prompt stack in direct artifact-backed smoke tests, including the V2 `video_aggregate_embed` / `audio_aggregate_embed` path and config-driven connector dimensions
- the repo-owned generation path now completes direct artifact-backed low-resolution T2V and I2V smoke tests against the real `LTX-2.3` weights, including stage-1 denoise, x2 latent upsample, stage-2 refinement, VAE decode, and runtime-managed `mp4` export
- the current AV bridge now decodes checkpoint-backed audio and exports it as muxed runtime-managed `mp4` or standalone `wav`, and the repo-owned MLX path now includes the checkpoint's full BWE wrapper instead of stopping at the `AMP1` base vocoder
- the real artifact-backed `video.condition.audio` path now passes the safe rung and preserves the resolved reference audio into the muxed output path instead of silently ignoring audio handles
- the text-first natural-audio path now has its first stronger success: the 10-second dog showcase receipt under `tmp/showcase-runs/20260308T034004Z-showcase-dog-park-natural/` clears the Gemini video-plus-audio review gate after the family-local negative-guidance plus rescaled-CFG slice
- the bridge now enforces the post-connector prompt contract and fails closed on 22B transformer-feature drift, missing VAE per-channel statistics, and unsupported x2 upsampler layouts instead of silently approximating them
- the repo-owned fidelity workflow in `scripts/ltx_debug_smoke.py` now passes a fixed-seed clear-dog `384x224 / 17f / 24fps` rung and a coherent `768x512 / 33f / 24fps` rung against the real Hugging Face-converted artifact set, with `stage1`, `post_x2`, and `final` snapshots saved in the run bundle
- the workflow-orchestration split is now strong enough to document as core `MLXR` doctrine, but it is still not freeze-ready until at least one second family survives the same core/family/host split
- this moves the next blocker forward again: the remaining truth gap is no longer prompt encoding, preview generation, silent bridge drift, first subject formation, silent audio export, the first `video.condition.audio` bridge slice, raw BWE audio fidelity, or the first natural-audio dog scene, but broader scene coverage for text-first natural-audio realism, conditioned-scene quality on `video.condition.audio`, broader capability completion beyond the distilled two-stage baseline, followed by recommended-resolution and HQ-profile validation on the real path

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

### Track 5: Workflow orchestration

- validate that a second family can declare truthful workflow templates without becoming a private scheduler
- validate that hosts consume the same runtime-owned workflow lifecycle without copying stage order
- pressure-test workflow state handoff, failure categories, and output completion semantics across more than LTX

### Track 6: Host parity

- CLI
- native daemon API
- desktop adapter
- Comfy execution bridge or adapter

Host parity is not “same feature buttons.” It is same lifecycle, same policy, and same truthful failures.

## Freeze Criteria

The architecture and native API can only be called stable when:

1. `LTX-2.3 Fast` T2V and I2V both pass the benchmark matrix.
2. One image diffusion family passes through the same source, artifact, scheduler, and workflow model.
3. One non-generation family passes through the same workflow-driven lifecycle.
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

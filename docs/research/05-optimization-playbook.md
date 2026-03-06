# Optimization Playbook

## Purpose

This playbook separates:

- production baseline tactics that should shape the runtime now
- family-specific defaults
- research bets worth testing on MLX
- native hotspot candidates
- Apple-specific output and memory-path optimizations

It also records what not to assume.

## 1. Production Baseline Optimizations

These are not optional “future tuning.” They should shape the runtime from the start.

### Stage-aware lifecycle control

- load only the stages that need to be live
- release memory between stages where quality and latency allow it
- keep per-stage telemetry so the scheduler can learn real budgets

### Selective quantization

- quantize only where the family tolerates it
- record quality deltas instead of assuming “4-bit faster” is enough evidence
- treat submodule-level quantization as a first-class tool

### Compile only stable repeated inner loops

- compile repeated pure inner loops, not orchestration
- bucket common shapes
- track compile count and cache hit rate

### Truthful cache ownership

- reuse portable artifacts where possible
- treat build and compile caches as machine-local
- do not confuse faster warm runs with portable artifact success

## 2. Apple-Specific Memory And Output Optimizations

These are some of the highest-value Apple-specific opportunities and were underplayed before.

### Unified-memory-aware scheduling

- use live memory telemetry
- keep admission conservative when stage spikes are uncertain
- measure active and peak memory per stage

### Output-path optimization

- reduce avoidable frame copies
- investigate `CVPixelBufferPool` with `IOSurface` backing, `CVMetalTextureCache`, and `VideoToolbox` handoff paths
- separate model time from encode or mux time in benchmarks

### Native media integration

- use Swift and Apple-native APIs where they are a better fit than generic Python media stacks
- keep this separate from the core family semantics so the runtime can evolve without API churn

## 3. Family-Specific Defaults

### LTX and video DiT families

Baseline tactics:

- stage-aware scheduler reservations
- careful stage splitting
- shape-bucketed compile on stable inner loops
- selective quantization
- tiled decode where quality and speed make sense

Do not assume:

- generic diffusion cache reuse papers transfer unchanged to LTX on MLX
- output encode cost is negligible

### Image diffusion

Baseline tactics:

- resolution buckets
- tiled or staged decode when needed
- selective quantization
- careful attention-helper profiling

### Speech and audio

Baseline tactics:

- chunk-aware scheduler classes
- reusable conditioning state where the family allows it
- fused preprocessing only after profiling
- explicit separation of model time from containerization or playback time

### Text and VLM

Baseline tactics:

- prompt or prefix caching when appropriate
- model-switch and warm-load measurements
- compatibility-facade translation only after the native path is clean

## 4. MLX-Compatible Research Bets

These are promising, but they are not architecture facts until benchmarked on the real workload.

### Video and DiT-oriented reuse methods

- `AdaCache`
- `FasterCache`
- `PAB`
- `HiCache`
- `ProCache`
- `TeaCache`
- `DeepCache`
- `T-GATE`

Classification:

- plausible and worth benchmarking: `AdaCache`, `FasterCache`, `PAB`, `TeaCache`
- plausible but more speculative or workload-sensitive: `HiCache`, `ProCache`, `DeepCache`, `T-GATE`

### Negative evidence and caveats

- naive adjacent-step reuse can hurt motion coherence
- temporal reuse can create artifact accumulation on long clips
- methods that look good on CUDA stacks may lose a large part of their headline speedup when ported to MLX
- shape churn can erase compile wins

## 5. Native Hotspot Candidate Shortlist

These are the most likely candidates for `runtime-mlx-ext` or `runtime-kernels`.

- VAE decode and tile decode
- patchify and unpatchify
- latent pack or unpack transforms
- attention or rotary helpers where profiling proves the need
- spectrogram and audio feature extraction
- vocoder-critical kernels
- Apple-native output-path glue

No hotspot should move into native code without:

- trace evidence
- a benchmark delta
- a maintenance owner

## 6. What Is Not Production-Ready Enough To Bake In

- experimental paged-attention assumptions for Apple or Metal paths
- compile-export portability assumptions
- recent DiT cache papers treated as guaranteed wins
- any optimization claim that does not separate model, decode, and encode time

## 7. Validation Rules

Every optimization experiment must record:

- speed delta
- memory delta
- quality delta
- compile count impact
- cache hit impact
- failure modes

If a method improves one benchmark but harms quality or stability on another, record that as a real result instead of smoothing it away.

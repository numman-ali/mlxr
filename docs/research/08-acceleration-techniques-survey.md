# Acceleration Techniques Survey

## Purpose

This survey classifies techniques by readiness and fit instead of mixing production tactics, research headlines, and Apple-specific system work into one bucket.

## Tier A: Production Baseline

These should shape the runtime without waiting for a research win.

| Technique | Family fit | MLX fit | Notes |
| --- | --- | --- | --- |
| stage-aware scheduling | all | strong | highest-leverage structural optimization |
| selective quantization | all | strong | requires family-specific quality checks |
| compile only stable repeated loops | all | strong | do not compile orchestration or side effects |
| build-cache separation from portable artifacts | all | strong | prevents wrong portability assumptions |
| zero-copy-ish media output investigation | video and audio | strong | Apple-specific and high-value |
| prompt or prefix caching | text and VLM | strong | useful but should not dominate video architecture |

## Tier B: Family-Specific Proven Candidates

These are worth planning for, but not as universal defaults.

| Technique | Family fit | MLX fit | Notes |
| --- | --- | --- | --- |
| tiled VAE decode | video and image | medium to strong | useful under memory pressure |
| low-memory staged load or unload | video, VLM, large audio | strong | already justified by current workloads |
| chunk-aware speech scheduling | speech and audio | strong | mandatory for long audio realism |
| reusable conditioning state | TTS, VLM, some video flows | medium | benchmark quality and correctness carefully |

## Tier C: Plausible MLX-Compatible Research Bets

These deserve benchmark tracks, not architecture commitments.

| Technique | Family fit | MLX fit today | Caveats |
| --- | --- | --- | --- |
| `AdaCache` | video DiT | plausible | verify quality on long clips |
| `FasterCache` | video DiT | plausible | verify real reuse benefit after MLX porting |
| `PAB` | attention-heavy video | plausible | integration complexity may be high |
| `HiCache` | video DiT | plausible but less proven | quality drift and workload sensitivity |
| `ProCache` | video DiT | plausible but less proven | not yet safe to assume as baseline |
| `TeaCache` | diffusion and video | plausible | still family-sensitive |
| `DeepCache` | diffusion | plausible | may not map cleanly to every DiT stack |
| `T-GATE` | diffusion and attention control | plausible | can change quality characteristics |

## Tier D: Speculative Or Not Production-Ready

These should not shape the first stable architecture.

| Technique | Why it stays speculative |
| --- | --- |
| compile-export portability as artifact content | MLX exporter remains experimental |
| broad paged-attention assumptions for Apple or Metal | not proven as a core runtime truth here |
| any paper headline speedup copied directly into MLX expectations | framework transfer loss is real |

## Apple-Specific Output-Path Work

This is not a side quest. For media workloads it may be one of the highest-value Apple-specific wins.

Candidate areas:

- `IOSurface`
- `CVPixelBuffer`
- `CVMetalTextureCache`
- `VideoToolbox`
- GPU to media-engine handoff with fewer extra copies

Why it matters:

- unified memory changes the economics of copies and handoffs
- output encode cost is large enough to distort “model-only” wins

## Negative Evidence To Carry Forward

- aggressive reuse methods can degrade motion or temporal coherence
- shape churn can erase compile benefits
- methods that assume CUDA-specific kernels or memory behavior may shrink sharply in benefit on MLX
- “faster” is not a win if encode, mux, or export dominates the new wall clock

## Highest-Leverage Breakthrough Areas

### Video and LTX

1. better scheduler reservations
2. Apple-native output path
3. stable shape buckets for compiled inner loops
4. validated video reuse methods that survive MLX implementation
5. native kernels only after profiling

### Audio and speech

1. chunk-aware scheduler classes
2. reusable conditioning state
3. fused preprocessing when profiling justifies it
4. clean output-path separation from encode or playback

### Text and VLM

1. prompt and prefix caching
2. batch-aware scheduling where it actually helps
3. truthful compatibility facades over the native contract

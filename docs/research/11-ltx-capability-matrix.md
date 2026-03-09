# LTX-2.3 Capability Matrix

## Purpose

This document is the canonical truth map for the current `LTX-2.3` family surface in `MLXR`.

Use it to answer six questions clearly:

1. what upstream `LTX-2.3` can do
2. how that capability maps into the generic runtime schema
3. whether the adapter advertises it yet
4. whether the implementation exists
5. whether it has been validated
6. whether the repo is allowed to promote it as a user-facing claim

Read this together with:

- [18-ltx-reference-map.md](18-ltx-reference-map.md) for the authoritative
  upstream reference per subsystem
- [19-ltx-compatibility-checklist.md](19-ltx-compatibility-checklist.md) for
  the exhaustive implementation and validation checklist

This matrix is capability-first. Resolution, clip length, and throughput promotion happen only after a capability row is green at the safe and coherence rungs.

## Status labels

- `implemented`: code path exists in the adapter or runtime today
- `safe-rung validated`: passes the current low-risk validation rung
- `coherence validated`: passes the mid-rung quality gate
- `promoted`: safe to describe as current repo truth
- `planned`: intentionally in scope, but not implemented yet
- `out of scope`: intentionally excluded from the current tranche

## Current matrix

Schema note:

- rows in the `MLXR schema shape` column are a mix of current active request
  shapes and proposed future mappings
- when a row is not yet implemented, treat that cell as the intended mapping,
  not as settled public contract truth
- today the active LTX workflow surface is still narrower than the full matrix

## Official upstream pipeline canon

This is the current official `LTX-2` pipeline set from the upstream README and
`ltx-pipelines` package, normalized into one canonical table.

| Upstream pipeline | Primary use | Upstream status | Core asset shape | Notes for `MLXR` |
| --- | --- | --- | --- | --- |
| `TI2VidTwoStagesPipeline` | text/image-to-video | production default | full `dev` checkpoint, Gemma text encoder, x2 spatial upsampler, distilled LoRA | this is the upstream default quality target |
| `TI2VidTwoStagesHQPipeline` | text/image-to-video | alternate high-quality variant | same as standard two-stage | same two-stage family with `res_2s`; upstream presents it as a higher-quality trade-off, not a universally dominant row |
| `TI2VidOneStagePipeline` | text/image-to-video | educational / prototyping | full `dev` checkpoint, Gemma text encoder | useful for parity and quick iteration, not the main product target |
| `DistilledPipeline` | fast text/image-to-video | fastest | distilled checkpoint, Gemma text encoder, x2 spatial upsampler | current `MLXR` proving path |
| `ICLoraPipeline` | video-to-video / strong-control image-to-video | current | distilled checkpoint, Gemma text encoder, x2 spatial upsampler, IC-LoRA | only works with distilled upstream |
| `KeyframeInterpolationPipeline` | image keyframe interpolation | current | full checkpoint, Gemma text encoder, x2 spatial upsampler, distilled LoRA, keyframes | broader control row, not yet in `MLXR` |
| `A2VidPipelineTwoStage` | audio-to-video | current | full checkpoint, Gemma text encoder, x2 spatial upsampler, distilled LoRA, input audio | current `MLXR` `video.condition.audio` is a narrower preserved-reference slice |
| `RetakePipeline` | regenerate a time region of an existing video | current | checkpoint, Gemma text encoder, source video; full or distilled behaviorally | editing-oriented row, not yet in `MLXR` |

Important upstream interpretation:

- the full `dev` checkpoint plus the two-stage family is the quality-first
  upstream path
- the HQ variant is the top-end upstream two-stage variant, but it is still a
  trade-off row rather than a universally dominant replacement for standard
  two-stage
- the standard two-stage row is still the production-default recommendation
- the distilled row is the speed-first path, not the quality ceiling
- temporal upscaler-backed flows are still future-facing in the upstream docs,
  not part of the current public pipeline set

| Capability | Upstream `LTX-2.3` surface | MLXR schema shape / planned mapping | Adapter advertised | Implementation | Validation | Current repo truth |
| --- | --- | --- | --- | --- | --- | --- |
| Distilled two-stage text-to-video | yes | `video.generate` | yes | implemented | coherence validated | promoted |
| Distilled two-stage image-to-video | yes | `video.condition.image` | yes | implemented | coherence validated | promoted |
| Audio-bearing output on AV path | yes | `video.generate` or `video.condition.image` with `artifact_format=mp4` or `wav` | yes | implemented | safe-rung validated; visual-gate BWE-enabled `48 kHz` dog clip confirmed | promoted |
| Silent video output | yes | `video.generate` or `video.condition.image` with `artifact_format=mp4` | yes | encoder behavior exists | not separately validated as a promoted user mode | implemented encoder path only |
| Audio-to-video conditioning | yes | `video.condition.audio` | yes | implemented | safe-rung validated; a 6-second dog/park/barking row and a 6-second anime-conditioned row both now pass through the real `mlxr` CLI path with strict Gemini review and correct stream receipts | promoted conditioned-audio row with preserved-reference caveat |
| Reference-video conditioning | yes | `video.condition.video` | no | planned | not started | not yet supported |
| Keyframe interpolation | yes | `video.interpolate` | no | planned | not started | not yet supported |
| Retake | yes | `video.retake` | no | planned | not started | not yet supported |
| One-stage T2V | yes | `video.generate` with `ltx.pipeline_variant=one_stage` | no | planned | not started | not yet supported |
| One-stage I2V | yes | `video.condition.image` with `ltx.pipeline_variant=one_stage` | no | planned | not started | not yet supported |
| Full two-stage / HQ T2V | yes | `video.generate` with `ltx.pipeline_variant=two_stage` or `two_stage_hq` | no | planned | not started | not yet supported |
| Full two-stage / HQ I2V | yes | `video.condition.image` with `ltx.pipeline_variant=two_stage` or `two_stage_hq` | no | planned | not started | not yet supported |
| IC-LoRA | yes | `ltx.control_variant=ic_lora` plus optional `lora` components | no | planned | not started | not yet supported |
| Union IC-LoRA | yes | `ltx.control_variant=union_ic_lora` plus optional `lora` components | no | planned | not started | not yet supported |
| Distilled LoRA support | yes | optional `lora` components | no | planned | not started | not yet supported |
| Prompt enhancement | ecosystem and product surface | `ltx.prompt_enhancement=*` | no | planned later | not started | not yet supported |
| STG controls | ecosystem and product surface | `ltx.stg=*` | no | planned later | not started | not yet supported |

## Output formats

Current output truth:

| Format | Meaning | Status |
| --- | --- | --- |
| `mp4` | runtime-managed video artifact; currently muxes audible audio when decoded audio is present | implemented |
| `wav` | runtime-managed audio-only artifact derived from the AV path | implemented |

Current non-truths:

- `wav` does not mean `audio-only job` support exists yet
- `audio` in `modalities_out` means the current AV bridge can now export audio, not that every LTX task is surfaced
- the current `video.condition.audio` row preserves reference audio through the output path and now has strict Gemini-passing conditioned rows for both a dog/park/barking scene and a non-dog anime rooftop scene through the real `mlxr` product path; it is now a real conditioned-scene capability, but it is still not yet a claim of broad conditioned-scene strength across every scene class

## Current upstream-versus-MLXR gap

The official upstream family surface is broader than the active `MLXR` surface.

Today `MLXR` has truly implemented:

- distilled two-stage text-to-video
- distilled two-stage image-to-video
- audio-bearing output on the AV path
- a now quality-promoted `video.condition.audio` preserved-reference row

Today `MLXR` does not yet expose the broader current upstream rows:

- standard two-stage on the full `dev` checkpoint
- two-stage HQ
- one-stage
- reference-video conditioning
- IC-LoRA
- keyframe interpolation
- retake

So the repo should not talk as if the current distilled path covers the whole
official `LTX-2.3` product surface. It does not.

## Validation ladder

These ladders are now capability-specific.

Every core capability row should eventually pass:

1. safe rung
   `384x224 / 17f / 24fps`
2. coherence rung
   `768x512 / 33f / 24fps`
3. recommended rung
   `1536x1024 / 121f / 24fps`
4. HQ first pass
   `1920x1088 / 65f / 24fps`

Current promoted rows only clear the first two rungs for:

- distilled two-stage T2V
- distilled two-stage I2V

Audio export is now real-weight validated at the safe and visual-gate rungs, the text-first AV path now uses the repo-owned MLX BWE wrapper at `48 kHz`, and `video.condition.audio` now passes strict conditioned-scene review on both dog and non-dog rows through the real `mlxr` path while still preserving reference audio. The next blocker has moved from raw audio fidelity and first conditioned-scene truth to broader text-first natural-audio coverage, additional conditioned scene classes, and capability completion.

Current showcase truth:

- the first promoted text-first five-clip showcase pack is the score-friendly set captured in `tmp/showcase-runs/showcase-pack-20260308.json`
- it proves that the current text-first AV path can carry distinct 10-second stylized scenes with good, audible audio at the safe rung
- the newer negative-guidance/rescaled-CFG path now gives the first successful 10-second natural-audio dog scene at `tmp/showcase-runs/20260308T034004Z-showcase-dog-park-natural/`
- the newer showcase-runner shaping slice clearly rescues `heron_marsh_documentary` as a no-music Gemini match at the 10-second `384x224 / 241f / 24fps` rung
- separate current repo receipts also show no-music Gemini matches for `basketball_court_dusk` and `cafe_sidewalk_human` at that same rung
- the current best-available promoted pack that incorporates that dog recovery is `tmp/showcase-runs/showcase-pack-20260308-best-available.json`
- it still does not prove that text-first natural-audio realism is solved broadly; passive animal ambience and vintage-natural scenes can still drift back into music-like audio, as shown by the current `cat_kitchen_natural` and rerun `vintage_laundromat` receipts

## Recommended next implementation order

If the repo priority is “highest quality achievable on Mac,” the next target
should not be more distilled-only polishing. But it also should not be layering
new non-distilled claims on top of donor runtime code that still powers the
current proving slice. The next target is the owned-substrate migration
described in [16-owned-substrate-migration-plan.md](16-owned-substrate-migration-plan.md),
followed immediately by the upstream full-checkpoint two-stage family.

Recommended order:

1. remove runtime `mlx-video` usage from the promoted distilled engine and keep
   the current proving slice green on repo-owned code
2. remove prompt-path `mlx-vlm` / `mlx-lm` usage after the generation engine is
   owned
3. land standard two-stage on the full `dev` checkpoint as the owned
   production-quality target
4. land `two_stage_hq` as the explicit top-end alternate high-quality variant
5. improve `video.condition.audio` scene quality from “real bridge slice” to
   “quality-promoted capability”
6. land `video.condition.video`
7. land `ICLoraPipeline` semantics for strong-control image/video conditioning
8. land `video.interpolate`
9. land `video.retake`
10. land one-stage as the educational / prototyping row
11. land longer-form and profile promotion only after those pipeline rows are
   truthful

## Notes

- The real `LTX-2.3` checkpoint currently used by `MLXR` already contains the audio VAE and vocoder weights, so audio export belongs to the core checkpoint-backed slice and does not require inventing extra artifact roles.
- The current audible MLX path now uses the checkpoint's full BWE wrapper on top of the `AMP1` base vocoder contract, and the first visual-gate receipt for that path is the dog clip under `tmp/manual-runs/20260308T014947Z-dog-bwe-visual-gate-check/`.
- The current truthful `video.condition.audio` slice preserves the resolved reference audio through the output path while conditioning generation on that runtime-managed audio handle. It now clears the same strict Gemini visual/audio bar on both a dog scene and a non-dog anime scene through the real `mlxr` product path, so the row is no longer just a bridge proof.
- The current workflow and adapter path already carry image references through `video.condition.audio`, which matches the official upstream direction for combined text + image + audio inputs. That combined slice is now covered by runtime workflow tests, but it is not yet promoted as a quality-validated showcase capability.
- The right canonical language is now: distilled is the current proving slice,
  standard two-stage is the next production-quality target, and HQ is the
  top-end alternate high-quality two-stage variant. Do not collapse those three
  into one vague “best” row.
- Landing the non-distilled two-stage family is not just a checkpoint swap. It
  also requires the supporting scheduler, sampler, guidance, and LoRA contract
  work that the current distilled path does not yet need, and it should now be
  built on the owned-substrate migration path instead of extending the donor
  runtime.
- Capability coverage and profile promotion are different gates. A capability may be implemented and still not be quality-promoted at recommended or HQ sizes.

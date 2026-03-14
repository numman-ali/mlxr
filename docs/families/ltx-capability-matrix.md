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

- [ltx-reference-map.md](./ltx-reference-map.md) for the authoritative
  upstream reference per subsystem
- [ltx-compatibility-checklist.md](./ltx-compatibility-checklist.md) for
  the exhaustive implementation and validation checklist

This matrix is capability-first. Resolution, clip length, and throughput promotion happen only after a capability row is green at the safe and coherence rungs.

## Status labels

- `implemented`: code path exists in the adapter or runtime today
- `safe-rung validated`: passes the current low-risk validation rung
- `meaningful-rung validated`: completes a higher-cost, human-meaningful receipt
  such as a `6s` clip at `768x448`, but still falls short of promotion because
  quality or semantics are not yet strong enough
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
| `ICLoraPipeline` | reference-video conditioning / strong-control image-to-video | current | distilled checkpoint, Gemma text encoder, x2 spatial upsampler, IC-LoRA | upstream treats this as the reference-video row; it can also use image conditioning for strong control and only works with distilled upstream |
| `KeyframeInterpolationPipeline` | image keyframe interpolation | current | full checkpoint, Gemma text encoder, x2 spatial upsampler, distilled LoRA, keyframes | broader control row; now implemented on the owned runtime path but still unvalidated |
| `A2VidPipelineTwoStage` | audio-to-video | current | full checkpoint, Gemma text encoder, x2 spatial upsampler, distilled LoRA, input audio | current `MLXR` `video.condition.audio` is a narrower preserved-reference slice |
| `RetakePipeline` | regenerate a time region of an existing video | current | checkpoint, Gemma text encoder, source video; full or distilled behaviorally | editing-oriented row; now implemented on the owned runtime path but still unvalidated |

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
| Distilled keyed-image I2V on the current image-conditioned row | yes | `video.condition.image` with multiple image refs carrying per-image `frame_index` and `strength` metadata | yes | implemented | manual first/last-frame CLI receipt validated | real current control slice; not the same thing as official keyframe interpolation |
| Audio-bearing output on AV path | yes | `video.generate` or `video.condition.image` with `artifact_format=mp4` or `wav` | yes | implemented | safe-rung validated; visual-gate BWE-enabled `48 kHz` dog clip confirmed | promoted |
| Silent video output | yes | `video.generate` or `video.condition.image` with `artifact_format=mp4` | yes | encoder behavior exists | not separately validated as a promoted user mode | implemented encoder path only |
| Audio-to-video conditioning | yes | `video.condition.audio` | yes | implemented | safe-rung validated; a 6-second dog/park/barking row and a 6-second anime-conditioned row both now pass through the real `mlxr` CLI path with strict Gemini review and correct stream receipts | promoted conditioned-audio row with preserved-reference caveat |
| Reference-video conditioning via `ICLoraPipeline` | yes | `video.condition.video` plus `ltx.control_variant=ic_lora`, `union_ic_lora`, or `motion_track_control` | yes, on fast artifacts | implemented on the owned runtime path | meaningful-rung validated at `768x448 / 145f / 24fps` on the union-control slice, with a Gemini `match` | real fast/distilled control row at the union-control slice; broader IC-LoRA coverage is still not fully promoted |
| Keyframe interpolation | yes | `video.interpolate` | yes, on `dev` artifacts that include the x2 upsampler and distilled LoRA | implemented on the owned runtime path | meaningful-rung validated at `768x448 / 145f / 24fps`; current Gemini verdict is `mismatch` | runtime-stable at the first meaningful rung, but keyframe adherence is still too weak for promotion |
| Retake | yes | `video.retake` | yes | implemented on the owned runtime path | meaningful-rung validated at `768x448 / 145f / 24fps`; current Gemini verdict is `mismatch` | runtime-stable at the first meaningful rung, but not yet quality-promoted |
| One-stage T2V | yes | `video.generate` with `ltx.workflow_variant=one_stage` | yes, on `dev` artifacts | implemented on the owned runtime path | meaningful-rung validated at `768x448 / 145f / 24fps`; current Gemini verdict is only `partial` | runtime-stable at the first meaningful rung, but currently much weaker than two-stage and not quality-promoted |
| One-stage I2V | yes | `video.condition.image` with `ltx.workflow_variant=one_stage` | yes, on `dev` artifacts | implemented on the owned runtime path | not started | implemented on the `dev` checkpoint path, but not yet validated or promoted |
| Standard two-stage T2V | yes | `video.generate` with `ltx.workflow_variant=two_stage` | yes, on `dev` artifacts that include the x2 upsampler and distilled LoRA | implemented on the owned runtime path | meaningful-rung validated at `768x448 / 145f / 24fps`; current Gemini verdict is still `mismatch` | runtime-stable at the first meaningful rung, but not yet quality-promoted |
| Standard two-stage I2V | yes | `video.condition.image` with `ltx.workflow_variant=two_stage` | yes, on `dev` artifacts that include the x2 upsampler and distilled LoRA | implemented on the owned runtime path | not started | implemented on the `dev` checkpoint path, but not yet validated or promoted |
| HQ two-stage T2V | yes | `video.generate` with `ltx.workflow_variant=two_stage_hq` | yes, on `dev` artifacts that include the x2 upsampler and distilled LoRA | implemented on the owned runtime path | meaningful-rung validated at `768x448 / 145f / 24fps`; current Gemini verdict is still `mismatch` | runtime-stable at the first meaningful rung, but not yet quality-promoted |
| HQ two-stage I2V | yes | `video.condition.image` with `ltx.workflow_variant=two_stage_hq` | yes, on `dev` artifacts that include the x2 upsampler and distilled LoRA | implemented on the owned runtime path | not started | implemented on the `dev` checkpoint path, but not yet validated or promoted |
| IC-LoRA | yes | `ltx.control_variant=ic_lora` plus optional `lora` components | yes, on fast artifacts | implemented on the owned runtime path | not started | generic distilled IC-LoRA lane remains unvalidated separately from the union-control slice |
| Union IC-LoRA | yes | `ltx.control_variant=union_ic_lora` plus optional `lora` components | yes, on fast artifacts | implemented on the owned runtime path | meaningful-rung validated at `768x448 / 145f / 24fps`; current Gemini verdict is `match` | current strongest validated fast control slice |
| Motion-track IC-LoRA | yes | `ltx.control_variant=motion_track_control` plus optional `lora` components | yes, on fast artifacts | implemented as the same owned IC-LoRA runtime lane with a distinct control-variant label | meaningful-rung validated at `768x448 / 145f / 24fps`; current Gemini verdict is `partial` | real product-surface slice, but still weaker than union-control and not yet promoted on its own |
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

Today `MLXR` does not yet promote the broader current upstream rows:

- validation and promotion of standard two-stage on the full `dev` checkpoint
- validation and promotion of two-stage HQ
- validation and promotion of reference-video conditioning via `ICLoraPipeline`
- validation and promotion of IC-LoRA

Today `MLXR` also has one important partial row:

- one-stage is now implemented and artifactized on the full `dev` checkpoint
  path, and it now has a first meaningful `6s` receipt at `768x448 / 145f /
  24fps`; that row is still not promotable because the current visual quality
  remains much weaker than two-stage
- keyframe interpolation is now implemented and artifactized on the full `dev`
  two-stage path, but it still lacks real validation receipts and should not
  be promoted yet
- standard two-stage and HQ now both complete a first meaningful `6s` dev
  receipt at `768x448 / 145f / 24fps`, but the current Gemini verdict for both
  rows is still `mismatch`, so neither row is quality-promoted yet
- retake now also completes a first meaningful `6s` receipt at
  `768x448 / 145f / 24fps`, but the current retake edit does not yet land the
  requested action change strongly enough for promotion
- `video.condition.video` now also completes meaningful `6s` fast-path receipts
  at `768x448 / 145f / 24fps`; the `union_ic_lora` slice currently passes with
  a Gemini `match`, while `motion_track_control` is real but still only
  `partial`
- `video.interpolate` now also completes a meaningful `6s` receipt at
  `768x448 / 145f / 24fps`, but the first fox-keyframe run ignored the start
  and end reference frames badly enough that it remains unpromoted

Artifact/runtime variant note:

- `LTX-2.3-fp8` is a dev-checkpoint precision variant, not a new capability row
- it should be tracked as a future artifact/runtime variant on the same
  `one_stage`, `two_stage`, `two_stage_hq`, `video.interpolate`, and `video.retake`
  task surface rather than as a separate family or task

Official-guidance note:

- upstream `DistilledPipeline` and `ICLoraPipeline` use fixed distilled
  schedules in practice, even though their CLI parsers still expose a
  `--num-inference-steps` flag
- for truthful validation, treat distilled fast rows and IC-LoRA as the fixed
  `8`-sigma stage-1 plus `4`-sigma stage-2 schedule unless upstream changes
  the implementation

So the repo should not talk as if the current distilled path covers the whole
official `LTX-2.3` product surface. It does not.

## Reference-video versus retake

The most important naming distinction to keep straight is:

- `video.condition.video` is the planned `MLXR` label for the official
  upstream `ICLoraPipeline` row
- that row means generation conditioned on one or more reference videos through
  IC-LoRA semantics, not generic "any video edit"
- `video.retake` is the planned `MLXR` label for the separate official
  `RetakePipeline` row
- retake is time-window regeneration of an existing video, with distinct
  temporal-mask semantics and different user-facing guarantees

So if the repo says `video.condition.video`, it should be read as
"reference-video conditioning aligned to `ICLoraPipeline`", not as a vague
catch-all for every upstream video-input feature.

## Official desktop product pressure

The official `ltx-desktop` product surface is now a second important pressure
source alongside the upstream `LTX-2` pipeline docs.

Current official desktop expectations:

- fast text-to-video
- fast image-to-video
- audio-to-video
- retake
- optional API-backed `Z Image Turbo` text-to-image generation

Current official desktop non-truths for `MLXR`:

- the desktop shell does not prove that every advertised product mode is
  already covered by the local `MLXR` runtime
- desktop `video-to-video` expectations should map to the official
  `ICLoraPipeline` row, not to a made-up generic video-input contract
- desktop retake expectations should map to the official `RetakePipeline`
  semantics, not to `video.condition.video`
- desktop support is not just "run distilled generation on macOS"; it also
  requires a thin compatibility adapter for the desktop backend HTTP contract

The right repo framing is:

- the current promoted local `MLXR` story covers distilled T2V, I2V, and the
  narrower preserved-reference `video.condition.audio` row
- the official desktop product surface is broader than the currently promoted
  local story, though retake, interpolation, and IC-LoRA-backed
  `video.condition.video` are now implemented locally and still waiting on real
  validation
- covering the full desktop-capable `LTX` surface means closing real upstream
  capability gaps, not only adding a host adapter

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

## Official default profiles

For future validation, keep the upstream defaults in mind:

- the shared baseline is `121` frames at `24fps`, which is just over `5s`
- one-stage defaults to `512x768`
- standard two-stage defaults to `1024x1536`
- HQ defaults to `1088x1920`
- non-distilled `LTX-2.3` rows use `30` stage-1 steps with video CFG `3.0`
  and audio CFG `7.0`
- HQ uses `15` stage-1 steps, disables STG, and keeps stage-specific
  distilled-LoRA strengths at `0.25` and `0.5`
- distilled fast rows and IC-LoRA use fixed distilled schedules in practice:
  `8` stage-1 sigmas plus `4` stage-2 sigmas

So the current `768x448 / 145f / 24fps` clips are best understood as a
meaningful dev rung for local runtime truth, not as the official production
profile.

Current showcase truth:

- the first promoted text-first five-clip showcase pack is the score-friendly set captured in `tmp/showcase-runs/showcase-pack-20260308.json`
- it proves that the current text-first AV path can carry distinct 10-second stylized scenes with good, audible audio at the safe rung
- the historical dog recovery receipt at `tmp/showcase-runs/20260308T034004Z-showcase-dog-park-natural/` came from an earlier family-local CFG experiment, but the promoted owned distilled product path now defaults back to positive-only denoising
- some older showcase receipts generated under repo-owned prompt-shaping experiments clear no-music review bars, but those should now be treated as historical evidence rather than as the current pass-through prompt contract
- the current best-available promoted pack that incorporates that dog recovery is `tmp/showcase-runs/showcase-pack-20260308-best-available.json`
- it still does not prove that text-first natural-audio realism is solved broadly; passive animal ambience and vintage-natural scenes still need fresh validation on the current pass-through prompt contract, and conditioned audio remains the stronger promoted control path when exact sound behavior matters

## Recommended next implementation order

If the repo priority is “highest quality achievable on Mac,” the next target
should not be more distilled-only polishing. But it also should not be layering
new non-distilled claims on top of donor runtime code that still powers the
current proving slice. The next target is the owned-substrate migration
described in [16-owned-substrate-migration-plan.md](../research/16-owned-substrate-migration-plan.md),
followed immediately by the upstream full-checkpoint two-stage family.

Recommended order:

1. remove runtime `mlx-video` usage from the promoted distilled engine and keep
   the current proving slice green on repo-owned code
2. remove prompt-path `mlx-vlm` / `mlx-lm` usage after the generation engine is
   owned
3. validate and, if earned, promote standard two-stage on the full `dev`
   checkpoint as the owned production-quality target
4. validate and, if earned, promote `two_stage_hq` as the explicit top-end
   alternate high-quality variant
5. keep improving `video.condition.audio` scene quality beyond the current
   promoted conditioned-audio wins
6. land `video.condition.video` as the official `ICLoraPipeline`
   reference-video row
7. land the broader `ICLoraPipeline` control semantics for strong-control
   image/video conditioning
8. validate and, if earned, promote `video.interpolate`
9. validate and, if earned, promote `video.retake`
10. validate and, if earned, promote one-stage as the educational /
    prototyping row
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

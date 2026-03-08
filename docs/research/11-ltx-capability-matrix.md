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

This matrix is capability-first. Resolution, clip length, and throughput promotion happen only after a capability row is green at the safe and coherence rungs.

## Status labels

- `implemented`: code path exists in the adapter or runtime today
- `safe-rung validated`: passes the current low-risk validation rung
- `coherence validated`: passes the mid-rung quality gate
- `promoted`: safe to describe as current repo truth
- `planned`: intentionally in scope, but not implemented yet
- `out of scope`: intentionally excluded from the current tranche

## Current matrix

| Capability | Upstream `LTX-2.3` surface | MLXR schema shape | Adapter advertised | Implementation | Validation | Current repo truth |
| --- | --- | --- | --- | --- | --- | --- |
| Distilled two-stage text-to-video | yes | `video.generate` | yes | implemented | coherence validated | promoted |
| Distilled two-stage image-to-video | yes | `video.condition.image` | yes | implemented | coherence validated | promoted |
| Audio-bearing output on AV path | yes | `video.generate` or `video.condition.image` with `artifact_format=mp4` or `wav` | yes | implemented | safe-rung validated; visual-gate BWE-enabled `48 kHz` dog clip confirmed | promoted |
| Silent video output | yes | `video.generate` or `video.condition.image` with `artifact_format=mp4` | yes | implemented | coherence validated | promoted |
| Audio-to-video conditioning | yes | `video.condition.audio` | yes | implemented | safe-rung validated; real bridge preserves resolved reference audio into muxed output; scene semantics still provisional | promoted with current passthrough-audio caveat |
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
- the current `video.condition.audio` row preserves reference audio through the output path; it is a truthful preserved-reference capability, not yet a claim of broader reference-video control or strong conditioned-scene semantics

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

Audio export is now real-weight validated at the safe and visual-gate rungs, the text-first AV path now uses the repo-owned MLX BWE wrapper at `48 kHz`, `video.condition.audio` now passes the real safe-rung bridge with preserved reference audio, and the newer negative-guidance/rescaled-CFG path now gives the first successful 10-second natural-audio dog scene at `384x224 / 241f / 24fps`. The next blocker has moved from raw audio fidelity to broader conditioned-scene quality and capability completion.

Current showcase truth:

- the first promoted text-first five-clip showcase pack is the score-friendly set captured in `tmp/showcase-runs/showcase-pack-20260308.json`
- it proves that the current text-first AV path can carry distinct 10-second stylized scenes with good, audible audio at the safe rung
- the newer negative-guidance/rescaled-CFG path now gives the first successful 10-second natural-audio dog scene at `tmp/showcase-runs/20260308T034004Z-showcase-dog-park-natural/`
- it does not yet prove that text-first natural-audio realism is solved broadly; nature and natural-vintage scenes still need to clear the same video-plus-audio Gemini review bar on the newer path

## Recommended next implementation order

1. broaden the newer text-first natural-audio guidance path beyond the dog scene and revalidate nature/vintage natural scenes
2. improve `video.condition.audio` scene quality from “real bridge slice” to “quality-promoted capability”
3. `video.condition.video`
4. `video.interpolate`
5. `video.retake`
6. one-stage T2V and I2V
7. full two-stage / HQ variants
8. IC-LoRA and distilled LoRA support
9. recommended, HQ, and longer-clip promotion for every green capability row

## Notes

- The real `LTX-2.3` checkpoint currently used by `MLXR` already contains the audio VAE and vocoder weights, so audio export belongs to the core checkpoint-backed slice and does not require inventing extra artifact roles.
- The current audible MLX path now uses the checkpoint's full BWE wrapper on top of the `AMP1` base vocoder contract, and the first visual-gate receipt for that path is the dog clip under `tmp/manual-runs/20260308T014947Z-dog-bwe-visual-gate-check/`.
- The first truthful `video.condition.audio` slice preserves the resolved reference audio through the output path while conditioning generation on that runtime-managed audio handle. It is a real bridge capability, but its scene semantics should still be treated as provisional until the conditioned path clears the same visual/audio review bar as the text-first dog ladder.
- Capability coverage and profile promotion are different gates. A capability may be implemented and still not be quality-promoted at recommended or HQ sizes.

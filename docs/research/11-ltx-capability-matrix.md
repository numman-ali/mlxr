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
| Audio-bearing output on AV path | yes | `video.generate` or `video.condition.image` with `artifact_format=mp4` or `wav` | yes | implemented | safe-rung validated; visual-gate audible clip confirmed; full BWE parity still pending | promoted with current base-vocoder caveat |
| Silent video output | yes | `video.generate` or `video.condition.image` with `artifact_format=mp4` | yes | implemented | coherence validated | promoted |
| Audio-to-video conditioning | yes | `video.condition.audio` | yes | implemented | safe-rung validated; real bridge preserves resolved reference audio into muxed output; full BWE parity still pending | promoted with current passthrough-audio caveat |
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
- the current `video.condition.audio` row preserves reference audio through the output path; it does not yet claim full upstream BWE parity or broader reference-video control

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

Audio export is now real-weight validated at the safe and visual-gate rungs, and `video.condition.audio` now passes the real safe-rung bridge with preserved reference audio. Full `LTX-2.3` BWE parity is still pending.

## Recommended next implementation order

1. full `LTX-2.3` BWE audio parity on the current `video.condition.audio` path
2. `video.condition.video`
3. `video.interpolate`
4. `video.retake`
5. one-stage T2V and I2V
6. full two-stage / HQ variants
7. IC-LoRA and distilled LoRA support
8. recommended, HQ, and longer-clip promotion for every green capability row

## Notes

- The real `LTX-2.3` checkpoint currently used by `MLXR` already contains the audio VAE and vocoder weights, so audio export belongs to the core checkpoint-backed slice and does not require inventing extra artifact roles.
- The current audible MLX path uses the checkpoint's `AMP1` base vocoder contract. The remaining audio-fidelity gap is the upstream BWE wrapper and its mel/STFT residual path.
- The first truthful `video.condition.audio` slice currently preserves the resolved reference audio through the output path while conditioning generation on that runtime-managed audio handle. It is a real bridge capability, but it is not yet a claim of full upstream audio-fidelity parity beyond the current passthrough/base-vocoder path.
- Capability coverage and profile promotion are different gates. A capability may be implemented and still not be quality-promoted at recommended or HQ sizes.

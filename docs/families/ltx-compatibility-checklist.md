# LTX Compatibility Checklist

## Purpose

This is the exhaustive working checklist for `LTX-2.3` compatibility in
`MLXR`.

It exists to stop feature drift and memory-based planning. If a row matters for
official `LTX-2`, it should appear here even if the answer is still “planned.”

Use this with:

- [ltx-capability-matrix.md](./ltx-capability-matrix.md) for the current
  public truth
- [ltx-reference-map.md](./ltx-reference-map.md) for the authoritative
  source of each subsystem

## Status keys

- `[x]` promoted and defended in current repo truth
- `[~]` implemented or partially landed, but not promoted
- `[ ]` planned or not started

## A. Canon and guidance

- [x] The official `LTX-2` public pipeline set is explicitly captured in repo
  canon.
- [x] The repo distinguishes the promoted distilled proving slice from the full
  official family surface.
- [x] The repo distinguishes standard two-stage from HQ instead of collapsing
  them into one vague “best quality” row.
- [x] The repo documents the donor-removal program for `mlx-video`.
- [x] The repo documents the prompt-substrate replacement program for the
  shared core MLX model substrate.
- [x] The repo has a dedicated LTX prompting guide.
- [x] The repo has a dedicated owned-substrate migration plan.
- [x] The repo has a dedicated multimodal-engine direction doc.
- [x] The repo now has an explicit LTX reference map.
- [x] The repo now has an explicit exhaustive compatibility checklist.

## B. Official pipeline rows

- [x] `DistilledPipeline` is represented in repo truth.
- [~] `TI2VidTwoStagesPipeline` is implemented on the owned engine, but not yet
  validated or promoted.
- [~] `TI2VidTwoStagesHQPipeline` is implemented on the owned engine, but not
  yet validated or promoted.
- [~] `TI2VidOneStagePipeline` is implemented on the owned engine, but not yet
  validated or promoted.
- [~] `ICLoraPipeline` semantics are implemented, but not yet validated or
  promoted.
- [~] `KeyframeInterpolationPipeline` semantics are implemented, but not yet
  validated or promoted.
- [~] `A2VidPipelineTwoStage` is represented by a narrower
  `video.condition.audio` preserved-reference slice, now quality-promoted for
  current dog and anime conditioned rows.
- [~] `RetakePipeline` semantics are implemented, but not yet validated or
  promoted.

## C. Asset and checkpoint contract

- [x] Distilled checkpoint support is real.
- [~] Dev checkpoint support is real for the current one-stage lane; the dev
  artifact contract no longer requires the x2 upsampler.
- [x] Gemma text encoder support is real.
- [x] x2 spatial upsampler support is real.
- [ ] x1.5 spatial upsampler support is implemented.
- [ ] Temporal upscaler support is implemented.
- [x] Distilled LoRA is modeled explicitly as the internal asset contract for
  standard/HQ rows.
- [~] Standard/HQ rows fail closed when distilled LoRA is absent; standard
  two-stage now does, while HQ is still not implemented.
- [ ] User-facing LoRA refs remain rejected until real LoRA-control rows exist.
- [~] IC-LoRA asset family is modeled explicitly through the current fast-path
  LoRA input contract, but not yet validated or promoted.
- [x] Motion-track IC-LoRA is either supported explicitly or documented as a
  control-variant extension on the existing `video.condition.video` row.
- [x] `LTX-2.3-fp8` is either supported explicitly or documented as a dev
  artifact/runtime variant rather than a new task.
- [ ] Camera-control LoRAs are either mapped explicitly or documented as out of
  scope.

## D. Donor-runtime removal

- [x] The promoted distilled path now owns latent upsampling, audio decoder
  loading, the audio encoder/decoder model stack, the audio latent patchify /
  unpatchify and per-channel statistics contract, VAE encoder loading, the x2
  spatial upsampler path, video unpatchify, encoder-side video-VAE substrate,
  decoder-side tiling and tile blending, the decoder-side convolution /
  timestep / residual / upsampling blocks, the transformer-side rotary
  embeddings, the transformer-side timestep embedding / AdaLN / transformer
  preprocessors, the attention module, the basic AV transformer block, the
  top-level `LTXModel` bridge, and the donor utility sanitizers and RMS norm
  helper it previously imported.
- [x] No runtime `mlx_video.*` import remains in the promoted distilled path.
- [x] No execution-path `sys.path` injection into the repo-local `mlx-video`
  checkout remains.
- [x] Distilled backend receipts use only `MLXR`-owned identities.
- [x] No runtime dependency on `mlx-vlm` remains.
- [x] No runtime dependency on `mlx-lm` remains.
- [x] The Gemma prompt stack runs on a repo-owned substrate aligned with doc 17.

## E. Scheduler and sampler substrate

- [x] Distilled fixed-sigma path is real.
- [x] Owned non-distilled scheduler exists.
- [x] Token-count-aware schedule logic is implemented.
- [~] Standard guided Euler loop is implemented for the current one-stage full
  checkpoint family.
- [~] HQ `res_2s` loop is implemented.
- [ ] HQ per-step noise keys evolve correctly.
- [ ] Gradient-estimation Euler is either implemented or explicitly deferred.
- [ ] Skip-step logic is either implemented or explicitly deferred.

## F. Guidance substrate

- [x] Distilled path truthfully remains guidance-light compared with upstream
  non-distilled rows.
- [~] Full non-distilled CFG guidance is implemented for the current one-stage
  row.
- [ ] STG substrate is implemented.
- [ ] STG is wired through a real request contract before promotion.
- [ ] Modality isolation guidance is implemented.
- [ ] Rescaling is implemented for the non-distilled family.
- [ ] Multimodal guider factory behavior is implemented or explicitly deferred.
- [ ] Advanced guidance controls remain unpromoted until validated.

## G. Core generation modes

- [x] Distilled text-to-video is promoted.
- [x] Distilled image-to-video is promoted.
- [~] Standard two-stage text-to-video is implemented.
- [~] Standard two-stage image-to-video is implemented.
- [~] HQ text-to-video is implemented and explicitly selectable.
- [~] HQ image-to-video is implemented and explicitly selectable.
- [~] One-stage text-to-video is implemented.
- [~] One-stage image-to-video is implemented.
- [~] Standard two-stage now has a first meaningful `6s / 768x448 / 145f`
  runtime receipt, but it is still not quality-promoted.
- [~] HQ two-stage now has a first meaningful `6s / 768x448 / 145f` runtime
  receipt, but it is still not quality-promoted.
- [~] One-stage now has a first meaningful `6s / 768x448 / 145f` runtime
  receipt, but it is still far from promotion quality.
- [~] Retake now has a first meaningful `6s / 768x448 / 145f` runtime receipt,
  but the edit semantics are still not strong enough for promotion.

## H. Audio modes

- [x] Audio-bearing output is real.
- [x] `wav` export is real as an output mode.
- [x] Checkpoint-backed BWE path is real when the checkpoint supports it.
- [x] `video.condition.audio` is real as a preserved-reference bridge slice.
- [x] `video.condition.audio` is quality-promoted as a conditioned-scene row.
- [ ] Text-first natural-audio realism is broad enough to promote beyond the
  current known-good scenes.
- [ ] Audio-to-video semantics match the upstream `A2VidPipelineTwoStage`
  family closely enough to promote the row.

## I. Conditioning and control modes

- [x] Single-image latent-replacement conditioning exists.
- [~] Keyframe-style additive image conditioning is implemented.
- [x] Multiple image conditioning is implemented and validated.
- [~] Reference-video conditioning is implemented.
- [ ] Combined text + image + audio conditioning is promoted truthfully.
- [ ] Combined-input support is mapped explicitly rather than described as vague
  “ingredients.”
- [x] Start/end-frame semantics are documented honestly: the current first/last
  frame story is a keyed-image bridge on `video.condition.image`, while
  official interpolation and retake remain separate future rows.

## J. Editing and advanced official rows

- [~] Keyframe interpolation is implemented.
- [~] Retake is implemented.
- [~] IC-LoRA video-to-video control is implemented.
- [~] Motion-track control is now accepted as a distinct IC-LoRA control
  variant on the same product surface, but it is still unvalidated.
- [~] Union-control now has a first meaningful `6s / 768x448 / 145f` match
  receipt on `video.condition.video`.
- [~] Motion-track control now has a first meaningful `6s / 768x448 / 145f`
  receipt, but it is still weaker than union-control.
- [~] Interpolation now has a first meaningful `6s / 768x448 / 145f` receipt,
  but keyframe adherence is still too weak for promotion.
- [ ] Strong-control image/video semantics are validated broadly enough to
  promote beyond the current union-control slice.

## K. Prompt and workflow layer

- [x] Default user story remains one prompt plus optional refs.
- [x] The promoted runtime path passes caller-authored prompts through verbatim.
- [ ] Prompt enhancement is implemented as a real workflow stage.
- [ ] Prompt enhancement is promoted only after real runtime validation.
- [ ] Workflow docs remain honest about planning vs orchestration.

## L. Outputs and profiles

- [x] H.264 MP4 is the default export path.
- [x] AAC audio muxing is real.
- [x] `+faststart` style portability defaults are preserved.
- [x] Safe-rung validation policy exists.
- [x] Coherence-rung validation policy exists.
- [ ] Recommended rung is promoted for the right capability rows.
- [ ] HQ rung is promoted for the right capability rows.
- [ ] Longer-form promotion is real for the right capability rows.
- [ ] Landscape validation is complete where it matters.
- [ ] Portrait validation is complete where it matters.
- [ ] Square validation is complete where it matters.

## M. Showcase and product proof

- [x] The current best-available showcase pack is documented canonically.
- [x] Gemini XML review is part of showcase promotion.
- [x] `ffprobe` receipts are part of showcase promotion.
- [ ] The showcase pack has been replaced by a stronger owned-engine pack.
- [ ] At least one strong natural-audio realism clip exists.
- [ ] Natural-audio realism is broad enough to promote beyond the current best
  scene-level wins.
- [ ] A conditioning-driven showcase row exists once the corresponding
  capability is real.

## N. Platform follow-through

- [x] The workflow layer is documented as planning, not full orchestration.
- [ ] Worker execution is generalized from the current fixed LTX-shaped sequence
  to adapter-declared stage graphs.
- [ ] A second family validates the broader platform assumptions.
- [ ] The benchmark matrix is being used for freeze-grade decisions rather than
  one-off demos.

## Next promotion gates

The next critical gates, in order, are:

1. [x] finish `mlx-vlm` / `mlx-lm` removal for the Gemma prompt path
2. [x] decide tokenizer policy explicitly for the shared core MLX model
   substrate
3. [ ] decide processor policy explicitly for the shared core MLX model
   substrate
4. [~] land standard two-stage on the dev checkpoint with required distilled
   LoRA
5. [~] land HQ as explicit opt-in
6. [x] strengthen `video.condition.audio`
7. [~] implement `video.condition.video`
8. [~] implement interpolation
9. [ ] validate and promote retake
9. [ ] replace the current best-available showcase pack once the stronger rows
   are real

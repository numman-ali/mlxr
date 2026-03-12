# LTX Capability Closure Plan

## Purpose

This is the concrete closure plan for the remaining official `LTX-2.3` family
surface in `MLXR`.

It exists to keep the work ordered, evidence-driven, and resistant to
capability drift. Use it together with:

- [11-ltx-capability-matrix.md](11-ltx-capability-matrix.md)
- [19-ltx-compatibility-checklist.md](19-ltx-compatibility-checklist.md)
- [18-ltx-reference-map.md](18-ltx-reference-map.md)

## Current truthful boundary

What is already real:

- promoted distilled text-to-video
- promoted distilled image-to-video
- promoted conditioned-audio row
- keyed-image bridge on the current image-conditioned row
- first meaningful `6s / 768x448 / 145f / 24fps` full-checkpoint `one_stage`
  receipt on the `dev` artifact path
- first meaningful `6s / 768x448 / 145f / 24fps` full-checkpoint receipts for
  both standard `two_stage` and `two_stage_hq`
- first meaningful `6s / 768x448 / 145f / 24fps` retake receipt on the fast
  distilled path
- first meaningful `6s / 768x448 / 145f / 24fps` union-control IC-LoRA receipt
  with a Gemini `match`
- first meaningful `6s / 768x448 / 145f / 24fps` motion-track receipt with a
  weaker Gemini `partial` verdict
- first meaningful `6s / 768x448 / 145f / 24fps` interpolation receipt with a
  Gemini `mismatch`

What is not closed yet:

- quality validation and promotion of one-stage on the full `dev` checkpoint
- quality validation and promotion of standard two-stage on the full `dev`
  checkpoint
- broader validation and promotion of reference-video conditioning via
  `ICLoraPipeline` beyond the current union-control slice
- stronger keyframe adherence for interpolation at the meaningful rung
- quality validation and promotion of retake semantics
- quality validation and promotion of HQ two-stage
- validation and promotion of keyframe interpolation
- explicit runtime/artifact support for the `LTX-2.3-fp8` checkpoint variant
- real validation of the extra IC-LoRA control assets such as motion-track

## Closure order

The next work should happen in this order:

1. validate and promote standard two-stage on the full `dev` checkpoint
2. validate and promote HQ as an explicit alternate row, not a silent default
3. keep one-stage honest as an educational row unless later receipts justify
   more
4. strengthen keyframe adherence and then validate and promote interpolation on
   top of the two-stage substrate
5. broaden `ICLoraPipeline` / `video.condition.video` beyond the current
   union-control win and decide whether motion-track earns independent
   promotion
6. strengthen retake semantics and then validate and promote retake
7. validate fp8 and extra control assets only after the base dev rows are
   already trustworthy
8. validate and promote the rows only after real receipts exist

## Concrete checklist

### A. Artifact and loading contract

- [x] `dev` one-stage no longer requires the x2 upsampler as a conversion
  prerequisite
- [x] `dev` artifacts can carry optional future-facing components such as the
  x2 upsampler and distilled LoRA without claiming those rows are already
  active
- [x] distilled LoRA is modeled explicitly as an internal component with a
  known filename contract
- [ ] real `dev` bundle conversion is validated against the official distilled
  LoRA asset
- [ ] `fp8` is recognized as a distinct checkpoint/artifact variant rather than
  being mis-read as the existing `dev` lane
- [ ] `fp8` loader and runtime behavior are validated deliberately instead of
  being assumed to behave like plain `bf16`

### B. Distilled LoRA substrate

- [x] implement owned MLX LoRA delta application for the LTX transformer
- [x] fail closed on unknown LoRA target names or shape mismatch
- [x] support the official `ltx-2.3-22b-distilled-lora-384.safetensors`
  contract first
- [x] keep user-facing generic LoRA refs unpromoted until the owned runtime
  path is real

### C. Two-stage family

- [x] add `two_stage` runtime dispatch on the `dev` checkpoint
- [x] use the full-checkpoint scheduler and CFG-guided stage 1
- [x] use x2 upsampling plus distilled LoRA refinement for stage 2
- [x] expose `two_stage` only when the artifact actually includes the required
  components
- [ ] validate safe rung and coherence rung before promotion

### D. HQ family

- [x] add `two_stage_hq` runtime dispatch
- [x] implement the `res_2s` second-order loop
- [x] support stage-specific distilled LoRA strengths
- [x] keep HQ opt-in rather than silently replacing standard two-stage
- [ ] validate at least one real HQ receipt before repo promotion

### E. Control and editing rows

- [x] keyframe interpolation: additive guiding-latent conditioning, not latent
  replacement
- [x] reference-video conditioning: `ICLoraPipeline` semantics with distilled
  checkpoint
- [~] conditioning attention strength is implemented as the current scalar
  first slice; richer mask semantics remain later work
- [x] retake: source-video encode, temporal retake mask, preserve outside
  window, optional audio retake

### F. Validation and promotion

- [ ] one real `one_stage` safe-rung receipt
- [ ] one real `two_stage` safe-rung receipt
- [ ] one real `two_stage` coherence-rung receipt
- [ ] one real `two_stage_hq` receipt
- [ ] one real interpolation receipt
- [ ] one real IC-LoRA receipt
- [ ] one real retake receipt
- [ ] docs and memory updated only after those receipts exist

## Decision rules

- Do not promote a row because the workflow or CLI can name it.
- Do not claim `dev` parity just because the checkpoint loads.
- Do not let host-adapter pressure redefine the core runtime task surface.
- Do not describe the keyed-image bridge as interpolation or retake.
- Do not talk about the full LTX family as complete until standard two-stage,
  HQ, interpolation, IC-LoRA, and retake are all real.

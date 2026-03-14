# Next Phase Task List

Status: active working checklist

## Purpose

This is the concrete follow-on checklist after the first best-available text-first showcase pack.

It exists so fresh sessions can keep moving without reconstructing the remaining roadmap from chat history.

Use this file together with:

- `docs/families/ltx-capability-matrix.md`
- `docs/research/13-ltx-showcase-execution-plan.md`
- `docs/workflow-orchestration-design.md`
- `docs/phased-delivery-plan.md`

## Current truth

Already true:

- the real non-preview `LTX-2.3` bridge is working
- the first best-available five-clip text-first showcase pack is promoted
- the natural-audio dog scene is now green on the stronger Gemini video-plus-audio review bar
- the promoted runtime path now passes caller-authored prompts through verbatim instead of composing prompt text in the workflow layer
- conditioned-audio validation is stronger than text-only audio steering
- the official upstream `LTX-2` pipeline canon and the official
  `ltx-desktop` product surface are both now important planning inputs

Not yet true:

- text-first natural-audio realism is not broadly solved
- `video.condition.audio` is now quality-promoted on the current dog and anime conditioned rows, but broader conditioned-scene coverage is still not solved
- `video.condition.video` as the official `ICLoraPipeline`
  reference-video row is now implemented, but still unvalidated
- interpolation, retake, one-stage, HQ, and LoRA rows are now implemented on
  the owned runtime, but still unvalidated or unpromoted
- the workflow layer is still planning, not full core-owned orchestration
- the desktop compatibility adapter does not exist yet
- the first image-family slot is now claimed by `Z-Image`, but broader
  editing-capable image-family coverage through `Qwen-Image` is still open
- owned `FLUX.2` `klein-4b` / `klein-9b` prompt-only generate and
  single-reference edit are now real, but multi-reference quality promotion,
  base rows, `dev`, LoRA loading, quantized execution, and prompt upsampling
  remain open

## Immediate next tasks

- [ ] Keep the current best-available pack truthful; do not replace it until a
  new pack clears the same review bar
- [ ] Do not claim that text-only no-music prompting is generally solved from
  one scene-level win alone
- [ ] Keep the capability docs aligned to the official `LTX-2` and
  `ltx-desktop` references whenever a row name could be ambiguous
- [ ] Decide whether the next primary implementation track is:
  - standard two-stage / HQ quality recovery
  - the first truthful desktop compatibility adapter slice
  - or the broader image-family tranche with `Qwen-Image` and `FLUX.2`

## LTX product-quality tasks

### A. Broaden natural-audio realism

- [ ] Revalidate the strongest natural-audio scenes on the current pass-through prompt contract
- [ ] Compare text-first and audio-conditioned results for the same scene and seed before blaming prompt wording alone
- [ ] Test whether the blocker is:
  - subject class
  - scene class
  - authored prompt wording
  - early denoise behavior
  - audio-conditioning absence
- [ ] Keep Gemini plus receipts as the acceptance gate for all reruns
- [ ] Promote a broader natural-audio pack only if multiple non-dog natural scenes clear the same bar

### B. Broaden `video.condition.audio`

- [x] Move `video.condition.audio` from “real bridge with preserved reference audio” to “quality-promoted capability”
- [x] Prove that the audio reference changes the scene result in a meaningful way, not just the muxed output
- [ ] Add comparison receipts:
  - text-first AV
  - audio-conditioned AV
  - same prompt, same seed, different reference audio
- [x] Add clearer product-facing docs for what this row currently guarantees and what it does not

### C. Implement the next capability rows

- [ ] validate and promote `video.condition.video` as the official
  `ICLoraPipeline` reference-video row
- [ ] validate and promote `video.interpolate`
- [ ] validate and promote `video.retake`
- [ ] validate and promote one-stage T2V
- [ ] validate and promote one-stage I2V
- [ ] validate and promote full two-stage / HQ T2V
- [ ] validate and promote full two-stage / HQ I2V
- [ ] validate and promote IC-LoRA
- [ ] distilled LoRA
- [ ] STG and prompt enhancement as second-ring controls

## Workflow and platform tasks

### A. Keep core, family, and host boundaries honest

- [ ] Keep the default user-facing generation contract simple:
  - one prompt
  - optional refs
  - actual inference parameters only
- [ ] Keep prompt composition out of the promoted runtime and first-party clients unless it earns a deliberate separate design later
- [ ] Keep host adapters thin; do not let desktop or CLI own inference logic

### B. Continue the workflow/orchestration roadmap

- [ ] Generalize the worker from the current fixed LTX-shaped sequence to adapter-declared stage graphs
- [ ] Promote the workflow layer from “planning” toward real core-owned orchestration only when the worker actually interprets stage graphs
- [ ] Keep docs aligned with that implementation truth at every step
- [ ] Add the next validating family after the LTX capability surface is broader, so the workflow layer stops being “LTX plus abstractions”

### C. Desktop adapter track

- [ ] Implement `packages/adapters/ltx-desktop/` as a thin compatibility layer
  over the shared runtime
- [ ] Preserve the current desktop backend contract for:
  - `/api/generate`
  - `/api/generate/cancel`
  - `/api/generation/progress`
  - local `video_path` completion responses
- [ ] Keep raw filesystem paths confined to the trusted desktop adapter seam;
  do not leak them back into the generic runtime HTTP contract
- [ ] Cover the first adapter slice with fast T2V and fast I2V before claiming
  broader desktop support
- [ ] Add explicit docs for which official desktop features are locally covered
  versus still API-only or unsupported

## Image-family track

- [x] Land `Z-Image` as the first explicit image family for `MLXR`
- [ ] Bring in an editing-capable image family so `MLXR` can generate related
  stills and edited keyframes without stretching LTX to solve that alone
- [ ] Treat `Qwen-Image-2512` plus `Qwen-Image-Edit-2511` as the primary
  editing-capable candidate unless a stronger Apple-Silicon fit appears
- [x] Treat `FLUX.2-klein-9b` as the primary FLUX row, with `4b` as the
  lighter sibling and `dev` as the high-end quality or editing row
- [x] Land owned `FLUX.2-klein-4b` / `klein-9b` `image.generate`
- [x] Land owned `FLUX.2-klein-4b` / `klein-9b` single-reference `image.edit`
- [ ] Fix and promote owned `FLUX.2` multi-reference `image.edit`
- [ ] Land owned `FLUX.2-klein-base-*` execution
- [ ] Land owned `FLUX.2-dev` execution
- [ ] Land owned `FLUX.2` LoRA loading and prompt-upsampling policy only when
  the shared/product seams are ready
- [ ] Keep the first image-family contract honest:
  - text-to-image first
  - runtime-managed `png` / `jpg` outputs
  - no editing or ControlNet-style claims until they are real
- [ ] Use the image-family tranche to support both:
  - Phase C cross-family validation
  - desktop image-to-video preparation workflows

## Typing and code-hygiene tasks

- [ ] Continue reducing broad production `object` usage where the schema or seam is already knowable
- [ ] Replace JSON-ish `dict[str, object]` with `TypedDict` where practical
- [ ] Replace donor-boundary `object` usage with narrow local protocols
- [ ] Keep `Any`, avoidable `cast`, and file-level mypy ignores out of runtime/family code
- [ ] Add repo-owned guidance for acceptable `object` boundaries versus under-modeled local code
- [ ] Keep the package split healthy; do not let new capability work collapse back into oversized backend modules

## Showcase and benchmarking tasks

### A. Showcase follow-through

- [ ] Keep the current best-available five-pack as the promoted baseline
- [ ] If broader natural-audio realism lands, build a second promoted pack rather than quietly rewriting the current one
- [ ] Keep first/mid/last stills, manifests, `ffprobe`, and Gemini review for every promoted scene

### B. Profile promotion

- [ ] Revalidate the best scenes at the coherence rung
- [ ] Promote `recommended` only after the corresponding capability rows are green
- [ ] Promote HQ only after recommended is green
- [ ] Promote longer clips only after the capability and profile rows are truthful

## Commit discipline for the next phases

- [ ] Commit from green states
- [ ] Use fresh-eyes review before commits that touch:
  - schemas
  - workflow planning
  - runtime-server
  - family execution code
  - validation doctrine
- [ ] Update docs in the same tranche when repo truth changes
- [ ] Keep temp receipts under `tmp/`, but encode durable conclusions into tracked docs and `MEMORY.md`

## Recommended execution order

1. Lock the official capability and host-contract docs to current upstream
   truth.
2. Choose one primary implementation track:
   `two_stage` / desktop adapter / `Z-Image`.
3. Complete that track at the smallest truthful slice.
4. Update the capability and host docs in the same tranche.
5. Commit from green.

## Current recommendation

The highest-leverage implementation decision is now broader than the earlier
"natural audio versus `video.condition.video`" fork.

If the goal is:

- highest-quality `LTX` parity, the next real target is standard two-stage on
  the `dev` checkpoint
- fastest host-surface value, the next real target is the first thin
  `ltx-desktop` adapter slice for fast T2V and I2V
- strongest platform validation, the next real target is the first image-family
  tranche, with `Z-Image` now the leading explicit candidate

# Next Phase Task List

## Purpose

This is the concrete follow-on checklist after the first best-available text-first showcase pack.

It exists so fresh sessions can keep moving without reconstructing the remaining roadmap from chat history.

Use this file together with:

- `docs/research/11-ltx-capability-matrix.md`
- `docs/research/13-ltx-showcase-execution-plan.md`
- `docs/workflow-orchestration-design.md`
- `docs/03-phased-delivery-plan.md`

## Current truth

Already true:

- the real non-preview `LTX-2.3` bridge is working
- the first best-available five-clip text-first showcase pack is promoted
- the natural-audio dog scene is now green on the stronger Gemini video-plus-audio review bar
- the promoted runtime path now passes caller-authored prompts through verbatim instead of composing prompt text in the workflow layer
- conditioned-audio validation is stronger than text-only audio steering

Not yet true:

- text-first natural-audio realism is not broadly solved
- `video.condition.audio` is now quality-promoted on the current dog and anime conditioned rows, but broader conditioned-scene coverage is still not solved
- `video.condition.video`, interpolation, retake, one-stage, HQ, and LoRA rows are still open
- the workflow layer is still planning, not full core-owned orchestration

## Immediate next tasks

- [ ] Decide whether the next product push is:
  - broader natural-audio realism on text-first AV
  - or the next capability row, `video.condition.video`
- [ ] Keep the current best-available pack truthful; do not replace it until a new pack clears the same review bar
- [ ] Do not claim that text-only no-music prompting is generally solved from one scene-level win alone

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

- [ ] `video.condition.video`
- [ ] `video.interpolate`
- [ ] `video.retake`
- [ ] one-stage T2V
- [ ] one-stage I2V
- [ ] full two-stage / HQ T2V
- [ ] full two-stage / HQ I2V
- [ ] IC-LoRA
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

1. Choose between broader natural-audio realism or `video.condition.video` as the next primary track.
2. Complete that track at safe rung with Gemini-backed review.
3. Update the capability matrix and showcase docs.
4. Commit from green.
5. Repeat for the next capability row.

## Current recommendation

The highest-leverage next step is:

- either broaden the natural-audio win beyond the dog scene
- or, if we want a cleaner capability step, implement the first truthful `video.condition.video` slice

If the goal is immediate product strength, broaden the natural-audio win first.

If the goal is capability surface coverage, start `video.condition.video` next.

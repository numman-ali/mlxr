# LTX Showcase Execution Plan

## Purpose

This is the current execution checklist for the next major `MLXR` tranche.

It exists so a fresh session can resume the work without reconstructing the plan from chat history.

The goal is:

- finish the remaining `LTX-2.3` audio and semantic-quality work
- complete the most important missing capability rows
- deliver the first truthful five-clip showcase pack

## Current state

Already true:

- real `LTX-2.3` text-to-video works
- real `LTX-2.3` image-to-video works
- real audio-bearing output works
- first real `video.condition.audio` slice works
- Gemini XML-based video review helper exists and is validated
- the repo has a cleaner package structure and split LTX internals

Still not true:

- text-first AV still drifts semantically on some clips
- the current `video.condition.audio` example proved the bridge, but not strong scene quality
- full upstream audio-fidelity parity is not complete yet
- the five-scene showcase pack is not ready to promote

## Execution checklist

### A. Audio quality and conditioning

- [ ] close the remaining BWE-related audio-fidelity gap on the checkpoint-backed path
- [ ] compare current MLXR audio output against official `LTX-2` vocoder/BWE expectations
- [ ] confirm whether the current weakness is mostly audio fidelity, scene semantics, or both
- [ ] improve `video.condition.audio` scene quality so the output actually matches the intended prompt/reference pairing
- [ ] keep the passthrough/reference-audio truth explicit until full parity is proven

### B. Text-first prompt and workflow quality

- [ ] use the new prompting guide as the source of truth for prompt shaping
- [ ] keep the default UX text-first
- [ ] improve the planner’s internal prompt shaping for:
  - `natural_audio`
  - `no_music`
  - duration
  - orientation
- [ ] keep all of those truthful as text-shaping aids, not hard guarantees
- [ ] validate that prompt shaping helps without hiding model limitations

### C. Semantic validation workflow

- [ ] keep `scripts/gemini_describe_video.py` as the default automated semantic cross-check
- [ ] use ffprobe and runtime manifests as execution truth
- [ ] inspect stills/keyframes whenever Gemini and runtime expectations disagree
- [ ] do not promote clips that Gemini identifies as the wrong scene or wrong audio character

### D. Core capability completion

- [ ] strengthen `video.condition.audio` from “real bridge slice” to “quality-promoted capability”
- [ ] implement `video.condition.video`
- [ ] implement `video.interpolate`
- [ ] implement `video.retake`
- [ ] implement one-stage T2V/I2V
- [ ] implement full two-stage / HQ variants
- [ ] implement IC-LoRA / distilled LoRA paths

### E. Showcase pack

- [ ] define five distinct text-first scenes
- [ ] keep each scene at 10 seconds
- [ ] require:
  - real runtime receipts
  - first/mid/last stills
  - ffprobe output
  - Gemini parsed review
- [ ] reject any clip that is semantically off-target or drifts into the wrong audio mode
- [ ] only promote the pack once all five clips clear the same truth bar

Suggested initial scene set:

- [ ] cinematic natural dog with owner in a park
- [ ] anime action scene
- [ ] vintage / old-school scene
- [ ] nature / documentary scene
- [ ] handcrafted / stop-motion-like scene

### F. After the showcase

- [ ] resume the remaining LTX capability matrix rows
- [ ] continue worker/stage-graph generalization so workflow planning can grow into true orchestration
- [ ] resume the broader multi-family platform validation path from the phased delivery plan

## Working rules

- commit from green states
- use fresh-eyes review on substantial diffs
- keep docs moving with repo truth
- do not claim semantic success from pipeline completion alone
- keep the default user story simple even when the internal workflow is more sophisticated

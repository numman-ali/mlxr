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
- the Gemini helper now stages an extracted review WAV when audio is present and reviews the video and audio together
- a repo-owned showcase runner now exists at `scripts/ltx_showcase_generate.py`
- the repo has a cleaner package structure and split LTX internals

Still not true:

- text-first AV still drifts semantically on some clips
- text-first natural scenes still drift into soundtrack-like music on the current 10-second dog, nature, and vintage receipts
- the current `video.condition.audio` example proved the bridge, but not strong scene quality
- conditioned-scene quality on `video.condition.audio` is still not strong enough to promote
- the five-scene showcase pack is not ready to promote

Already promoted at the current score-friendly bar:

- anime neon chase
- stop-motion workshop
- vintage nostalgic street
- cinematic spacewalk
- noir rainy city

Current blocker for the original natural-audio showcase intent:

- the 10-second dog, nature, and vintage natural-scene receipts still come back as visual matches with music-like audio on the stronger video-plus-audio Gemini review path
- that means the current text-first `natural_audio` / `no_music` story is still not strong enough to promote as a natural-sound showcase pack

## Execution checklist

### A. Audio quality and conditioning

- [x] close the remaining BWE-related audio-fidelity gap on the checkpoint-backed path
- [x] compare current MLXR audio output against official `LTX-2` vocoder/BWE expectations
- [x] confirm that the current remaining weakness is now mostly scene semantics on the conditioned path, not raw audio-fidelity plumbing
- [ ] improve `video.condition.audio` scene quality so the output actually matches the intended prompt/reference pairing
- [ ] keep the passthrough/reference-audio truth explicit until full parity is proven

### B. Text-first prompt and workflow quality

- [ ] use the new prompting guide as the source of truth for prompt shaping
- [x] add a repo-owned text-first prompt shaper shared by the LTX workflow layer and the showcase runner
- [ ] keep the default UX text-first
- [ ] improve the planner’s internal prompt shaping for:
  - `natural_audio`
  - `no_music`
  - duration
  - orientation
- [ ] keep all of those truthful as text-shaping aids, not hard guarantees
- [ ] validate that prompt shaping helps without hiding model limitations

### C. Semantic validation workflow

- [x] keep `scripts/gemini_describe_video.py` as the default automated semantic cross-check
- [x] stage an extracted review WAV alongside showcase MP4s when audio is present so Gemini reviews both in one pass
- [x] make `scripts/ltx_showcase_generate.py` save `ffprobe` receipts and fail promotion when Gemini does not clear the expected subject/audio gate
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
- [x] promote the first score-friendly five-clip pack once all five clips clear the current truth bar

Current dog-showcase truth:

- the first 10-second dog showcase receipt at `384x224 / 241f / 24fps` now produces the intended dog-with-owner park scene visually
- Gemini still classifies its audio as `music`, so that clip is not promotable for the natural-audio/no-music scene

Current natural-audio showcase truth:

- the stronger video-plus-audio Gemini pass now also classifies the 10-second nature and vintage clips as visual matches with music-like audio
- the blocker is no longer “weak review tooling”; it is the current text-first AV behavior on these natural scenes

Current score-friendly showcase truth:

- a distinct five-clip text-first pack now exists at `384x224 / 241f / 24fps`
- the consolidated receipt is `tmp/showcase-runs/showcase-pack-20260308.json`
- this pack is truthful for stylized and score-friendly scenes; it is not evidence that natural-audio realism is solved

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

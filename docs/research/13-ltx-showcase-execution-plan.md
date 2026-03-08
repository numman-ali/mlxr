# LTX Showcase Execution Plan

## Purpose

This is the current execution checklist for the next major `MLXR` tranche.

It exists so a fresh session can resume the work without reconstructing the plan from chat history.

For the full upstream surface and owned-substrate sequencing, read this
alongside:

- [18-ltx-reference-map.md](18-ltx-reference-map.md)
- [19-ltx-compatibility-checklist.md](19-ltx-compatibility-checklist.md)

The goal is:

- finish the remaining `LTX-2.3` audio and semantic-quality work
- complete the most important missing capability rows
- deliver the first truthful five-clip showcase pack

The next quality target after the current distilled showcase tranche is not
simply “bigger or longer.” It is the upstream full-checkpoint two-stage family.
But the current engine still needs the owned-substrate migration described in
[16-owned-substrate-migration-plan.md](16-owned-substrate-migration-plan.md)
before those non-distilled rows should be promoted.

- `TI2VidTwoStagesPipeline` as the production-default quality path
- `TI2VidTwoStagesHQPipeline` as the next top-end alternate high-quality
  variant

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
- nature and vintage natural scenes still need to be re-run on the newer guidance path
- the current `video.condition.audio` example proved the bridge, but not strong scene quality
- conditioned-scene quality on `video.condition.audio` is still not strong enough to promote
- the five-scene natural-audio showcase pack is not ready to promote

Already promoted at the current score-friendly bar:

- anime neon chase
- stop-motion workshop
- vintage nostalgic street
- cinematic spacewalk
- noir rainy city

Current blocker for the original natural-audio showcase intent:

- the 10-second dog scene now clears the stronger video-plus-audio Gemini review path on the newer negative-guidance/rescaled-CFG path
- nature and vintage natural scenes have not yet been revalidated on that newer path, so the current text-first `natural_audio` / `no_music` story is still not broad enough to promote as a natural-sound showcase pack

## Execution checklist

### A. Audio quality and conditioning

- [x] close the remaining BWE-related audio-fidelity gap on the checkpoint-backed path
- [x] compare current MLXR audio output against official `LTX-2` vocoder/BWE expectations
- [x] confirm that the current remaining weakness is now mostly scene semantics on the conditioned path, not raw audio-fidelity plumbing
- [ ] improve `video.condition.audio` scene quality so the output actually matches the intended prompt/reference pairing
- [ ] keep the passthrough/reference-audio truth explicit until full parity is proven

### B. Text-first prompt and workflow quality

- [x] use the new prompting guide as the source of truth for prompt shaping
- [x] add a repo-owned text-first prompt shaper shared by the LTX workflow layer and the showcase runner
- [ ] keep the default UX text-first
- [x] improve the planner’s internal prompt shaping for:
  - `natural_audio`
  - `no_music`
  - duration
  - orientation
- [x] keep all of those truthful as text-shaping aids, not hard guarantees
- [ ] validate that prompt shaping helps without hiding model limitations
- [x] add the first family-local negative-guidance slice and rescaled CFG recovery path for text-first natural-audio scenes

### C. Semantic validation workflow

- [x] keep `scripts/gemini_describe_video.py` as the default automated semantic cross-check
- [x] stage an extracted review WAV alongside showcase MP4s when audio is present so Gemini reviews both in one pass
- [x] make `scripts/ltx_showcase_generate.py` save `ffprobe` receipts and fail promotion when Gemini does not clear the expected subject/audio gate
- [ ] use ffprobe and runtime manifests as execution truth
- [ ] inspect stills/keyframes whenever Gemini and runtime expectations disagree
- [ ] do not promote clips that Gemini identifies as the wrong scene or wrong audio character

### D. Core capability completion

- [ ] remove runtime `mlx-video` usage from the promoted distilled engine
- [ ] remove prompt-path `mlx-vlm` / `mlx-lm` usage from the Gemma prompt stack
- [ ] extend the artifact and adapter contract for non-distilled two-stage:
  full `dev` checkpoint plus distilled LoRA on top of Gemma and x2 upsampler
- [ ] implement the non-distilled scheduler/sampler/guidance substrate needed by upstream two-stage rows on the owned engine
- [ ] land the upstream production-default `TI2VidTwoStagesPipeline` semantics on the full `dev` checkpoint
- [ ] land the upstream `TI2VidTwoStagesHQPipeline` semantics as the next
  top-end alternate high-quality variant
- [ ] strengthen `video.condition.audio` from “real bridge slice” to “quality-promoted capability”
- [ ] implement `video.condition.video`
- [ ] implement `ICLoraPipeline` semantics for strong-control image/video conditioning
- [ ] implement `video.interpolate`
- [ ] implement `video.retake`
- [ ] implement one-stage T2V/I2V
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

- the early 10-second dog receipt at `tmp/showcase-runs/20260308T022707Z-showcase-dog-park-natural/` remained blocked on music-like audio
- the improved 10-second dog receipt at `tmp/showcase-runs/20260308T034004Z-showcase-dog-park-natural/` now clears the stronger Gemini video-plus-audio review path
- Gemini identifies a golden retriever with its owner in a park and classifies the audio as barking plus natural outdoor ambience
- this is the first scene-level success for the natural-audio/no-music story, not yet broad proof for every natural scene

Current natural-audio showcase truth:

- the newer negative-guidance/rescaled-CFG path recovers the 10-second dog scene
- the rerun `nature_documentary` receipt at `tmp/showcase-runs/20260308T035528Z-showcase-nature-documentary/` still comes back as a visual match with piano-like music
- the rerun `vintage_old_school` receipt at `tmp/showcase-runs/20260308T035838Z-showcase-vintage-old-school/` still comes back as a visual match with nostalgic instrumental music
- the blocker is no longer “weak review tooling”; it is broadening the newer text-first natural-audio behavior beyond the dog scene

Current score-friendly showcase truth:

- a distinct five-clip text-first pack now exists at `384x224 / 241f / 24fps`
- the consolidated receipt is `tmp/showcase-runs/showcase-pack-20260308.json`
- this pack is truthful for stylized and score-friendly scenes; it is not evidence that natural-audio realism is solved

Current best-available promoted pack:

- the current strongest five-clip pack is `tmp/showcase-runs/showcase-pack-20260308-best-available.json`
- it keeps four previously promoted stylized scenes and replaces the earlier blocked dog row with the newer natural-audio dog success under `tmp/showcase-runs/20260308T034004Z-showcase-dog-park-natural/`
- this is the best truthful pack today, but it still does not promote the broader natural-scene story beyond that dog scene

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

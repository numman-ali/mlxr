# LTX-2.3 Prompting Guide

## Purpose

This note captures the current prompting best practices for `LTX-2.3` in `MLXR`.

It has two jobs:

- record the official upstream guidance we should follow for text-first prompting
- define how `MLXR` should shape one simple user prompt into a stronger text-first generation request without inventing a more complex default UX

This guide is specifically for the current first-party `LTX-2.3` proving path. It does not claim to be a universal prompting guide for all families.

## Upstream guidance

The official `LTX-2` guidance is consistent on a few important points:

- prompts should ideally be one flowing paragraph
- prompts should start with the action
- descriptions should be literal, specific, and chronological
- prompts should cover:
  - subject appearance
  - motion and gesture
  - environment and background
  - camera framing and movement
  - lighting and color
  - important scene changes
- prompts should stay under roughly 200 words

For text-first generation, the best mental model is:

describe the shot the way a cinematographer would describe the shot list, not the way a user would write a vague vibe request.

## Audio guidance

Official LTX supports both text-first AV generation and a stronger audio-driven path.

That means:

- text-first AV is a valid simple UX path
- but precise sound behavior is better controlled through audio-conditioned generation

So for `MLXR`, the product posture should be:

- default UX stays text-first
- audio-specific intent is still worth writing explicitly into the prompt
- when exact sound matters, `video.condition.audio` is the stronger control path

Do not pretend prompt wording alone is always enough to force exact natural sound behavior.

## MLXR text-first prompt policy

The simple user-facing contract remains:

- one main prompt
- optional references
- simple preferences such as `natural_audio`, `no_music`, `enhance_prompt`, duration, and orientation

Internally, `MLXR` may split that into:

- base visual prompt
- explicit audio details
- advisory framing and duration cues

The current repo-owned implementation of that shaping lives in:

- `packages/families/ltx/src/mlxr/families/ltx/prompting.py`

and is shared by:

- the LTX workflow strategy
- the showcase runner

The planner should not require users to think in terms of separate audio and video prompts by default, but it may derive those internally.

The current workflow layer also warns explicitly when `natural_audio` or `no_music` are being treated as text-first guidance rather than a guaranteed audio-control path.

Current implementation caveat:

- the repo-owned shaper currently emits structured multi-line guidance blocks such as `Audio details:` and `Audio prohibition:`
- that is a deliberate prompt-shaping experiment for the current proving path, not a claim that this exact formatting is upstream-canonical

## Prompt shaping rules in MLXR

For the current LTX workflow strategy:

- the base `prompt` remains the main source of truth
- `video_prompt` and `audio_prompt` are optional advanced details
- `natural_audio` becomes an explicit “natural diegetic sound only” instruction
- `no_music` becomes an explicit “no soundtrack / no background music” instruction
- `duration_seconds` becomes an advisory duration line
- `orientation` becomes an advisory framing line when the value is one of:
  - `portrait`
  - `landscape`
  - `square`

These are text-shaping improvements, not hard guarantees.

## Good prompt pattern

Preferred shape:

1. main action
2. subject details
3. environment
4. camera
5. lighting and color
6. audio details when relevant

Example:

`A golden retriever runs happily beside its owner through a sunlit park path, looking up toward them as they move together past green grass and trees, filmed in a smooth handheld tracking shot at waist height with warm natural afternoon light. Natural park ambience, light footsteps, and happy barking only, with no music.`

## Bad prompt pattern

Avoid prompts like:

- “beautiful cinematic dog video”
- “epic anime vibes”
- “make this cool and dramatic”

Those are not concrete enough to drive the current model well.

## Prompt enhancement

Prompt enhancement is a valid future workflow stage and is already part of the official product story.

When `MLXR` eventually enables it on the real runtime path, it should:

- add visual detail
- add motion cues
- add sound cues

But until that is implemented, the planner and docs should treat it as unavailable rather than pretending it is active.

## Validation rule

Do not promote a prompt pattern as “working” from one lucky clip.

A prompt/process improvement should count only after:

- the runtime receipts are real and non-preview
- Gemini review and stills agree on the scene
- the clip clears the intended semantic bar at the current rung

Current caveat:

- stronger text shaping improves scene description quality
- `no_music` is still not a blanket guarantee for text-first natural-audio scenes
- the first repo-owned scene-level recovery path that improved this was family-local negative guidance plus rescaled CFG on the distilled AV bridge
- that path is now strong enough for the 10-second dog receipt at `tmp/showcase-runs/20260308T034004Z-showcase-dog-park-natural/`, where Gemini reviews the clip as a golden retriever with barking and natural outdoor ambience
- the current showcase runner now feeds each scenario's `audio_intent` into the shared prompt shaper and adds style-aware anti-music negatives for `naturalistic`, `documentary`, and `vintage` scenes
- that newer runner-only slice clearly rescues `heron_marsh_documentary` at the 10-second `384x224 / 241f` no-music review bar
- separate current repo receipts also show no-music wins for `basketball_court_dusk` and `cafe_sidewalk_human`, but those are not clean proof that this specific shaping change caused the improvement
- it still does not generalize to every natural scene; `cat_kitchen_natural` and the rerun `vintage_laundromat` still come back with music-like audio
- score-friendly text-first scenes remain the safest promoted showcase set until more natural scenes clear the same bar

## Sources

- `references/official/LTX-2/README.md`
- `references/official/LTX-2/packages/ltx-pipelines/README.md`
- `references/official/ltx-desktop/frontend/components/SettingsModal.tsx`
- `references/official/ltx-desktop/frontend/views/GenSpace.tsx`

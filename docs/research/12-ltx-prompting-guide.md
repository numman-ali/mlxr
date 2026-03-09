# LTX-2.3 Prompting Guide

## Purpose

This note captures the current prompting contract for `LTX-2.3` in `MLXR`.

It has two jobs:

- record the official upstream guidance we should follow for text-first prompts
- state the current `MLXR` product truth about what happens to a prompt before inference

This guide is specific to the current first-party `LTX-2.3` proving path. It is
not a universal prompting guide for every family.

## Upstream guidance

The official `LTX-2` guidance is consistent on a few important points:

- prompts work best as one flowing paragraph
- start with the action
- be literal, specific, and chronological
- cover:
  - subject appearance
  - motion and gesture
  - environment and background
  - camera framing and movement
  - lighting and color
  - important scene changes
- stay under roughly 200 words

Good mental model:

describe the shot the way a cinematographer would describe a shot list, not the
way a user would write a vague vibe request.

## Current MLXR prompt contract

The current promoted `MLXR` runtime path now passes the caller-authored prompt
through verbatim.

That means:

- no repo-owned prompt composition in the LTX workflow layer
- no synthesized `Audio details:` or `Audio prohibition:` lines
- no synthesized negative prompt from `natural_audio`, `no_music`, duration, or
  orientation hints
- no first-party CLI prompt-authoring flags for those behaviors

Current first-party contract:

- one prompt
- optional conditioning references such as image or audio inputs
- generation parameters such as width, height, frames, fps, seed, output format,
  and quality

If a user wants extra wording in the prompt, they must author that wording
explicitly. The runtime does not invent or rewrite it.

## Audio guidance

Official LTX supports both text-first AV generation and a stronger
audio-conditioned path.

So the current `MLXR` posture is:

- text-first AV remains a valid simple path
- exact sound behavior is not guaranteed from prompt wording alone
- when exact sound matters, `video.condition.audio` is the stronger control path

Do not treat prompt wording as a substitute for real audio conditioning.

## Good prompt pattern

Preferred shape:

1. main action
2. subject details
3. environment
4. camera
5. lighting and color
6. any audio detail you want the model to see, written directly in the same
   prompt

Example:

`A golden retriever runs happily beside its owner through a sunlit park path, looking up toward them as they move together past green grass and trees, filmed in a smooth handheld tracking shot at waist height with warm natural afternoon light. Natural park ambience, light footsteps, and happy barking only, with no music.`

## Bad prompt pattern

Avoid prompts like:

- “beautiful cinematic dog video”
- “epic anime vibes”
- “make this cool and dramatic”

Those are not concrete enough to drive the current model well.

## Prompt enhancement

Prompt enhancement remains a possible future workflow feature, but it is not
part of the promoted runtime contract today.

Until it exists as a real validated stage, the docs and product surface should
treat it as unavailable.

## Validation rule

Do not promote a prompt pattern as “working” from one lucky clip.

A prompt/process claim counts only after:

- the runtime receipts are real and non-preview
- Gemini review and stills agree on the scene
- the clip clears the intended semantic bar at the current rung

Current caveat:

- the repo intentionally removed runtime-owned prompt shaping on `2026-03-09`
  because it obscured the exact prompt being sent to the model and did not
  belong in the promoted inference path
- older receipts generated under repo-owned prompt shaping should be treated as
  historical evidence, not as the current runtime contract
- text-first no-music or natural-audio wins are still not broad enough to
  promote as a general guarantee
- conditioned audio remains the stronger control row when sound behavior matters

## Sources

- `references/official/LTX-2/README.md`
- `references/official/LTX-2/packages/ltx-pipelines/README.md`
- `references/official/LTX-Video/README.md`

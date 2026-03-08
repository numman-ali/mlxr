# LTX Fidelity Debug Playbook

## Purpose

This playbook captures the working method for real `LTX-2.3` fidelity debugging in `MLXR`.

Use it when the real path runs but the output is still visually wrong, ambiguous, or too weak to promote into repo truth.

This is not the general dev loop. It is the narrow workflow for:

- real-weight local smoke runs
- stage-local visual debugging
- safe iteration on a large Apple-Silicon workload
- deciding whether the next suspect is prompt/stage 1, x2/stage 2, or decode/export

This runner is transitional tooling, not the long-term product surface.

The durable goal is for `mlxr` itself to absorb the important debug controls:

- trace capture
- safety guardrails
- backend verification
- debug artifact export

Use `ltx_debug_smoke.py` only when you need engine-local diagnostics that the CLI does
not expose yet.

## Canonical Method

The method is:

1. start from the smallest truthful rung
2. use real converted artifacts for visual conclusions
3. change one major variable at a time
4. trust stage-local evidence more than the final clip when localizing a bug
5. fail closed on contract drift instead of reshaping or defaulting through it
6. do not promote an ambiguous result

The debugging ladder is:

1. confirm the path is real, not preview
2. confirm subject formation at a safe rung
3. localize the first broken stage
4. fix the earliest broken stage
5. rerun the same rung with the same seed
6. only then scale up

## Safe Execution Rules

Real local `LTX-2.3` AV smokes can stress a `64 GB` Mac hard enough to flirt with watchdog resets.

Use clean lifecycle by default:

1. create prompt encoder
2. encode prompt
3. close prompt encoder
4. clear MLX cache
5. create video generator
6. generate

For long real-weight smokes, use detached execution. The recommended pattern is `tmux`, not a fragile foreground terminal session.

Canonical runner:

```bash
uv run python scripts/ltx_debug_smoke.py \
  --artifact-root /absolute/path/to/artifact/payload \
  --prompt "a golden retriever dog playing in a grassy park, cinematic, natural light" \
  --profile safe-smoke \
  --seed 165783600 \
  --run-name dog-safe-rung
```

Detached helper:

```bash
uv run python scripts/ltx_debug_smoke.py \
  --artifact-root /absolute/path/to/artifact/payload \
  --prompt "a golden retriever dog playing in a grassy park, cinematic, natural light" \
  --run-name dog-safe-rung \
  --print-tmux-command
```

If a smoke is interrupted, check for orphaned long-running `uv run python` / smoke-script processes before launching another.

The smoke runner now has built-in hard safety guardrails:

- MLX memory limit via `mx.set_memory_limit()`
- watchdog aborts for active-memory or peak-memory overruns
- watchdog abort for wall-clock overruns
- abort receipts written back into `run_manifest.json`

Do not disable these casually on shared developer machines.

## Resolution Ladder

The acceptance ladder is strict:

1. `256x160 / 17f / 24fps`
   This is the first meaningful semantic gate.
2. `384x224 / 17f / 24fps`
   This is the first clarity-improvement rung.
3. `768x512 / 33f / 24fps`
   This is the first coherence rung.
4. `1536x1024 / 121f / 24fps`
   Only after the smaller rungs are visually correct.
5. `1920x1088`
   Only after recommended-profile success.

The canonical profile names in `scripts/ltx_debug_smoke.py` are:

- `safe-smoke`
- `visual-gate`
- `coherence`
- `recommended`
- `hq-first-pass`

Do not use `128x96` as a semantic success gate. It is too small to distinguish “subject formation” from “actual dog.”

## Output Bundle

The repo-owned smoke script writes a predictable run bundle under `tmp/manual-runs/<timestamp>-<slug>/`:

- final MP4
- reviewer stills: `frame_0001.png`, `frame_mid.png`, `frame_last.png`
- `debug/` stage snapshots when stage debug is enabled
- `run_manifest.json` with prompt, seed, shape, artifact source, lifecycle mode, and timings

Treat the manifest as the run receipt. If a result later becomes part of repo truth, the manifest should explain what actually ran.

## Gemini Review Pass

Use `scripts/gemini_describe_video.py` as the default automated semantic cross-check for promoted clips.

The helper stages the video into a non-ignored review directory, extracts a mono WAV review track when the clip has audio, asks Gemini for one strict XML review over both assets, and saves:

- the staged review copy
- the staged review audio track when present
- the exact review prompt
- the raw Gemini response
- the parsed review JSON

Recommended pattern:

```bash
uv run python scripts/gemini_describe_video.py \
  /absolute/path/to/clip.mp4 \
  --save-dir /tmp/mlxr-gemini-review/dog-clip \
  --strict
```

Promotion rule:

- do not promote a clip if Gemini returns a semantically wrong scene or subject
- do not promote a clip if Gemini reports soundtrack-like music when the prompt and manifest expect natural audio with no music
- if Gemini and the runtime receipt disagree, inspect reviewer stills, ffprobe output, and keyframes before promoting anything
- Gemini does not override runtime receipts, ffprobe output, or obvious manual-review contradictions

Current LTX showcase truth:

- the audio-aware Gemini pass confirms that the current text-first natural scenes are still drifting into soundtrack-like music
- this is now confirmed on the 10-second dog, nature, and vintage receipts, not just inferred from one weak review pass
- until that changes, text-first `natural_audio` / `no_music` must remain advisory rather than promotable guarantees

## How To Interpret Stage Snapshots

The default stage-debug ladder is:

- `stage1`
- `post_x2`
- `final`

Use it as a decision ladder:

- if `stage1` is already abstract or semantically wrong:
  the next suspect is prompt encoding, prompt-to-transformer handoff, or stage-1 denoise
- if `stage1` is coherent but `post_x2` collapses:
  the next suspect is VAE stats, x2 remap/layout, or stage boundary logic
- if `post_x2` is coherent but `final` collapses:
  the next suspect is stage 2 or final decode/export

Prefer stage-local localization over guessing from the final MP4 alone.

## Claim Discipline

Use these labels honestly:

- subject formation
- animal-like subject
- clear dog
- coherent mid-resolution dog scene
- recommended-profile success
- HQ success

Important rule:

- generic quadruped != dog
- ambiguous cartoon animal != dog
- “looks like something” != milestone cleared

The first real dog milestone is only cleared when an independent human would call it a dog without prompting.

## Docs To Load Alongside This Playbook

- `docs/research/07-ltx-integration-seams.md`
- `docs/research/09-open-questions-and-validation-plan.md`
- `MEMORY.md`
- `.agents/skills/ltx-fidelity-debugging/SKILL.md`

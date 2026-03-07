---
name: ltx-fidelity-debugging
description: Use this skill when debugging real LTX-2.3 output quality, running safe manual smoke tests, interpreting stage snapshots, or deciding whether a result is specific enough to count as a clear dog or other visually coherent subject.
---

# LTX Fidelity Debugging

Use this skill for the narrow slice where the real `LTX-2.3` path runs but the output is still visually wrong, ambiguous, or not strong enough to promote into repo truth.

This skill is for:

- real-weight local smoke runs
- stage-local visual debugging
- safe iteration on the `22B` AV path
- deciding whether the next suspect is prompt/stage 1, x2/stage 2, or decode/export

Do not use this skill for the common implementation path. The default repo loop still lives in `AGENTS.md`, `MEMORY.md`, and `scripts/dev.py`.

## Default workflow

1. Read `docs/research/10-ltx-fidelity-debug-playbook.md`.
2. Use `scripts/ltx_debug_smoke.py` with a fixed seed and real converted artifacts.
3. Start with the named profile ladder: `safe-smoke`, then `visual-gate`, then `coherence`.
4. Keep clean lifecycle on unless you are disproving the lifecycle itself.
5. Use detached `tmux` execution for long real-weight smokes.
6. Inspect `stage1`, `post_x2`, and `final` snapshots before deciding what to fix next.
7. Review `frame_0001`, `frame_mid`, and `frame_last` before promoting a rung.
8. Downgrade the claim if the result is ambiguous.

## What to load

- `docs/research/10-ltx-fidelity-debug-playbook.md`
- `docs/research/07-ltx-integration-seams.md`
- `docs/research/09-open-questions-and-validation-plan.md`
- `MEMORY.md`

Load these code paths only when you need the implementation seam:

- `packages/families/ltx/src/ltx/_prompt_encoding_backend.py`
- `packages/families/ltx/src/ltx/_generation_backend.py`
- `scripts/ltx_debug_smoke.py`

## Rules

- Start at the smallest truthful rung, not the highest resolution.
- Use real converted artifacts for visual conclusions.
- Change one major variable at a time.
- Trust stage-local evidence over the final clip when localizing a bug.
- Generic quadruped output does not count as dog success.

---
name: mlxr-runtime-validation
description: Use this skill when working on MLXR runtime debugging, log triage, validation discipline, benchmark-backed claims, or deciding whether runtime behavior is strong enough to count as evidence rather than a one-off run.
---

# MLXR Runtime Validation

Use this skill for the narrow slice that sits beyond the normal dev loop:

- debugging runtime behavior
- inspecting logs after source, artifact, or daemon changes
- validating whether a claim is benchmark-grade or still provisional
- triaging whether a behavior is a platform truth, a family quirk, or a likely bug

Do not use this skill for the common implementation path. The default loop still lives in `AGENTS.md` and `docs/dev-harness.md`.

## Default workflow

1. Run the repo harness first:
   - `uv run python scripts/dev.py verify`
   - `uv run python scripts/dev.py logs --tail 120`
2. Inspect the relevant platform docs before interpreting results.
3. Distinguish between:
   - implementation bug
   - missing benchmark evidence
   - stale docs
   - still-open architecture question
4. If the result is still uncertain, downgrade the claim instead of over-stating it.

## What to load

- For benchmark strength and freeze criteria:
  - `docs/benchmark-matrix.md`
  - `docs/research/09-open-questions-and-validation-plan.md`
- For runtime and scheduler interpretation:
  - `docs/02-universal-mlx-runtime-design.md`
  - `docs/research/05-optimization-playbook.md`
- For LTX-specific runtime pressure:
  - `docs/research/07-ltx-integration-seams.md`

## Validation rules

- Prefer measured local evidence over inferred explanations.
- Separate model behavior, output-path behavior, and harness behavior.
- Treat logs as evidence, not decoration.
- Do not promote a result to a stable claim if it still depends on one machine, one family, or one narrow path.

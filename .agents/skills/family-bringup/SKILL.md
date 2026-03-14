---
name: family-bringup
description: Use this skill when bringing up a new model family, re-basing an existing family onto the current platform contract, or deciding what belongs in core workflow orchestration versus family or host code.
---

# Family Bring-Up

Use this skill for the narrow slice where `MLXR` is onboarding a new family or reshaping a family boundary in a way that affects workflow ownership, capability claims, or the truthful first slice.

This skill is for:

- defining the smallest truthful family slice
- separating core workflow, family adapter, and host adapter responsibilities
- documenting source, artifact, workflow, and capability contracts before code drifts
- building the validation ladder and fresh-eyes review gate for the family

Do not use this skill for routine bug fixes or the common implementation path. The default repo loop still lives in `AGENTS.md`, `MEMORY.md`, and `docs/agent-native-development.md`.

## Default workflow

1. Read `docs/workflow-orchestration-design.md`.
2. Read `docs/family-bringup/README.md`, `docs/family-bringup/checklist.md`, and `docs/family-bringup/template.md`.
3. When the family has real-weight runtime bugs or "wrong output" failures, read `docs/family-bringup/debugging-decision-tree.md`.
4. Read `docs/technical-design.md`, `docs/phased-delivery-plan.md`, and `docs/research/09-open-questions-and-validation-plan.md`.
5. Inspect the relevant current family and host code before claiming a clean boundary.
6. Name the smallest truthful slice before optional features.
7. Keep reusable sequencing in the core workflow layer, family truth in the family adapter, and UX glue in the host.
8. Record fail-closed boundaries and the validation ladder before promoting support.
9. Do the fresh-eyes review pass before commit.

## What to load

- `docs/workflow-orchestration-design.md`
- `docs/family-bringup/README.md`
- `docs/family-bringup/checklist.md`
- `docs/family-bringup/template.md`
- `docs/family-bringup/debugging-decision-tree.md` when isolating runtime-quality or runtime-contract failures
- `docs/product-requirements.md`
- `docs/technical-design.md`
- `docs/phased-delivery-plan.md`
- `docs/research/09-open-questions-and-validation-plan.md`

Load the relevant family research doc and adapter code only when needed for the specific family.

For `LTX-2.3` family pressure, also load:

- `docs/research/07-ltx-integration-seams.md`

## Rules

- Hosts do not own inference logic or reusable workflow sequencing.
- Families do not become mini-schedulers.
- Core does not pretend one family-specific detail is universal before a second family validates it.
- Start with the smallest truthful slice, not the longest feature wish list.
- One successful smoke is evidence, not stable scope.

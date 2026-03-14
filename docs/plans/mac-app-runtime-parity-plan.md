# MLXR Mac App Runtime-Parity Plan

Last updated: 2026-03-14

## Summary

The Mac app should be a thin, beautiful studio over the same runtime contract
used by the CLI and future adapters.

The core rule is simple:

- the runtime owns capability truth
- the runtime owns planning truth
- the runtime owns run acceptance truth
- the app owns presentation, organization, and UX language

No host should invent a second workflow-validity matrix or a second model
default system.

## Product Shape

Top-level surfaces stay fixed:

- `Home`
- `Studio`
- `Library`
- `Models`
- `Settings`

`Jobs` is not a primary destination. Execution state is a secondary utility
through `Activity` and in-context Studio progress.

## Runtime Parity Rules

The app, CLI, and future adapters must all consume the same runtime seams:

- `/v1/capabilities`
- `/v1/models`
- `/v1/models/supported`
- `/v1/workflows/plan`
- `/v1/workflows/run`
- `/v1/jobs`
- input import and output routes

The shared planning seam must be the readiness source for draft validation.

`WorkflowPlanResult.readiness` should tell hosts:

- whether the draft is runnable
- what is blocking
- which references are required
- which output formats are allowed

Capability numeric constraints remain the source of width, height, frame, step,
and guidance truth.

## App Rules

### Studio

- `Studio` is the steady-state center of gravity.
- The draft lives in app state, not view-local temporary state.
- Draft changes debounce planning through `/v1/workflows/plan`.
- Submission only creates visible run-group activity after the runtime accepts a
  run.
- The current run stays in the Studio canvas.

### Library

- `Library` is asset-first, not job-first.
- Generated outputs and imported assets share one asset model.
- The grid uses uniform `4:5` tiles with centered crop.
- Focused preview opens in a large in-window modal viewer, not a sidebar.
- Keyboard navigation should feel like a file browser:
  - arrows move selection
  - `Return` or `Space` opens preview
  - `Escape` closes preview, then clears selection
  - `Delete` removes imported assets only

### Activity

- `Activity` lives in the left rail footer.
- It only reflects runtime-accepted work and runtime-backed installs.
- Running and queued work stays visible.
- Failures stay visible until dismissed.
- Old completed work belongs in `Library`, not an eternal activity log.

### Models

- `Models` owns starter setup, curated installs, install queue, detail, and
  removal.
- Hugging Face is presented as the source.
- MLXR is presented as the installed home.

## Shared Implementation Principles

- no host-owned inference logic
- no host-owned scheduler logic
- no family-specific host hacks when the runtime can own the seam
- no duplicate model-default ranking tables across hosts
- no app-only validation rules that drift from runtime constraints

## Current Concrete Direction

The current hardening path is:

1. enrich runtime planning readiness and capability normalization
2. keep CLI `--plan-only` aligned with the same readiness truth
3. keep Studio runtime-led and draft-persistent
4. keep Activity secondary and truthful
5. keep Library fast, modal, and asset-first
6. keep docs aligned with the real app/runtime contract

## Done Definition

This tranche is only done when all of these hold together:

- `swift test --package-path packages/clients/mlxr-mac-app`
- targeted runtime/schema tests for planning and validation
- `uv run python scripts/dev.py verify`
- staged dev `.app` validation with Peekaboo
- runtime logs show no duplicate daemon spawning or invalid-request spam

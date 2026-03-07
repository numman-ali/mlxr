# Agent-Native Development

Status: current operating model for implementation work in `MLXR`.

## Purpose

This doc explains how the repo is meant to run day to day when Codex is the primary execution engine.

The human is expected to:

- set direction
- define success
- review outcomes
- make judgment calls when tradeoffs are real

Codex is expected to:

- inspect context
- implement changes
- run the harness
- inspect logs when runtime behavior changes
- update docs when repo truth changes
- commit only from green

Fresh sessions should bootstrap from repo files alone. The repo should not depend on a custom kickoff prompt to establish the operating model.

## Startup Order

At session start:

1. read `AGENTS.md`
2. read `MEMORY.md`
3. read the declared core docs
4. inspect `git status`
5. inspect recent commits
6. inspect the current implementation and harness state

This startup order is part of the repo contract, not optional ceremony.

## Normal Working Sequence

1. Read the repo doctrine and relevant platform docs.
2. Inspect the local implementation before making assumptions.
3. If the user has not tightly scoped the next task, derive the next step from repo state.
4. Implement the smallest complete change that satisfies the task.
5. Run the local harness.
6. If the task touches runtime behavior, inspect logs as part of validation.
7. If the task changes repo-level truth, update docs in the same pass.
8. Do a fresh-eyes review pass on the diff before commit.
9. Commit once the repo is green.

## How To Decide What To Do Next

When the user has not provided a narrowly scoped next task, derive the next step from repo evidence.

Priority order:

1. broken startup or development loop
2. broken validation, logging, or observability
3. docs-code mismatches or architecture contract drift
4. the highest-leverage incomplete implementation step implied by current docs and recent commits

Do not stall waiting for more direction when the next step is discoverable from the repo.

## Workflow Ownership Split

When work touches execution flow, keep the boundary explicit:

- the core workflow layer owns stage ordering, scheduler integration, lifecycle state, telemetry, and runtime-managed artifacts
- family adapters own truthful workflow templates, stage implementations, capability constraints, and fail-closed compatibility checks
- host adapters own UX, capability-driven request shaping, and runtime-client mapping

If a change crosses those boundaries, name the split before coding instead of letting the implementation decide it by accident.

## What Codex Owns By Default

- formatting
- linting
- type checks
- unit and integration checks
- package builds
- local runtime bring-up
- log inspection
- regression detection from local signals
- fresh-eyes review before commit

## When To Escalate

Escalate when:

- the tradeoff is product or architecture, not mechanical
- local evidence conflicts with repo docs
- the change would violate a declared platform invariant
- a destructive or user-affecting action is under consideration

Do not escalate for routine tool running, local debugging, or repetitive validation work.

## Doc Update Rule

If a code change alters repo truth, update the matching docs in the same pass.

Minimum expectation:

- platform truth: update the relevant design doc or ADR
- framing change: update `README.md`
- operating-model change: update `AGENTS.md` and this doc

For family-onboarding or workflow-boundary work, also update:

- `docs/workflow-orchestration-design.md`
- `docs/family-bringup/`
- the relevant skill doc if the specialized workflow changed

## Commit Discipline

- commit from green, not “almost green”
- keep commit messages crisp and behavior-oriented
- prefer multiple clean commits over one mixed commit when the boundaries are real
- do not leave the repo in a state where the next agent must rediscover why something is broken

## Fresh-Eyes Review Gate

Before commit, review the diff as if you did not write it.

At minimum, check:

- whether reusable workflow logic drifted into a host or family adapter
- whether claims got ahead of evidence
- whether docs, skills, and open questions moved with the new truth
- whether the host stayed thin and the family stayed honest

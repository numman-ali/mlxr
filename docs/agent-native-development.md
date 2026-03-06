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

## Normal Working Sequence

1. Read the repo doctrine and relevant platform docs.
2. Inspect the local implementation before making assumptions.
3. Implement the smallest complete change that satisfies the task.
4. Run the local harness.
5. If the task touches runtime behavior, inspect logs as part of validation.
6. If the task changes repo-level truth, update docs in the same pass.
7. Commit once the repo is green.

## What Codex Owns By Default

- formatting
- linting
- type checks
- unit and integration checks
- package builds
- local runtime bring-up
- log inspection
- regression detection from local signals

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

## Commit Discipline

- commit from green, not “almost green”
- keep commit messages crisp and behavior-oriented
- prefer multiple clean commits over one mixed commit when the boundaries are real
- do not leave the repo in a state where the next agent must rediscover why something is broken

# Open-Source Maintenance Model

## Summary

`MLXR` is an agent-native open-source repo.

That means the project is designed so a future engineer or agent can continue
the work from tracked docs, receipts, and repo truth without relying on
private chat history.

## Core Rules

- repo truth beats marketing
- docs move with implementation truth
- capabilities are promoted only with receipts
- host adapters stay thin
- family adapters keep family truth local
- the shared runtime remains the canonical contract

## Human And Agent Roles

Humans are expected to:

- set direction
- choose tradeoffs
- review outcomes
- decide what gets promoted publicly

Agents are expected to:

- inspect the repo and docs first
- implement changes
- run validation
- inspect logs when runtime behavior changes
- update docs in the same pass
- keep the repo legible for the next session

## Public Contribution Standard

A contribution is not complete if it changes repo truth without changing docs.

That especially applies to:

- capability status
- benchmark claims
- provider behavior
- host-surface behavior
- release posture

## Receipts And Durable Truth

- transient receipts belong under `tmp/`
- durable conclusions belong in tracked docs
- recurring sharp edges belong in [MEMORY.md](/Users/numman/Repos/mlxr/MEMORY.md)

Do not treat a one-off run as public truth unless the docs say it has earned
that status.

## Maintenance Rhythm

The healthy rhythm for this repo is:

1. inspect current truth
2. make the change
3. run the harness
4. inspect logs when needed
5. update docs
6. review the diff with fresh eyes

## Scope Discipline

When adding public-facing features:

- prefer the shared runtime over host-local inference
- prefer simple shared task surfaces over one-family debug knobs
- prefer optional host-side helpers over hidden runtime dependencies

This is especially important for:

- prompt enhancement
- LoRA or adapter UX
- desktop compatibility seams
- app-private convenience paths

## Why This Exists

The public repo should be understandable and maintainable by someone who was
not in the original conversation.

This document exists to make that expectation explicit.

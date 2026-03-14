# Open-Source Maintenance Model

Status: public-facing maintenance guidance for contributors

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

## Public Contribution Standard

A contribution is not complete if it changes repo truth without changing docs.

That especially applies to:

- capability status
- benchmark claims
- provider behavior
- host-surface behavior
- release posture

## Human And Agent Workflow

The normal operating split is:

- humans set direction, taste, and review criteria
- agents execute routine implementation, validation, and docs upkeep
- either side should leave enough tracked context that a fresh contributor can
  continue without private chat history

This matters most when work changes public-facing claims or project direction.

## Maintenance Rhythm

The recurring maintenance pattern is:

- refresh canonical docs when product or platform truth changes
- refresh `current-status.md` and `roadmap.md` when the public story shifts
- refresh family matrices when promoted capability truth changes
- keep active initiative material in `docs/working/`, not in canonical docs

## Receipts And Durable Truth

- transient receipts belong under `tmp/`
- durable conclusions belong in tracked docs
- recurring sharp edges belong in [MEMORY.md](../MEMORY.md)

Do not treat a one-off run as public truth unless the docs say it has earned
that status.

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

For the fuller repo operating model, validation loop, and agent workflow, use:

- [agent-native-development.md](./agent-native-development.md)
- [AGENTS.md](../AGENTS.md)

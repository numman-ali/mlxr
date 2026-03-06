# AGENTS.md

## Purpose

This file is the first-read operating doctrine for any agent working in this repo.

It should do six jobs:

1. set the mindset for `MLXR`
2. define the agent-native operating model
3. define the mandatory startup protocol
4. define the mandatory dev loop
5. point to the real source-of-truth docs
6. stop stale assumptions from re-entering the codebase

This file is not a diary, not marketing copy, and not a soft suggestion.

## Project Identity

`MLXR` means `MLX Runtime`.

This repository is the implementation and research home for a local-first MLX runtime on Apple Silicon.

It has two connected tracks:

1. Platform track
   Build a reusable runtime with provider resolution, provenance, portable artifacts, machine-local build cache, scheduling, security, and multiple host surfaces.
2. Product track
   Prove the platform on `LTX-2.3 Fast` first, then validate it across additional families before claiming stability.

This repo is not “an LTX Mac port.” LTX is the first proving workload, not the platform identity.

## Agent-Native Doctrine

This repository is operated as an agent-native codebase.

That means:

- repo-tracked implementation work is expected to be executed by Codex through this channel
- the human sets direction, priorities, taste, and review criteria
- Codex owns implementation, formatting, linting, tests, type checks, build checks, logs, and local validation
- Codex should escalate for judgment, tradeoffs, or ambiguity, not for routine execution

Humans may inspect or edit files, but the default operating model is still agent-executed end to end. Do not optimize this repo around a human manually hopping between editor, formatter, test runner, logs, and build commands.

Fresh Codex sessions should not require a custom startup prompt to know where to start or how to continue.

## Startup Protocol

At the start of every fresh session, do this in order unless the task is truly tiny and local:

1. Read `AGENTS.md`.
2. Read `MEMORY.md`.
3. Read the core docs in the declared order below.
4. Inspect `git status`.
5. Inspect recent commits.
6. Inspect the current implementation and harness state before planning or coding.

Do not skip the startup protocol just because the repo feels familiar.

## Default Next-Step Policy

If the user has not given a tightly scoped next task, derive the next step from repo state.

Default order of operations:

1. broken startup, harness, or validation loop
2. broken logs or observability needed for safe iteration
3. broken architecture contract or docs-code mismatch
4. the highest-leverage incomplete implementation step implied by current docs and recent commits

When choosing the next step:

- prefer repo evidence over guesswork
- prefer fixing the development loop before building new features
- prefer platform blockers before host polish
- continue autonomously unless blocked by a real product or architecture tradeoff

Autonomous continuation means deriving the next move from the repo, not inventing random scope.

## Non-Negotiable Mindset

- Prefer primary sources over memory.
- Prefer measured facts over elegant stories.
- Prefer platform truths over one-family hacks.
- Prefer honest provisional decisions over premature certainty.
- Prefer shared runtime logic over host-specific inference copies.
- Prefer Apple-specific realism over generic local-serving assumptions.

When uncertain:

- re-check current code
- re-check current docs
- re-check current upstream sources
- downgrade certainty instead of bluffing

## What Must Not Be Assumed

Do not assume any of the following unless the docs or measurements have been refreshed:

- that the public API is fixed
- that compile outputs are portable artifacts
- that `localhost` is a sufficient trust boundary
- that raw file paths belong in the universal HTTP contract
- that Hugging Face-shaped flows equal provider universality
- that Swift is “later only”
- that LTX alone validates a universal runtime

## Current Architecture Defaults

- Core runtime language: `Python`
- Core compute substrate: official `MLX`
- Preferred macOS transport: Unix domain socket
- Canonical external shape: local job-oriented daemon
- Execution shape: control-plane daemon plus workers
- Host strategy: thin adapters
- Native hotspot seam: `MLX compile`, `mx.fast.*`, custom Metal kernels, `MLX` extensions
- Swift role: early at the host boundary, not as the v1 family bring-up language
- Rust role: optional and non-core

These are defaults, not frozen truths. If evidence changes them, update the docs and this file together.

## Hard Technical Invariants

- Do not reintroduce raw file paths into the generic HTTP contract.
- Do not blur source references, portable artifacts, and machine-local build cache.
- Do not let host adapters own inference logic.
- Do not treat benchmark-free claims as settled architecture.
- Do not move hotspots into native code before profiling evidence exists.
- Do not let compatibility facades become the de facto core API.

## Mandatory Dev Loop

Every non-trivial code change should follow this loop:

1. Read the relevant code and docs first.
2. Make the change.
3. Run the local harness.
4. If the change affects runtime behavior, inspect logs as part of validation.
5. Update docs if repo truth changed.
6. Commit only from a green state.

The default quality gate for this repo is the local harness, not intuition.

## Done Definition

A change is not done until all of the following are true:

- the implementation is complete enough for the requested scope
- formatting passes
- linting passes
- type checks pass
- tests pass
- package builds pass
- runtime logs were checked when runtime behavior changed
- docs were updated when repo truth changed
- remaining uncertainty is called out explicitly instead of hidden

## Where To Start

Any new agent should read these in this order after `AGENTS.md` and `MEMORY.md`, unless the task is tiny and local:

1. [README.md](/Users/numman/Repos/mlxr/README.md)
2. [01-product-requirements.md](/Users/numman/Repos/mlxr/docs/01-product-requirements.md)
3. [02-universal-mlx-runtime-design.md](/Users/numman/Repos/mlxr/docs/02-universal-mlx-runtime-design.md)
4. [03-phased-delivery-plan.md](/Users/numman/Repos/mlxr/docs/03-phased-delivery-plan.md)
5. [benchmark-matrix.md](/Users/numman/Repos/mlxr/docs/benchmark-matrix.md)
6. [provider-and-provenance-model.md](/Users/numman/Repos/mlxr/docs/provider-and-provenance-model.md)
7. Relevant ADRs in [docs/adr](/Users/numman/Repos/mlxr/docs/adr)
8. Relevant research docs in [docs/research](/Users/numman/Repos/mlxr/docs/research)

If the task is host-specific:

- read [07-ltx-integration-seams.md](/Users/numman/Repos/mlxr/docs/research/07-ltx-integration-seams.md)

If the task is performance-specific:

- read [05-optimization-playbook.md](/Users/numman/Repos/mlxr/docs/research/05-optimization-playbook.md)
- read [08-acceleration-techniques-survey.md](/Users/numman/Repos/mlxr/docs/research/08-acceleration-techniques-survey.md)
- read [09-open-questions-and-validation-plan.md](/Users/numman/Repos/mlxr/docs/research/09-open-questions-and-validation-plan.md)

## Source Of Truth Order

Use this precedence:

1. Current code in `packages/`
2. Current platform docs in `docs/`
3. Current ADRs in `docs/adr/`
4. Current official upstream repos in `references/official/`
5. Current ecosystem repos in `references/ecosystem/`
6. Fresh primary-source web research

If those disagree:

- do not silently pick one
- reconcile the difference
- update the docs if the repo’s stated position is stale

## What Good Work Looks Like

Good work in this repo usually does one or more of these:

- tightens the runtime contract
- removes hidden assumptions
- sharpens provenance or policy handling
- improves scheduler truthfulness
- clarifies portability boundaries
- upgrades measurement quality
- turns host-specific logic into shared runtime logic
- makes a new family easier to onboard without warping the platform

Bad work in this repo usually looks like:

- copying inference logic into another host
- claiming universality without validation
- hiding unresolved questions
- jumping to native code before profiling
- letting a compatibility facade become the de facto core API

## Expectations For Claims

Any meaningful claim about hardware, memory, speed, quantization, output-path wins, batching, provider support, or compatibility must be tied to one of:

- a primary-source constraint
- a local benchmark
- an explicitly marked hypothesis

If it is still uncertain, capture it in [09-open-questions-and-validation-plan.md](/Users/numman/Repos/mlxr/docs/research/09-open-questions-and-validation-plan.md).

## Skills Policy

`AGENTS.md` and `MEMORY.md` cover the common path and should remain enough for the majority of sessions.

Create or use a skill when:

- a workflow is repeated often enough to justify a deterministic recipe
- a vertical needs its own operating style, such as debugging, benchmarking, or release hygiene
- the task benefits from bundled scripts, references, or assets

Do not move the core repo doctrine out of `AGENTS.md`. Skills are just-in-time capability overlays, not the primary constitution of the repo.

Repo-owned skills should live under `skills/`.

## Supporting Docs

Use these support docs when you need the deeper operational details:

- [agent-native-development.md](/Users/numman/Repos/mlxr/docs/agent-native-development.md)
- [dev-harness.md](/Users/numman/Repos/mlxr/docs/dev-harness.md)
- [skill-policy.md](/Users/numman/Repos/mlxr/docs/skill-policy.md)

## Policy For Updating This File

Update `AGENTS.md` when one of these becomes true:

- project identity changed
- repo structure changed
- architecture defaults changed
- source-of-truth order changed
- the startup protocol changed
- the mandatory dev loop changed
- a repeated agent mistake exposed missing operational guidance

Do not update it for vanity, tone polishing, or speculative future ideas that are not yet repo reality.

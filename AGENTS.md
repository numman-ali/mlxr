# AGENTS.md

## Purpose

This file is the first-read operating guide for any agent working in this repo.

It should do six jobs:

1. set the mindset for `MLXR`
2. define the agent-native operating model
3. define the mandatory startup protocol
4. define the mandatory dev loop
5. point to the real source-of-truth docs
6. stop stale assumptions from re-entering the codebase

Keep this file practical, durable, and grounded in repo reality.

## Repo Guidance Layers

- `AGENTS.md` owns repo doctrine: mindset, startup protocol, dev loop, invariants, and source-of-truth order.
- `MEMORY.md` owns durable repo facts and sharp edges that future sessions should not have to rediscover.
- Avoid duplicating the same rule in both files unless the repetition is intentionally temporary during a transition.

## Project Identity

`MLXR` means `MLX Runtime`.

This repository is the implementation and research home for a local-first MLX runtime on Apple Silicon.

It has two connected tracks:

1. Platform track
   Build a reusable runtime with provider resolution, provenance, portable artifacts, machine-local build cache, workflow planning, scheduling, security, and multiple host surfaces.
2. Product track
   Prove the platform on `LTX-2.3 Fast` first, then validate it across additional families before claiming stability.

This repo is not “an LTX Mac port.” LTX is the first proving workload, not the platform identity.

## Repo Layout

The workspace is grouped by role rather than by historical package names:

- `packages/core/` contains shared platform packages such as schemas, runtime core, server, and workflow planning
- `packages/families/` contains model-family packages; `LTX` currently lives at `packages/families/ltx/` with Python sources under `src/mlxr/families/ltx/`
- `packages/adapters/` contains host or integration adapters such as `comfyui` and `ltx-desktop`
- `packages/clients/` contains first-party clients such as `runtime-cli`

Python imports should now use the shared `mlxr.*` namespace:

- `mlxr.core.*` for shared platform packages
- `mlxr.families.*` for model families
- `mlxr.clients.*` for first-party clients
- `mlxr.adapters.*` for host or integration adapters as they are re-based

The shared `mlxr`, `mlxr.core`, `mlxr.families`, `mlxr.clients`, and `mlxr.adapters` roots are implicit namespace packages. Do not add `__init__.py` files just to make those roots importable.

Tests should mirror ownership too:

- package-specific unit tests live under the package root at `packages/.../tests/unit/`
- top-level `tests/` is reserved for integration, e2e, shared fixtures, and legacy test files that have not been migrated yet
- do not add new package-specific unit coverage to giant root-level monolithic test files when a package-local unit test home exists

When moving files or updating imports, keep both layers clean:

- repo paths should communicate the package role clearly
- Python package names should stay short, semantic, and non-duplicative

## Agent-Native Operating Model

This repository is operated as an agent-native codebase.

That means:

- repo-tracked implementation work is expected to be executed by Codex through this channel
- the human sets direction, priorities, taste, and review criteria
- Codex is expected to carry implementation, formatting, linting, tests, type checks, build checks, logs, and local validation
- Codex should surface judgment calls, tradeoffs, and ambiguity rather than routine execution questions

Humans may inspect or edit files, but the default operating model is still agent-executed end to end. Avoid shaping the repo around a human manually hopping between editor, formatter, test runner, logs, and build commands.

Fresh Codex sessions should not require a custom startup prompt to know where to start or how to continue.

## Startup Protocol

At the start of every fresh session, do this in order unless the task is truly tiny and local:

1. Read `AGENTS.md`.
2. Read `MEMORY.md`.
3. Read the core docs in the declared order below.
4. Inspect `git status`.
5. Inspect recent commits.
6. Inspect the current implementation and harness state before planning or coding.

Do not skip the startup protocol just because the repo feels familiar; that is how stale assumptions re-enter.

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
- recommend one next step clearly instead of defaulting to an unprioritized dump of possibilities

Autonomous continuation means deriving the next move from the repo, not inventing random scope.

When reporting the next step back to the user:

- start with a short plain-English recommendation
- connect the recommendation to the current phase, product goal, or repo gap it closes
- use code and doc references as support, not as the opening move
- keep the first answer orienting; add deeper implementation evidence after the user understands the direction

## Repo Communication Defaults

When explaining repo status, architecture, tradeoffs, or implementation progress:

- orient first, then substantiate
- start with the direct answer in plain English before diving into file-level evidence
- connect low-level details back to what they mean for `MLXR` as a platform or proving path
- prefer one clear recommendation over a neutral wall of technically correct observations
- use file references to support claims, not to carry the whole explanation
- when a system shape is hard to picture, offer a compact mental model or ASCII diagram
- optimize for clarity and cognitive load, especially when the user is asking for orientation rather than a code review

## Mindset Defaults

- Prefer primary sources over memory.
- Prefer measured facts over elegant stories.
- Prefer platform truths over one-family hacks.
- Prefer honest provisional decisions over premature certainty.
- Prefer shared runtime logic over host-specific inference copies.
- Prefer Apple-specific realism over generic local-serving assumptions.
- Prefer readable, modular seams over clever convenience code.
- Prefer inversion of control, typed interfaces, and explicit boundaries over hard-wired concrete coupling.
- Prefer repo-enforced quality rules over “remember not to do that next time.”
- For external SDKs, provider CLIs, auth flows, and hosted-service integrations, validate current official docs and live behavior before narrowing the repo around an assumption from local wrapper code.

When uncertain:

- re-check current code
- re-check current docs
- re-check current upstream sources
- downgrade certainty instead of bluffing

## Assumptions To Avoid

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
- Workflow layer: core runtime-owned workflow planning between scheduler, family adapters, and host adapters
- Host strategy: thin adapters
- Native hotspot seam: `MLX compile`, `mx.fast.*`, custom Metal kernels, `MLX` extensions
- Swift role: early at the host boundary, not as the v1 family bring-up language
- Rust role: optional and non-core

These are defaults, not frozen truths. If evidence changes them, update the docs and this file together.

## Hard Technical Invariants

- Do not reintroduce raw file paths into the generic HTTP contract.
- Do not blur source references, portable artifacts, and machine-local build cache.
- Do not let host adapters own inference logic.
- Do not let family adapters or host adapters become the de facto home for reusable workflow planning.
- Do not leak family-specific validation or parity knobs into the shared product surface just because they help debug one current engine. If a control is not yet a clean cross-family concept, keep it family-local or diagnostic-only until the shared abstraction is designed deliberately.
- Do not treat benchmark-free claims as settled architecture.
- Do not move hotspots into native code before profiling evidence exists.
- Do not let compatibility facades become the de facto core API.
- Do not let model-family bridges silently default or reshape through critical config or weight mismatches; fail closed when the checkpoint contract is uncertain.
- Do not treat pipeline completion as visual correctness on model-family bring-up; promote claims only after stage-local and human-visible validation.
- Do not use file-level validation bypasses such as `# mypy: ignore-errors` as a shortcut. If typing is broken, fix the typing, narrow the seam, add the needed stubs or protocols, or delete the dead code.
- Do not hide validation debt behind broad suppressions. Per-line ignores must be rare, specific, and justified by a real upstream typing gap, not by local convenience.

## Mandatory Dev Loop

Every non-trivial code change should follow this loop:

1. Read the relevant code and docs first.
2. Make the change.
3. Run the local harness.
4. If the change affects runtime behavior, inspect logs as part of validation.
5. Update docs if repo truth changed.
6. Do a fresh-eyes review pass on your own diff before commit.
7. Commit only from a green state.

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
- fresh-eyes review was completed before commit
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

If the task is family-bring-up-specific:

- read [workflow-orchestration-design.md](/Users/numman/Repos/mlxr/docs/workflow-orchestration-design.md)
- read [docs/family-bringup/README.md](/Users/numman/Repos/mlxr/docs/family-bringup/README.md)
- use the repo-owned `.agents/skills/family-bringup/` workflow

If the task is LTX-fidelity-specific:

- read [10-ltx-fidelity-debug-playbook.md](/Users/numman/Repos/mlxr/docs/research/10-ltx-fidelity-debug-playbook.md)
- use the repo-owned `.agents/skills/ltx-fidelity-debugging/` workflow

If the task is LTX-capability-specific:

- read [11-ltx-capability-matrix.md](/Users/numman/Repos/mlxr/docs/research/11-ltx-capability-matrix.md)
- read [12-ltx-prompting-guide.md](/Users/numman/Repos/mlxr/docs/research/12-ltx-prompting-guide.md) when text-first prompt quality or showcase work is in scope
- read [13-ltx-showcase-execution-plan.md](/Users/numman/Repos/mlxr/docs/research/13-ltx-showcase-execution-plan.md) when continuing the current capability-to-showcase tranche
- read [14-first-showcase-pack.md](/Users/numman/Repos/mlxr/docs/research/14-first-showcase-pack.md) when validating, replacing, or promoting the current five-clip showcase set
- read [18-ltx-reference-map.md](/Users/numman/Repos/mlxr/docs/research/18-ltx-reference-map.md) when implementing or validating against official LTX semantics
- read [19-ltx-compatibility-checklist.md](/Users/numman/Repos/mlxr/docs/research/19-ltx-compatibility-checklist.md) when deciding whether an upstream LTX feature is still missing, merely planned, or truly complete
- read [16-owned-substrate-migration-plan.md](/Users/numman/Repos/mlxr/docs/research/16-owned-substrate-migration-plan.md) when donor-runtime removal, non-distilled recovery, or Gemma substrate ownership is in scope
- read [17-multimodal-inference-engine.md](/Users/numman/Repos/mlxr/docs/research/17-multimodal-inference-engine.md) when replacing `mlx-vlm` / `mlx-lm` or deciding what shared MLX multimodal execution logic belongs in `packages/core/`

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
- keeps reusable workflow planning in the core workflow layer instead of scattering it across hosts or families
- keeps product-surface controls intentional and layered: simple user intent at the edge, shared typed profiles only when they genuinely generalize, and family-specific debug controls kept out of the shared/public surface until proven reusable
- makes a new family easier to onboard without warping the platform
- improves agent legibility through modularity, separation of concerns, and explicit dependency seams
- reduces the need for unsafe typing escape hatches such as avoidable `typing.cast` or broad `Any` in runtime code

Patterns to avoid in this repo usually look like:

- copying inference logic into another host
- claiming universality without validation
- hiding unresolved questions
- jumping to native code before profiling
- teaching a host adapter to orchestrate family stages the core runtime should own
- letting a compatibility facade become the de facto core API
- solving design friction with type escapes or implicit coupling instead of fixing the seam
- relying on reviewer memory for preventable bad practices instead of encoding them into harness or hook checks

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

Do not move the core repo doctrine out of `AGENTS.md`. Skills are just-in-time capability overlays, not the primary operating guide.

Repo-owned skills should live under `.agents/skills/` so Codex can discover them natively. Use `.codex/` for project-local Codex config, not as the repo skill directory.

## Supporting Docs

Use these support docs when you need the deeper operational details:

- [agent-native-development.md](/Users/numman/Repos/mlxr/docs/agent-native-development.md)
- [dev-harness.md](/Users/numman/Repos/mlxr/docs/dev-harness.md)
- [test-layout.md](/Users/numman/Repos/mlxr/docs/test-layout.md)
- [workflow-orchestration-design.md](/Users/numman/Repos/mlxr/docs/workflow-orchestration-design.md)
- [docs/family-bringup/README.md](/Users/numman/Repos/mlxr/docs/family-bringup/README.md)
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

Do not update it for style churn, vanity edits, or speculative future ideas that are not yet repo reality.

# AGENTS.md

## Purpose

This file is the repo-local operating doctrine for any agent working in this workspace.

It should do four jobs:

1. explain what this project is
2. explain how to start work without drifting into stale assumptions
3. explain how to think and make decisions here
4. preserve durable repo-operating knowledge across sessions

Treat this file as repo memory with teeth. It is not marketing copy.

## Project Identity

This repository is the implementation and research home for `MLXR`, an MLX-native local runtime on Apple Silicon.

`MLXR` means `MLX Runtime`.

The project has two connected tracks:

1. Platform track
   Build a reusable Apple Silicon runtime with provider resolution, provenance, portable artifacts, machine-local build cache, scheduling, security, and host surfaces.
2. Product track
   Prove the platform on `LTX-2.3 Fast` first, then validate it across additional families before claiming stability.

This repo is not “an LTX Mac port.” LTX is the first proving workload, not the identity of the platform.

## Non-Negotiable Mindset

- Prefer primary sources over memory.
- Prefer measured facts over elegant stories.
- Prefer platform truths over one-family hacks.
- Prefer honest provisional decisions over premature certainty.
- Prefer shared runtime logic over host-specific inference copies.
- Prefer Apple-specific realism over generic local-serving assumptions.

When in doubt:

- re-check upstream docs or code
- re-check the benchmark and validation docs
- downgrade certainty instead of bluffing

## What Must Not Be Assumed

Do not assume any of the following unless the docs or measurements have been refreshed:

- that the public API is fixed
- that build or compile outputs are portable artifacts
- that `localhost` is a sufficient trust boundary
- that raw file paths belong in the generic HTTP contract
- that Hugging Face-shaped flows equal provider universality
- that Swift is a “later only” concern
- that LTX alone is enough to validate a universal runtime

## Current Architecture Defaults

- Core runtime language: `Python`
- Core compute substrate: `official MLX`
- Preferred external transport on macOS: `Unix domain socket`
- Canonical external shape: `local job-oriented daemon`
- Execution shape: `control-plane daemon + workers`
- Host strategy: `thin adapters`
- Native hotspot seam: `MLX compile`, `mx.fast.*`, `custom Metal kernels`, `MLX extensions`
- Swift role: early at the host boundary, not as the v1 family bring-up language
- Rust role: optional and non-core

These are defaults, not eternal truths. If evidence changes them, update the docs and this file together.

## Where To Start

Any new agent should start in this order unless the task is tiny and local:

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
- update the docs if the repo’s stated position is now stale

## Reference Repo Rules

### `references/official/`

Use for highest-authority upstream repos maintained by the original authors or organization.

Examples:

- `ml-explore/*`
- `Lightricks/*`

### `references/ecosystem/`

Use for strong community projects that inform:

- serving patterns
- conversion strategies
- model-family feasibility
- optimization ideas

These are valuable inputs, not default truth over first-party sources.

These reference repos are intentionally gitignored and not part of the authored source tree.

## Durable Repo Rules

### Platform versus product

Keep platform and product thinking separate.

- Platform track asks: what is reusable, truthful, secure, and benchmarkable?
- Product track asks: how do we make the first real user path excellent?

Do not let LTX urgency freeze the wrong platform decisions.

### Portability classes

Keep these distinct:

1. source references and provenance
2. portable converted artifacts
3. machine-local build cache

If a change blurs these together, it is probably wrong.

### Security

Default assumptions:

- UDS first
- HTTP opt-in
- mandatory auth on mutating HTTP routes
- browser-origin protections when HTTP exists
- handle-based generic API
- `trust_remote_code` off by default

### Capability honesty

Hosts must learn constraints, outputs, profiles, hardware tiers, and policy from the runtime, not from scattered README text.

### Benchmark honesty

No architectural claim is stable until it survives the benchmark matrix and cross-family validation.

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

Any meaningful claim about:

- hardware support
- memory requirements
- speedups
- quantization quality
- output-path improvements
- batching gains
- provider support
- compatibility

must be tied to one of:

- a primary-source constraint
- a local benchmark
- an explicitly marked hypothesis

If it is still uncertain, capture it in [09-open-questions-and-validation-plan.md](/Users/numman/Repos/mlxr/docs/research/09-open-questions-and-validation-plan.md).

## Expectations For Editing Docs

When repo-level truth changes, update:

- the relevant design doc
- any affected ADR
- `README.md` if the change affects project framing
- this `AGENTS.md` if the change affects how future agents should operate

Do not leave repo-operating rules stranded only in conversation context.

## Policy For Updating AGENTS.md

Agents are allowed to update this file when one of these is true:

- the project identity changed
- the repo structure changed
- the architecture defaults changed
- the source-of-truth order changed
- a repeated agent mistake exposed missing operational guidance
- new memory or workflow guidance would materially help future sessions

Agents should not update this file for vanity, tone polishing, or speculative future ideas that are not yet repo reality.

If you update `AGENTS.md`, keep it:

- durable
- operational
- short enough to be read on entry
- aligned with the actual docs

## Memory Guidance

If a persistent memory system is available, this repo should be treated as a long-lived project entity.

Durable items worth writing to memory:

- repo identity and any rename
- frozen architectural decisions
- provider and security model decisions
- benchmark-backed hardware truths
- high-value user preferences specific to this repo
- known traps or repeated failure modes

Do not rely on temporary session context to preserve those.

When memory is updated, prefer storing:

- concise project summaries
- durable facts
- superseding decisions rather than deleting older ones

## Rename Guidance

If the parent directory is renamed again away from `mlxr`, update:

- `README.md`
- this file
- any plan or memory entry that still uses the old name as if it were the project identity

The rename should make the platform sound broader than LTX while still sounding concrete and technical.

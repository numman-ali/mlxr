# MLXR Workspace

Last verified: 2026-03-06

This repository is the planning and implementation workspace for `MLXR`, a local-first MLX runtime on Apple Silicon.

The current scope we can defend is narrower and more honest than the first draft:

- universal across host surfaces: CLI, native API, desktop adapters, Comfy-style adapters, and embedded first-party access
- one core workflow-orchestration layer between scheduler, family adapters, and host adapters
- broad across local generative and multimodal families on Apple Silicon
- provider-extensible by architecture, but not provider-universal in v1 implementation
- not yet frozen as a final architecture or public API

`MLXR` stands for `MLX Runtime`.

## Working Scope

The platform target is:

- a shared Apple Silicon runtime for local generative and multimodal workloads
- one canonical native job API
- one shared source, artifact, provenance, and scheduling model
- one core workflow layer that can serve multiple product surfaces without each host owning inference logic or stage sequencing

The first product proving path is `LTX-2.3 Fast` text-to-video and image-to-video, but LTX urgency is no longer allowed to freeze the universal runtime design prematurely.

The current design does not claim to be universal across every possible MLX workload. Pure LLM and embedding workloads are now treated as a platform validation pressure test, not an implied solved problem.

## What Is Not Frozen Yet

These items remain provisional until the benchmark matrix and cross-family validation pass:

- the exact native HTTP surface beyond the current handle-based core contract
- whether the preferred worker topology stays per-family or becomes more granular per-job
- the final storage layout under `$MLX_RUNTIME_HOME`
- the exact benchmark-backed hardware tiers for `32 GB`, `64 GB`, and `128 GB` systems
- how much of the video/audio output path should move into native extensions early
- whether a second provider beyond Hugging Face and local bundles is part of the first stable release

## MLXR Platform Track

- [Product requirements](docs/01-product-requirements.md)
- [Universal runtime design](docs/02-universal-mlx-runtime-design.md)
- [Phased delivery plan](docs/03-phased-delivery-plan.md)
- [Workflow orchestration design](docs/workflow-orchestration-design.md)
- [Capability schema](docs/capability-schema.md)
- [Benchmark matrix](docs/benchmark-matrix.md)
- [Provider and provenance model](docs/provider-and-provenance-model.md)
- [Operations and packaging](docs/operations-and-packaging.md)

## MLXR Product And LTX Track

- [Upstream baseline](docs/research/01-upstream-baseline.md)
- [Model research matrix](docs/research/02-model-research-matrix.md)
- [Language and runtime choice](docs/research/03-language-and-runtime-choice.md)
- [Serving architecture and API](docs/research/04-serving-architecture-and-api.md)
- [Optimization playbook](docs/research/05-optimization-playbook.md)
- [Source catalog](docs/research/06-source-catalog.md)
- [LTX integration seams](docs/research/07-ltx-integration-seams.md)
- [Acceleration techniques survey](docs/research/08-acceleration-techniques-survey.md)
- [Open questions and validation plan](docs/research/09-open-questions-and-validation-plan.md)

## ADRs

- [ADR-0001: Source provider adapter](docs/adr/0001-source-provider-adapter.md)
- [ADR-0002: Artifact vs build cache](docs/adr/0002-artifact-vs-build-cache.md)
- [ADR-0003: Local security model](docs/adr/0003-local-security-model.md)
- [ADR-0004: Worker process topology](docs/adr/0004-worker-process-topology.md)
- [ADR-0005: MLX extension and upstream roadmap](docs/adr/0005-mlx-extension-and-upstream-roadmap.md)

## Current External Assumptions Verified On 2026-03-06

These are the baseline assumptions the rewritten docs are built around. They are intentionally specific because these are the areas most likely to drift.

| Area | Current signal |
| --- | --- |
| MLX core | Local clone at `be872ebd` from `2026-03-05`; compile remains shape-sensitive and purity-sensitive; exporter/importer remains experimental. |
| MLX C | Local clone at `1370f59` from `2026-03-05`; C surface is an official bridge layer, not a separate runtime. |
| MLX Swift | Local clone at `a2f0d76` from `2026-02-23`; Swift is now important for host embedding and Apple-native integration, not just a far-future option. |
| LTX runtime repo | Local `LTX-2` clone at `9e8a28e` from `2026-03-05`; official repo points to `LTX-2.3` weights and current pipelines, including audio-video flows. |
| LTX Desktop | Local clone at `32589e6` from `2026-03-05`; current official desktop app still treats macOS as API-only, which is exactly the gap this runtime is meant to close. |
| Comfy + LTX | `ComfyUI-LTXVideo` README now states that `LTX-2` is built into ComfyUI core and the repo adds advanced nodes and workflows. |
| LTX 2.3 weights | Official model card and files are live; `Diffusers` support is described as “coming soon”; the community license is not a generic OSS license. |
| Active MLX ecosystem | `mlx-vlm`, `mlx-audio`, and `vllm-metal` all moved on `2026-03-05`; the ecosystem is alive enough to mine for serving patterns, but not authoritative for the core architecture. |

## Reference Repos

Upstream repos are kept under `references/` and are intentionally not part of this repository's versioned source.

- `references/official/`: first-party upstream repos maintained by the original project authors
- `references/ecosystem/`: community repos that shape the research, benchmark, and integration strategy

“Official” in this workspace means “highest-authority external reference,” not “our implementation.”

## Workspace Layout

```text
mlxr/
  README.md
  AGENTS.md
  docs/
  packages/
  references/
    official/
    ecosystem/
  scripts/
  benchmarks/
```

The `packages/` tree is reserved for implementation work authored here. The `references/` tree is for upstream inspection and should remain gitignored.

## Development Setup

The repo uses a `uv` workspace rooted at the top-level `pyproject.toml`.

Use:

```bash
uv sync
```

That installs the active Python workspace packages plus the default `dev` group into `.venv/`.

Useful commands:

```bash
uv run python scripts/dev.py verify
uv run python scripts/dev.py logs
uv run pre-commit run --all-files
```

`verify` is the main acceptance gate. It runs formatting, linting, strict typing, the repo-owned type-escape check, the `unittest` suite under `coverage.py`, enforces the current `85%` package coverage floor, and builds all workspace packages.

Only directories under `packages/` that contain real Python package metadata are workspace members. Placeholder adapter folders stay outside the workspace until they gain their own `pyproject.toml`.

Repo operating docs:

- [AGENTS.md](AGENTS.md)
- [MEMORY.md](MEMORY.md)
- [agent-native-development.md](docs/agent-native-development.md)
- [dev-harness.md](docs/dev-harness.md)
- [workflow-orchestration-design.md](docs/workflow-orchestration-design.md)
- [family-bringup/README.md](docs/family-bringup/README.md)
- [skill-policy.md](docs/skill-policy.md)

Repo-owned Codex skills live under `.agents/skills/`. Use `.codex/` only for project-local Codex config overrides.

Fresh Codex sessions are expected to bootstrap from the repo itself:

1. `AGENTS.md`
2. `MEMORY.md`
3. the core docs listed in `AGENTS.md`
4. current repo state such as `git status`, recent commits, and the local harness state

# Contributing To MLXR

`MLXR` is an agent-native, local-first runtime project for Apple Silicon.

We welcome contributions, but we want the public repo to stay honest about what
is real, what is merely implemented, and what is still planned.

## Before You Start

Read these first:

- [README.md](/Users/numman/Repos/mlxr/README.md)
- [current-status.md](/Users/numman/Repos/mlxr/docs/current-status.md)
- [roadmap.md](/Users/numman/Repos/mlxr/docs/roadmap.md)
- [AGENTS.md](/Users/numman/Repos/mlxr/AGENTS.md)
- [agent-native-development.md](/Users/numman/Repos/mlxr/docs/agent-native-development.md)

If your change is family-specific, also read the relevant family matrix or
family candidate doc under [docs/research](/Users/numman/Repos/mlxr/docs/research).

## What Good Contributions Look Like

- tighten the runtime contract
- improve capability truthfulness
- keep host adapters thin
- improve receipts, validation, or observability
- update docs when repo truth changes
- avoid letting marketing claims get ahead of evidence

## Repo Rules That Matter

- Do not move inference ownership into a host adapter.
- Do not reintroduce raw file paths into the generic runtime HTTP contract.
- Do not claim a capability as promoted without real receipts.
- Do not add family-specific debug knobs to the shared public surface unless
  they have earned a deliberate shared design.
- Keep provider, provenance, and license facts intact.

## Development Setup

```bash
uv sync
```

Useful commands:

```bash
uv run python scripts/dev.py verify
uv run python scripts/dev.py logs
uv run pre-commit run --all-files
```

`verify` is the main repo gate.

## Contribution Flow

1. Inspect the current implementation and docs before making assumptions.
2. Make the smallest complete change that closes the problem truthfully.
3. Run the local harness.
4. Inspect runtime logs when runtime behavior changed.
5. Update docs in the same pass if repo truth changed.
6. Review the diff with fresh eyes before submitting it.

## Claims And Receipts

This repo is strict about evidence.

If you change anything related to:

- capability claims
- speed or memory
- quality or fidelity
- provider support
- model compatibility

you should either:

- attach a local receipt, benchmark, or log-backed validation result, or
- explicitly mark the claim as a hypothesis or planned work

Temporary receipts belong under `tmp/`. Durable conclusions belong in tracked
docs and, when appropriate, [MEMORY.md](/Users/numman/Repos/mlxr/MEMORY.md).

## Pull Request Checklist

- the change matches the current roadmap or updates the roadmap
- docs moved with the new truth
- public claims are backed by evidence
- the repo gate was run, or any remaining failure is called out clearly
- new files or APIs do not undermine the runtime-first architecture

## Licensing And Models

The repo code is intended to be released under Apache-2.0.

Model families preserve upstream model licenses independently. A permissive repo
license does not make upstream model weights redistributable.

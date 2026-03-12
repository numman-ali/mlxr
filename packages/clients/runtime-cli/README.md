# mlxr CLI

`mlxr` is the first-party command-line client for `MLXR`.

It stays intentionally thin over the shared runtime. The CLI should expose the
runtime's real task surface without becoming a second orchestration layer or a
family-specific dumping ground.

## What It Is For

- trusted local use by humans
- predictable local automation
- agent access to the shared runtime
- fast inspection of jobs, artifacts, and model state as the public surface grows

## Current Real Surface

The current real entrypoints are:

- `mlxr generate`
- `mlxr serve`
- `mlxr doctor`
- `mlxr models list`
- `mlxr models install`
- `mlxr feedback`

The current CLI already supports:

- workflow-oriented generation requests
- local UDS-first transport by default
- no-args help with examples, docs, and feedback links
- automatic startup of a reusable local daemon for runtime-backed commands
- curated supported-model discovery and install
- `--plan-only`
- `--wait`
- trusted local export of the first output artifact with `--wait --export-path`

## Install And First Run

Published install target:

```bash
uv tool install mlxr
mlxr
```

Repo-local development flow:

```bash
uv sync
uv run mlxr
```

## Public Growth Path

Still-growing command groups:

- `mlxr jobs *`

The current CLI ergonomics direction is documented in:

- [`docs/cli-ergonomics.md`](../../../docs/cli-ergonomics.md)

## Relationship To Other Surfaces

- the daemon API is the canonical runtime contract
- the CLI is the current public thin client
- the future Mac app should reuse the same runtime behaviors rather than invent
  a separate local app-only path

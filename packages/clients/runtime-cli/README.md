# runtime-cli

`runtime-cli` is the first-party command-line client for `MLXR`.

It is intentionally thin over the shared runtime. The CLI should expose the
runtime's real task surface without becoming a second orchestration layer or a
family-specific dumping ground.

## What It Is For

- trusted local use by humans
- predictable local automation
- agent access to the shared runtime
- fast inspection of jobs, artifacts, and model state as the public surface grows

## Current Real Surface

The currently real entrypoint is:

- `mlxr generate`

That command already supports:

- workflow-oriented generation requests
- local UDS-first transport by default
- `--plan-only`
- `--wait`
- trusted local export of the first output artifact with `--wait --export-path`

## Public Growth Path

These command groups are intended public follow-ons:

- `mlxr serve`
- `mlxr models *`
- `mlxr jobs *`

The current CLI ergonomics direction is documented in:

- [`docs/cli-ergonomics.md`](../../../docs/cli-ergonomics.md)

## Relationship To Other Surfaces

- the daemon API is the canonical runtime contract
- the CLI is the current public thin client
- the future Mac app should reuse the same runtime behaviors rather than invent
  a separate local app-only path

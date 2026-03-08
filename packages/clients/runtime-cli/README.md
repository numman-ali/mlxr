# runtime-cli

This package is the first-party thin client over the local daemon.

Current command groups:

- `mlxr generate`
  - plans and runs the current workflow-oriented generation surface
  - defaults to the local Unix-domain socket transport
  - supports `--plan-only`
  - supports `--wait`
  - supports trusted local export of the first output artifact with `--wait --export-path`

Planned or still-growing command groups:

- `mlxr serve`
- `mlxr models *`
- `mlxr run`
- `mlxr jobs *`

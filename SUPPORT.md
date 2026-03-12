# Support

## Start Here

Before opening an issue, check:

- [README.md](/Users/numman/Repos/mlxr/README.md)
- [current-status.md](/Users/numman/Repos/mlxr/docs/current-status.md)
- [roadmap.md](/Users/numman/Repos/mlxr/docs/roadmap.md)
- the relevant family matrix under [docs/research](/Users/numman/Repos/mlxr/docs/research)

## Best Issue Reports

The most helpful reports include:

- the exact model or model ID
- the exact command or workflow request
- machine details and macOS version
- relevant logs from `$MLX_RUNTIME_HOME/logs/control-plane.log`
- whether the problem is:
  - a runtime bug
  - a docs mismatch
  - a capability claim problem
  - a feature request

## Scope

Current support is best-effort and focused on:

- the shared runtime
- the current first-party CLI
- currently documented family surfaces
- docs and capability truthfulness

We are less likely to prioritize:

- unsupported upstream rows
- speculative product requests outside the roadmap
- custom local environments without a reproducible case

## Maintenance Model

`MLXR` is maintained as an agent-native repo. That means the project is meant
to stay executable, measurable, and easy for a future agent or engineer to
continue without private context.

See [open-source-maintenance.md](/Users/numman/Repos/mlxr/docs/open-source-maintenance.md).

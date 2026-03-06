# ADR-0002: Artifact Vs Build Cache

Status: proposed

## Decision

Split storage into three portability classes:

1. source references and provenance
2. portable converted artifacts
3. machine-local build cache

Machine-local compile and build state is explicitly not part of the portable artifact contract.

## Why

The older design treated compile outputs and build state as if they were durable artifact content.

That does not survive current MLX evidence:

- compile behavior is shape-sensitive
- exporter and importer are still experimental
- native kernel builds are machine-local

## Consequences

### Positive

- the artifact contract becomes honest
- cache invalidation becomes more tractable
- multi-machine portability is less likely to be overclaimed

### Negative

- users and implementers now need to understand more than one cache layer
- warm performance cannot be described only in terms of artifact reuse

## Notes

Portable artifacts may contain weights, processors, and metadata. They do not contain machine-local compiled traces, kernel binaries, or other build products tied to a specific MLX or macOS environment.

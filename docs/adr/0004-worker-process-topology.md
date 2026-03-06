# ADR-0004: Worker Process Topology

Status: proposed

## Decision

Use a control-plane daemon plus execution workers as the preferred topology.

The daemon owns:

- transport and auth
- provider resolution
- provenance and artifact indexes
- scheduler
- job state and event streaming

Workers own:

- family-specific loading
- model execution stages
- MLX compile and build-cache activity
- optional native extensions

## Why

This topology is a better fit than a monolithic single process because it improves:

- crash isolation
- memory reclamation
- dependency isolation
- remote-code containment
- profiling clarity per family

## Consequences

### Positive

- cleaner safety and policy boundaries
- easier future per-family or per-job specialization
- better operational reasoning

### Negative

- more IPC and orchestration complexity
- higher cold-start overhead if workers are not reused carefully

## Notes

This ADR does not freeze the exact worker granularity yet. It freezes the direction: control-plane responsibilities stay separated from execution responsibilities.

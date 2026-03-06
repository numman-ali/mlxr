# ADR-0001: Source Provider Adapter

Status: proposed

## Decision

Introduce a first-class `SourceProviderAdapter` layer parallel to `ModelFamilyAdapter`.

The runtime architecture models multiple provider classes now, even though the first implementation only requires:

- Hugging Face
- trusted local filesystem bundles

## Why

The earlier design was family-broad but provider-narrow. That caused:

- hidden Hugging Face assumptions in source resolution
- weak provenance modeling
- no clean place for auth, license, access-state, or remote-code policy

Provider scope is architecture, not a convenience wrapper around downloads.

## Consequences

### Positive

- provenance becomes a platform concern
- provider auth and policy stop leaking into host adapters
- the architecture can honestly grow beyond one source type

### Negative

- more abstraction surface early
- more manifest and policy plumbing before the first full product flow

## v1 Provider Scope

Implemented:

- Hugging Face
- trusted local bundles

Modeled but not promised in v1:

- GitHub releases or repos
- OCI-style registries
- object-store backed registries
- internal enterprise registries

## Notes

This ADR does not freeze the exact provider interface yet. It freezes the existence of the provider seam and the need to persist provenance and policy with resolved sources.

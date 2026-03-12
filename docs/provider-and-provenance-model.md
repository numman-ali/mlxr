# Provider And Provenance Model

Status: provisional, but required before implementation expands beyond one narrow download flow.

## Purpose

The runtime cannot treat “download model from Hugging Face” as an implementation detail. Provider resolution, provenance, access state, and remote-code policy are part of the platform contract.

## Core Objects

### SourceRef

User-supplied reference to a source.

Example:

```json
{
  "provider": "huggingface",
  "locator": {
    "repo": "Lightricks/LTX-2.3",
    "revision": "main"
  },
  "materialization": {
    "mode": "provider-cache-ref"
  },
  "auth": {
    "token_ref": "hf-default"
  },
  "policy": {
    "allow_remote_code": false
  }
}
```

### ResolvedSource

Normalized result of provider resolution.

Must include:

- canonical provider ID
- pinned revision or digest when available
- resolved file manifest
- access state
- auth requirements
- remote-code requirement if discoverable

### ProvenanceRecord

Persistent record attached to portable artifacts and model registrations.

Must include:

- provider
- locator
- resolved revision or digest
- license
- access state
- blob digests where available
- remote-code requirement and approval state
- fetch timestamp

## Provider Adapter Responsibilities

Each provider adapter owns:

- resolution
- inspection
- auth policy
- fetch behavior
- provenance emission

It does not own family conversion or execution logic.

## v1 Provider Scope

### Implemented

- Hugging Face
- trusted local filesystem bundles

### Modeled but not yet promised

- GitHub repos or releases
- OCI registries
- object-store backed registries
- private enterprise stores

## Hugging Face Requirements

The Hugging Face provider path must support:

- revision pinning
- selective download instead of naive whole-repo copies
- gated and private access flows
- safetensors metadata preflight where possible
- explicit remote-code policy
- reuse of provider cache semantics where possible

See also [Model acquisition and cache](model-acquisition-and-cache.md) for the product-side install and cache direction layered above this provider contract.

The runtime should prefer immutable references to provider-managed cache entries rather than copying everything into a private source cache tree.

Current scaffold notes:

- the initial auth mapping supports `auth.token_ref = "hf-default"` only
- `hf-default` resolves from the local `HF_TOKEN` environment variable
- fetch materialization is a provider-cache-backed snapshot path, not a private source copy
- remote-code requirement detection is currently best-effort and records its confidence in provenance metadata

## Local Filesystem Requirements

Trusted local bundles need:

- digest capture
- explicit manifest generation
- declared license and policy fields when the source does not provide them automatically

Local paths are allowed as sources. They are not allowed as the default generic HTTP job-input contract.

## Policy Propagation

Every portable artifact must carry enough provenance to answer:

- where did this come from
- what revision was used
- what license governs it
- whether access was public, gated, or private
- whether remote code was required and approved

Hosts should not need to reach back into provider APIs to answer those questions.

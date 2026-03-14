# MLXR Mac App Runtime Contract

Status: canonical host/runtime boundary for the first-party Mac app

## Summary

The Mac app is a thin native client over the same runtime contract used by the
CLI and future adapters.

The runtime owns:

- capability truth
- planning truth
- run acceptance truth

The app owns:

- presentation
- organization
- UX language

No host should invent a second workflow-validity matrix, a second model-default
system, or a family-specific inference path.

## Canonical Runtime Seams

The Mac app, CLI, and future adapters should consume the same runtime seams:

- `/v1/capabilities`
- `/v1/models`
- `/v1/models/supported`
- `/v1/workflows/plan`
- `/v1/workflows/run`
- `/v1/jobs`
- input import routes
- output export routes

`/v1/workflows/plan` is the readiness source for draft validation.

## Planning Truth

`WorkflowPlanResult.readiness` tells hosts:

- whether the draft is runnable
- what is blocking
- which references are required
- which output formats are allowed

Capability numeric constraints remain the source of:

- width truth
- height truth
- frame-count truth
- inference-step truth
- guidance-scale truth

Hosts may shape UX around those facts, but they must not redefine them.

## Host Boundary Rules

The Mac app should:

- keep its draft in app state rather than view-local temporary state
- debounce planning through `/v1/workflows/plan`
- create visible run-group activity only after the runtime accepts a run
- present runtime-backed installs, jobs, and artifacts truthfully
- keep advanced host-side helpers optional and non-authoritative

The Mac app should not:

- own scheduling
- own inference logic
- own family-specific stage sequencing
- own capability validation truth
- create app-only model-default ranking tables that drift from the runtime

## Surface Rules

### Library

- Library is asset-first, not job-first.
- Generated outputs and imported assets share one asset model.
- Focused preview opens in a large modal viewer, not a permanent sidebar
  inspector.

### Activity

- Activity lives in the left rail footer.
- It reflects runtime-accepted work and runtime-backed installs only.
- Old completed work belongs in Library, not an eternal activity log.

### Models

- Models owns starter setup, curated installs, install queue, detail, and
  removal.
- Hugging Face is presented as the source.
- MLXR is presented as the installed home.

## Shared Implementation Principles

- no host-owned inference logic
- no host-owned scheduler logic
- no family-specific host hacks when the runtime can own the seam
- no duplicate model-default ranking tables across hosts
- no app-only validation rules that drift from runtime constraints

## Related Docs

- [Mac app product spec](./mac-app-product-spec.md)
- [Mac app runtime hardening plan](./working/mac-app-runtime-parity.md)
- [Unified studio master plan](./working/unified-studio/00-master-plan.md)

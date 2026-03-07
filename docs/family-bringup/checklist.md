# Family Bring-Up Checklist

Status: reusable checklist for new-family onboarding and major family-contract resets.

## Scope And Truthfulness

- [ ] The first supported task and profile are named plainly.
- [ ] The smallest truthful slice is documented before optional features.
- [ ] The family-specific non-goals are written down.
- [ ] Claims are downgraded if they still depend on one path, one machine, or one manual run.

## Source And Provenance

- [ ] Required source roles are explicit.
- [ ] Provider, revision, license, access state, and remote-code expectations are captured.
- [ ] Preflight inspection requirements are documented before full fetch assumptions.
- [ ] Trusted local shortcuts are separated from the generic runtime contract.

## Artifact Contract

- [ ] Portable artifact contents are explicit by role.
- [ ] Build-cache-local assumptions are not blurred into the artifact contract.
- [ ] Required metadata for load-time fail-closed validation is named.
- [ ] Output artifact kinds are truthful and user-visible.

## Workflow Split

- [ ] The core workflow layer responsibilities are named explicitly.
- [ ] Family-owned stage implementations are named explicitly.
- [ ] Host-owned responsibilities are limited to UX, imports or exports, and runtime-client mapping.
- [ ] Reusable sequencing did not drift into a host adapter.
- [ ] Family logic did not turn into a private scheduler.

## Capability And Failure Model

- [ ] Constraints and supported profiles are documented.
- [ ] Unsupported modes fail closed instead of silently defaulting.
- [ ] The capability surface matches what the runtime can actually execute.
- [ ] Failure categories are specific enough for logs, review, and host reporting.

## Validation Ladder

- [ ] There is a fast smoke rung.
- [ ] There is a meaningful semantic or quality rung.
- [ ] Logs or stage telemetry are part of the evidence.
- [ ] Promotion criteria are stated before the result is called stable.
- [ ] New platform uncertainty is recorded in `docs/research/09-open-questions-and-validation-plan.md`.

## Before Commit

- [ ] The relevant docs moved with the implementation truth.
- [ ] The local validation evidence for the claimed slice was actually checked.
- [ ] A fresh-eyes review pass looked for workflow-boundary leaks, overclaims, and missing updates.

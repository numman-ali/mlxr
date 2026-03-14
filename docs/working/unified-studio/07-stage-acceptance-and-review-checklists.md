# Stage Acceptance And Review Checklists

Status: working acceptance and review checklist

## Cross-Stage Invariants

Every stage must preserve these rules:

- runtime planning remains the only readiness authority
- the end-state has no top-level `Studio` destination
- the composer is shared, not duplicated per screen
- project means `WorkspaceRecord`
- `Activity` stays secondary
- standalone audio creation is not implied or exposed
- `swiftui-pro` review is mandatory, not optional

During migration, the internal `.studio` fallback seam may survive temporarily,
but it must stay hidden from the visible rail and must not regain a separate
prompt or submit path.

## SwiftUI-Pro Review Protocol

Before a stage is approved, review the touched files against the relevant `swiftui-pro` references:

- `views.md`
- `data.md`
- `navigation.md`
- `design.md`
- `accessibility.md`
- `performance.md`
- `swift.md`
- `hygiene.md`

The review packet for each stage must name:

- the files reviewed
- the reference categories applied
- any issues found and fixed
- any deliberate deferrals

## Stage 1: Runtime-Led Composer Contract

Accept when:

- the runtime returns a typed presentation block in `WorkflowPlanResult`
- generate enablement is derived from `WorkflowPlanResult.readiness`
- reference requirements come from planning
- packs and prompt helper do not bypass runtime truth
- submission records visible work only after runtime acceptance

Review questions:

- Did any host-owned validation table sneak back in?
- Can the host render composer state without inventing a second workflow matrix?

## Stage 2: Shell And Global Composer

Accept when:

- the rail contains `Home`, `Library`, `Models`, and `Settings`
- activity is reachable from the footer, not as a primary destination
- the composer appears on `Home` and `Library`
- the composer stays hidden on `Models` and `Settings`
- draft state survives navigation between `Home` and `Library`

Review questions:

- Does any screen still behave like a hidden full-screen `Studio` owner?
- Is the composer rendered once at the shell level instead of being copied into multiple screens?
- Are unsupported states explained using runtime readiness rather than custom heuristics?

## Stage 3: Project And Canvas

Accept when:

- the user-facing project model is backed by `WorkspaceRecord`
- results are grouped by run group inside a project
- generating from `Home` lands in project detail after acceptance
- recreate, edit, animate, and use-as-reference actions mutate the shared draft

Review questions:

- Can a user understand where their work lives without learning repo terms?
- Does project detail feel like the continuity surface for progress and history?

## Stage 4: Library Browser

Accept when:

- top-level library is project-first
- the old side filter rail is no longer primary
- project detail and project browser coexist inside one library shell
- imported assets remain immediately reusable in the composer

Review questions:

- Does the library still feel like a flat asset dump?
- Are collections still overexposed relative to their product value?

## Stage 5: Home, Models, And Activity

Accept when:

- `Home` is calm, minimal, and creation-oriented
- `Models` remains the install and management surface
- `Activity` surfaces only work and installs that the runtime actually knows about
- runtime status is legible without turning `Home` into a dashboard

Review questions:

- Is `Home` trying to become a canvas again?
- Is old completed work correctly discoverable in `Library` instead of being stranded in `Activity`?

## Stage 6: Architecture And Hardening

Accept when:

- shell, composer, project, and library responsibilities are cleanly separated by package
- no second persistence model was introduced
- app-model tests and Swift package tests cover the new routing and project behavior
- repo verification still passes after the redesign lands

Review questions:

- Did the redesign make state ownership clearer?
- Did any view-local workaround start owning workflow or persistence logic?

## Manual Review Flows

Run these end-to-end before calling the redesign complete:

1. Fresh start with no models installed.
2. Install a starter model and create the first project from `Home`.
3. Open that project in `Library`, generate again, and confirm history stays grouped.
4. Use a result action to switch from image generation to video animation.
5. Import a source asset into a project and reuse it immediately.
6. Trigger a running job and confirm `Activity`, composer status, and project detail stay in sync.
7. Disconnect or kill the runtime and confirm shell messaging stays truthful.

## Required Validation Lanes

Every stage must pass:

- `swift test --package-path packages/clients/mlxr-mac-app`
- targeted Python/runtime tests when shared runtime contracts changed
- `uv run python scripts/dev.py verify`
- staged dev `.app` validation with Peekaboo
- runtime log inspection when creation or install behavior changed

## Final Sign-Off

The redesign is ready for implementation or promotion only when:

- the staged docs still agree with `visual-brief.md` except where
  `00-master-plan.md` explicitly closes or overrides a design question
- the staged docs still agree with `../mac-app-runtime-parity.md`
- open questions have been turned into decisions or clearly deferred work

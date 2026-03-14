# Unified Studio Master Plan

Status: approved redesign meta-plan and implementation governance for the unified studio tranche

Primary visual brief:
- `docs/working/unified-studio/visual-brief.md`

Primary implementation spec:
- this file plus the numbered stage plans in `docs/working/unified-studio/`

Supporting product and architecture anchors:
- `docs/working/mac-app-runtime-parity.md`
- `docs/mac-app-product-spec.md`
- `docs/mac-app-runtime-contract.md`

## Summary

Rebuild the Mac app into the Hailuo-inspired unified studio without violating the runtime-parity rules already hardened in the repo.

The governing rule is fixed:

- redesign the surface aggressively
- keep planning, capability, and run-acceptance truth in the runtime
- use `swiftui-pro` as a mandatory engineering and review gate for every stage

This redesign is executed as a staged plan set under
`docs/working/unified-studio/`, with each stage carrying its own acceptance
criteria, Peekaboo validation flow, runtime-log checks, and `swiftui-pro`
checklist.

## Outcome

`MLXR` becomes one runtime-led creative shell:

- `Home` orients the user and gets them creating quickly.
- `Library` is the real work surface: project browser at the top level, project canvas when a project is open.
- `Models` and `Settings` stay management surfaces.
- The composer is global across creation surfaces instead of being trapped inside a `Studio` destination.
- `Activity` stays secondary and truthful as a rail-footer overlay, not a primary destination.

This keeps the Hailuo-style "one app, one prompt bar, one body of work" feel from the screen brief while staying aligned with the runtime-parity rules already adopted in the repo.

## Locked Decisions

- Remove the top-level `Studio` destination in the end-state. A temporary visible `Studio` entry may remain during migration, but it is fallback-only and must not gain new primary creation behaviors.
- Keep the primary composer switcher to `Video` and `Image`.
- Do not expose standalone `Audio` mode yet. `MLXR` supports audio-conditioned video, not standalone audio creation, so audio remains a video sub-workflow instead of a third top-level segment.
- Treat `project` as the user-facing name for `WorkspaceRecord`.
- Keep `RunGroupRecord` as the iteration unit inside a project.
- Keep `CollectionRecord` as secondary organization only; it does not drive the primary browser or project model.
- Keep runtime workflow planning as the only source of readiness, reference requirements, and allowed output formats.
- Keep prompt enhancement host-side. The helper may suggest or prefill text, but it never changes capability truth or creates a second prompt contract.
- The end-state visible information architecture is:
  - `Home`
  - `Library`
  - `Models`
  - `Settings`
  - `Activity` in the rail footer
- Show the composer on `Home` and `Library` only.
- Hide the composer on `Models` and `Settings`.
- Do not add a dedicated right-edge filmstrip in the first redesign pass. The project canvas grid and hero preview are enough.

## Resolved Differences From The Screen Brief

The screen brief is still the visual and interaction anchor, but these details are now closed:

- `Activity` uses the rail footer overlay shape from the current app and parity plan instead of becoming a full rail destination.
- `Project` maps to `WorkspaceRecord`, not `CollectionRecord`.
- `Audio` is omitted from the primary segmented control until the runtime has a real standalone audio row.
- The full results canvas lives inside project detail in `Library`.
- Creating from a project detail appends to that project.
- Creating from the top-level project browser creates a new project on runtime acceptance unless the user explicitly resumed an existing project context.
- Creating from `Home` creates a new project by default and only reuses an existing one when the user explicitly resumes it.

## User-Facing Flow

1. App boots into one of three states:
   - no installed starter model: `Models` setup
   - installed models but no meaningful work: `Home`
   - existing work: `Library` focused on the active or most recent project
2. The composer is always available on `Home` and `Library`.
3. Submitting from `Home` creates a new project by default, unless the user explicitly resumed an existing project.
4. Submitting from the top-level `Library` browser creates a new project after runtime acceptance.
5. Submitting from `Library` project detail keeps the user in place and appends work into the current project.
6. Results, references, iteration actions, and project history stay together.

## Stage Set

The redesign doc tranche is split into these docs:

1. `01-shell-and-global-composer-plan.md`
2. `02-runtime-led-composer-contract-plan.md`
3. `03-project-and-canvas-plan.md`
4. `04-library-browser-plan.md`
5. `05-home-models-activity-plan.md`
6. `06-implementation-architecture-plan.md`
7. `07-stage-acceptance-and-review-checklists.md`
8. `08-implementation-todo-hub.md`

The docs are ordered for readability, not for implementation order.

The implementation order is fixed:

1. governance and seam prep from `00`, `06`, `07`, and `08`
2. runtime-led composer contract in `02`
3. shell-level global composer in `01`
4. project and canvas work in `03`
5. project-first library browser in `04`
6. home, models, and activity alignment in `05`
7. cutover and cleanup using `06` plus the final checks in `07`

## SwiftUI-Pro Hard Gate

`swiftui-pro` is mandatory for this redesign. It is not advisory.

Before any stage is called done, review the touched files against the relevant `swiftui-pro` references:

- `views.md`
- `data.md`
- `navigation.md`
- `design.md`
- `accessibility.md`
- `performance.md`
- `swift.md`
- `hygiene.md`

Each stage doc in this folder must name the exact files and the exact `swiftui-pro` references to review for that stage.

## Non-Goals

- No host-owned workflow legality tables.
- No host-owned inference or scheduler logic.
- No new persistent database model for projects.
- No standalone audio creation UX.
- No second inspector pane or workflow sidebar.
- No separate "pro mode" surface; advanced controls stay behind the composer tune popover and model-specific packs.

## Done Definition

This redesign is only complete when all of these are true together:

- `Studio` is gone from the primary navigation.
- `Home` and `Library` share one global composer.
- Project detail is the continuity surface for progress and results.
- The app still plans and submits through the same runtime seams used today.
- Project, library, model install, and activity flows all remain truthful to runtime state.

## Cross-Stage Validation Gate

Every implementation stage must pass all of these before merge:

- `swift test --package-path packages/clients/mlxr-mac-app`
- targeted Python/runtime tests when runtime contracts changed
- `uv run python scripts/dev.py verify`
- staged dev `.app` validation with Peekaboo
- runtime log inspection for creation and install changes
- explicit `swiftui-pro` review notes for the touched files

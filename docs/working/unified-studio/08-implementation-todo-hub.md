# Unified Studio Implementation Todo Hub

Status: working implementation checklist for the unified studio redesign

Use this as the execution hub for implementation, review, and regression checking.

Primary plan sources:

- `docs/working/unified-studio/00-master-plan.md`
- `docs/working/unified-studio/01-shell-and-global-composer-plan.md`
- `docs/working/unified-studio/02-runtime-led-composer-contract-plan.md`
- `docs/working/unified-studio/03-project-and-canvas-plan.md`
- `docs/working/unified-studio/04-library-browser-plan.md`
- `docs/working/unified-studio/05-home-models-activity-plan.md`
- `docs/working/unified-studio/06-implementation-architecture-plan.md`
- `docs/working/unified-studio/07-stage-acceptance-and-review-checklists.md`
- `docs/working/unified-studio/09-performance-and-responsiveness-plan.md`

Review gate:

- `swiftui-pro` is mandatory for every stage.

## Stage 0: Spec Lock And Architecture Prep

- [x] Confirm `docs/working/unified-studio/visual-brief.md` still matches the intended north-star UI where it remains non-normative inspiration.
- [x] Confirm all eight reference frames still support the chosen IA and interaction model.
- [x] Keep `00-master-plan.md` aligned with the latest locked decisions.
- [x] Keep `07-stage-acceptance-and-review-checklists.md` aligned with the validation and review gate.
- [x] Keep this todo hub updated when stage scope changes.
- [x] Preserve runtime-parity rules: no host-owned readiness, capability, or run-acceptance truth.
- [x] Preserve the current runtime bridge and app model as the integration backbone.
- [x] Decide the exact migration boundary for temporary `Studio` support.
- [x] Identify which current files are transitional and should be deleted only after cutover.
- [x] Identify which current feature seams are reusable and should be preserved.
- [x] Keep `swiftui-pro` review categories named for each stage.
- [x] Confirm no stage requires a Mac-only protocol or second workflow endpoint.
- [x] Lock the implementation order separately from the doc file order.
- [x] Lock `WorkspaceRecord` as the project backing type and demote `CollectionRecord` to secondary organization.
- [x] Lock creation behavior so the explicit top-level `New Project` action opens a blank project immediately, top-level browser submission without an open project creates one on acceptance, and project detail appends to the open project.
- [x] Mark `visual-brief.md` as the visual brief and the stage docs as the normative implementation spec.

## Stage 1: Runtime-Led Composer Contract

- [x] Add the shared `presentation` block to `WorkflowPlanResult`.
- [x] Keep the new planning payload typed in shared schemas rather than burying it in metadata.
- [x] Export the new presentation types from the shared schema package.
- [x] Extend `FamilyWorkflowStrategy` with `presentation(...)`.
- [x] Have `WorkflowPlanner.plan_for_model()` include `presentation` in the returned result.
- [x] Add reusable helper builders for presentation controls, subworkflows, and slots.
- [x] Keep `WorkflowPlanReadiness` as the only source of blocking issues and reference requirements.
- [x] Avoid duplicating numeric constraint truth into the host.
- [x] Ensure the Swift-side execution schemas mirror the new planning payload cleanly.
- [x] Add explicit presentation coverage for Z-Image workflows.
- [x] Add explicit presentation coverage for FLUX.2 workflows.
- [x] Add explicit presentation coverage for Qwen workflows.
- [x] Add explicit presentation coverage for LTX workflows.
- [x] Add planner-level tests proving presentation is returned by the planner.
- [x] Add shared-schema tests for presentation serialization and strictness.
- [x] Patch runtime-server fake planner tests so they return an explicit presentation payload.
- [x] Update runtime workflow endpoint tests to assert returned presentation values.
- [x] Fix all typing issues introduced by the new presentation helpers.
- [x] Confirm CLI fixtures still decode or construct `WorkflowPlanResult` cleanly.

## Stage 2: Global Composer Shell

- [x] Mount one shell-level bottom composer in `RootView`.
- [x] Show the composer on `Home`.
- [x] Show the composer on `Library`.
- [x] Decide whether the temporary `Studio` seam should also show the shared composer during migration.
- [x] Hide the composer on `Models`.
- [x] Hide the composer on `Settings`.
- [x] Hide the composer while bootstrap or starter-setup overlays are blocking the shell.
- [x] Add shell-level planning scheduling so draft changes re-plan even outside `Studio`.
- [x] Remove dead global-composer code paths once the shell actually owns the composer.
- [x] Ensure submission and disabled-state logic have one source of truth.
- [x] Remove duplicate submit logic from `StudioScreen`.
- [x] Remove duplicate readiness and disabled-state logic from `StudioScreen`.
- [x] Move the visible prompt entry point out of `StudioCanvasView`.
- [x] Remove the old prompt card from the canvas once the shared composer is live.
- [x] Add a modality switcher to the composer using runtime-presented options.
- [x] Add a sub-workflow selector using runtime-presented options.
- [x] Add a visible runtime status label to the composer.
- [x] Add a model pill to the composer.
- [x] Add quality, aspect, and duration or variation pills to the composer.
- [x] Decide that the compact composer pills remain the surfaced advanced controls for this tranche; no separate tune popover is needed yet.
- [x] Decide how reference management is exposed during the temporary migration phase.
- [x] Keep prompt helper behavior explicit and host-side only.
- [x] Ensure there is exactly one visible creation system after the slice lands.
- [x] Collapse the composer into a small floating command pill instead of a layout-reserving footer slab.
- [x] Add a true hidden/resting composer state instead of forcing only collapsed versus expanded.
- [x] Keep `Home` submission in place until runtime acceptance, then route into `Library`.

## Stage 3: Project Model And Canvas

- [x] Decide the exact user-facing naming for projects versus workspaces in the UI.
- [x] Keep `WorkspaceRecord` as the durable project backing type.
- [x] Add any thin presentation type needed for project summaries.
- [x] Auto-create or rename the default project on first accepted run.
- [x] Preserve run groups as the per-iteration history unit inside a project.
- [x] Attach accepted run groups to the current project only after runtime acceptance.
- [x] Build a proper project header surface.
- [x] Build an empty project canvas state.
- [x] Build an inline generating state in the project canvas.
- [x] Build a grouped results state in the project canvas.
- [x] Add focused-asset selection plus modal viewer entry from the project canvas.
- [x] Add grouped thumbnail browsing inside the canvas.
- [x] Add viewer-driven actions for `Edit`, `Animate`, `Guide`, `Retake`, and `Use as reference`.
- [x] Add `Reveal in Finder` and imported-asset removal where truthful and safe.
- [x] Keep seeded-action rules strict: actions seed the draft, re-plan through the runtime, and wait for explicit submit.
- [x] Make failed planning stay inline instead of appearing as accepted queued work.
- [x] Keep only accepted runtime work visible in project activity and history.
- [x] Ensure switching prompts or modes does not destroy project continuity.

## Stage 4: Library As Project Browser

- [x] Remove the old filter-rail-first layout from the main library path.
- [x] Build a project-first top-level library browser.
- [x] Build project cards with hero thumbnail, subtitle, quick counts, and active badge.
- [x] Keep search at the top of the library browser.
- [x] Keep light media filter chips at the top of the library browser.
- [x] Keep import as a top-level library action.
- [x] Keep large-preview viewing in a modal rather than a permanent side inspector.
- [x] Preserve keyboard navigation for the viewer and browser.
- [x] Keep uniform thumbnail sizing and cropping rules.
- [x] Ensure thumbnailing is lazy and cached.
- [x] Ensure video playback is not instantiated in grid tiles.
- [x] Keep imported assets immediately reusable in the composer.
- [x] Attach top-level imports without forcing a visible project jump before import succeeds.
- [x] Prevent empty or failed top-level imports from materializing a project before assets exist.
- [x] Keep favorites lightweight and secondary.
- [x] Keep collections as secondary data, not primary navigation.
- [x] Ensure project detail and project browser share the same overall library shell.
- [x] Ensure search can match project titles, prompt headlines, model names, and useful filenames.

## Stage 5: Home, Models, And Activity

- [x] Simplify `Home` into a launchpad rather than a dashboard.
- [x] Remove heavy explanatory cards that repeat what the UI should make obvious.
- [x] Keep `Home` focused on starter actions, recent projects, and starter-model setup.
- [x] Keep the composer visible on `Home` when it is actually usable.
- [x] Hide or disable the composer on `Home` when no runnable model exists yet.
- [x] Keep `Models` as the install and removal surface.
- [x] Preserve first-run starter-model setup behavior in `Models`.
- [x] Keep model install phases explicit and truthful.
- [x] Keep `Activity` in the rail footer.
- [x] Ensure `Activity` shows only runtime-accepted running or queued work plus install operations.
- [x] Keep failures sticky until dismissed.
- [x] Keep old successful work in `Library`, not `Activity`.
- [x] Mirror runtime/install state qualitatively in the composer.
- [x] Keep runtime status calm and legible rather than metric-heavy.

## Stage 6: Cutover And Cleanup

- [x] Remove visible `Studio` from the nav and remove the hidden fallback route from the shell.
- [x] Retire `StudioScreen` when its responsibilities are fully absorbed elsewhere.
- [x] Retire `StudioSidebarView` when workflow picking and source management move to the new surfaces.
- [x] Retire `StudioInspectorView` when primary creation controls live in the composer pill row.
- [x] Delete old workflow-list concepts from the visible product surface.
- [x] Delete the old library filter rail once project-first browsing is stable.
- [x] Remove dead code paths left by the temporary migration seam.
- [x] Keep one clear creation story across the whole app.
- [x] Update the product docs and current-status docs to match the new visible IA.

## SwiftUI-Pro Hard Checks

- [x] Review shell files against `views.md`.
- [x] Review shell files against `data.md`.
- [x] Review composer files against `views.md`.
- [x] Review composer files against `data.md`.
- [x] Review composer files against `performance.md`.
- [x] Review project-canvas files against `design.md`.
- [x] Review project-canvas files against `accessibility.md`.
- [x] Review library files against `navigation.md`.
- [x] Review library files against `performance.md`.
- [x] Review library files against `accessibility.md`.
- [x] Review app-model and shared-state files against `swift.md`.
- [x] Review all touched files against `hygiene.md`.

## Validation And Regression Checklist

These checks cover the shipped shell, viewer, import safeguards, planning
behavior, and the project-detail generation loop of this compact tranche. The
remaining first-run, Home-entry, and import-specific journeys below remain the
separate sign-off list for follow-up live validation.

- [x] `swift test --package-path packages/clients/mlxr-mac-app`
- [x] targeted Python/runtime tests for changed shared contracts
- [x] `uv run python scripts/dev.py verify`
- [x] staged dev `.app` run through `uv run python scripts/dev.py mac-app`
- [x] Peekaboo validation on the staged app
- [x] runtime log inspection after creation/install behavior changes
- [x] confirm idle app does not spam the runtime
- [x] confirm no duplicate daemon spawning
- [x] confirm failed planning does not create ghost activity items
- [x] confirm only runtime-accepted work appears in `Activity`

## Required Live End-To-End Flows

- [x] first launch with no starter models installed
- [x] install a starter model from `Models`
- [x] create from `Home` using the global composer
- [x] create from `Library` or project detail using the same composer
- [x] switch between image and video without losing the draft
- [x] use a generated asset as a reference
- [x] open and close the modal viewer with mouse and keyboard
- [x] confirm grouped results remain attached to the current project
- [x] confirm imported assets land in the right project and are immediately reusable

## Sign-Off Rule

Do not call the redesign done until:

- the implementation matches the current unified-studio working docs
- the runtime remains the only truth source for readiness and run acceptance
- the SwiftUI-pro review is recorded for the touched stage
- validation passes locally
- the live app behaves correctly under Peekaboo and log inspection

## Performance Tranche

- [x] Keep `09-performance-and-responsiveness-plan.md` aligned with the latest live findings.
- [x] Keep local presentation-state persistence off the UI hot path.
- [x] Keep shell-level derived state cached rather than rebuilt on routine screen switches.
- [x] Keep Library and project presentation work out of the hottest render loops.
- [x] Keep viewer image decode asynchronous and lightweight.
- [x] Keep thumbnail generation cancellation-safe and bounded to visible work.
- [x] Keep `Models` entry cheap by limiting eager preview work.
- [x] Re-validate route switching and asset-open latency in the staged app after each tranche.

## Current Tranche Review Notes

- `swiftui-pro` categories applied during the final review:
  - shell: `RootView`, `PageScaffolding`, `AppModel+Gallery`, `AppModel+Workspace` against `views`, `data`, `swift`, and `hygiene`
  - composer: `GlobalComposerBar` and shell overlay ownership against `views`, `data`, and `performance`
  - library/project surfaces: `LibraryWorkspaceView`, `ProjectBrowserView`, `LibraryGridView`, `LibraryViewerSheet`, `ProjectHeaderView` against `navigation`, `design`, `performance`, and `accessibility`
- Local Peekaboo review during this tranche covered:
  - isolated first-run launch against an empty runtime home
  - starter-model queueing from the first-run and `Models` surface
  - `Home` draft entry, mode switching, and acceptance-driven routing into `Library`
  - top-level project browser
  - new blank-project creation from the top-level browser
  - project detail grid
  - project detail pending-run strip and inline state badges
  - single-click asset viewer
  - viewer-to-composer `Edit` flow
  - viewer-to-composer `Use as reference` flow on both generated and imported images
  - real image import into an existing project and immediate reuse in the composer
  - project-detail submit flow keeping the accepted run attached to the same project
  - top-level browser keyboard `Return` reopening the selected project
- Local validation receipts captured during the tranche include:
  - the `Home` composer prompt before acceptance
  - `Home` submission routing into `Library` after runtime acceptance
  - the recovered real-runtime project browser after install cleanup
  - isolated first-run starter-model setup against an empty runtime home
  - isolated starter-model queue and download state
  - real image import opening directly in the viewer
  - imported-image reuse seeding the shared composer
- Known intentional deferrals for later polish, not blockers for this compact tranche:
  - dedicated advanced-settings drawer beyond the current pill row
  - explicit inline pending-result placeholder inside the project grid
  - a dedicated `Re-create` button distinct from the current seed-and-replan action family

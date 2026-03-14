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

Review gate:

- `swiftui-pro` is mandatory for every stage.

## Stage 0: Spec Lock And Architecture Prep

- [ ] Confirm `docs/working/unified-studio/visual-brief.md` still matches the intended north-star UI.
- [ ] Confirm all eight reference frames still support the chosen IA and interaction model.
- [ ] Keep `00-unified-studio-master-plan.md` aligned with the latest locked decisions.
- [ ] Keep `07-stage-acceptance-and-review-checklists.md` aligned with the validation and review gate.
- [ ] Keep this todo hub updated when stage scope changes.
- [ ] Preserve runtime-parity rules: no host-owned readiness, capability, or run-acceptance truth.
- [ ] Preserve the current runtime bridge and app model as the integration backbone.
- [x] Decide the exact migration boundary for temporary `Studio` support.
- [ ] Identify which current files are transitional and should be deleted only after cutover.
- [ ] Identify which current feature seams are reusable and should be preserved.
- [ ] Keep `swiftui-pro` review categories named for each stage.
- [ ] Confirm no stage requires a Mac-only protocol or second workflow endpoint.
- [x] Lock the implementation order separately from the doc file order.
- [x] Lock `WorkspaceRecord` as the project backing type and demote `CollectionRecord` to secondary organization.
- [x] Lock creation behavior so top-level project browsing creates a new project on acceptance, while project detail appends to the open project.
- [x] Mark `unified-studio-screens.md` as the visual brief and the stage docs as the normative implementation spec.

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
- [ ] Confirm CLI fixtures still decode or construct `WorkflowPlanResult` cleanly.

## Stage 2: Global Composer Shell

- [x] Mount one shell-level bottom composer in `RootView`.
- [x] Show the composer on `Home`.
- [x] Show the composer on `Library`.
- [x] Decide whether the temporary `Studio` seam should also show the shared composer during migration.
- [x] Hide the composer on `Models`.
- [x] Hide the composer on `Settings`.
- [x] Hide the composer while bootstrap or starter-setup overlays are blocking the shell.
- [x] Add shell-level planning scheduling so draft changes re-plan even outside `Studio`.
- [ ] Remove dead global-composer code paths once the shell actually owns the composer.
- [x] Ensure submission and disabled-state logic have one source of truth.
- [x] Remove duplicate submit logic from `StudioScreen`.
- [x] Remove duplicate readiness and disabled-state logic from `StudioScreen`.
- [x] Move the visible prompt entry point out of `StudioCanvasView`.
- [x] Remove the old prompt card from the canvas once the shared composer is live.
- [ ] Add a modality switcher to the composer using runtime-presented options.
- [ ] Add a sub-workflow selector using runtime-presented options.
- [x] Add a visible runtime status label to the composer.
- [x] Add a model pill to the composer.
- [x] Add quality, aspect, and duration or variation pills to the composer.
- [ ] Add the tune-popover trigger to the composer.
- [ ] Decide how reference management is exposed during the temporary migration phase.
- [ ] Keep prompt helper behavior explicit and host-side only.
- [ ] Ensure there is exactly one visible creation system after the slice lands.

## Stage 3: Project Model And Canvas

- [ ] Decide the exact user-facing naming for projects versus workspaces in the UI.
- [ ] Keep `WorkspaceRecord` as the durable project backing type.
- [ ] Add any thin presentation type needed for project summaries.
- [ ] Auto-create or rename the default project on first accepted run.
- [ ] Preserve run groups as the per-iteration history unit inside a project.
- [ ] Attach accepted run groups to the current project only after runtime acceptance.
- [ ] Build a proper project header surface.
- [ ] Build an empty project canvas state.
- [ ] Build an inline generating state in the project canvas.
- [ ] Build a grouped results state in the project canvas.
- [ ] Add hero selection behavior inside the canvas.
- [ ] Add grouped thumbnail browsing inside the canvas.
- [ ] Add inline actions for `Re-create`, `Edit`, `Animate`, and `Use as reference`.
- [ ] Add `Reveal in Finder` and `Delete` where truthful and safe.
- [ ] Keep recreate rules strict: exact reruns can auto-submit, others must re-plan first.
- [ ] Make failed planning stay inline instead of appearing as accepted queued work.
- [ ] Keep only accepted runtime work visible in project activity and history.
- [ ] Ensure switching prompts or modes does not destroy project continuity.

## Stage 4: Library As Project Browser

- [ ] Remove the old filter-rail-first layout from the main library path.
- [ ] Build a project-first top-level library browser.
- [ ] Build project cards with hero thumbnail, updated time, and quick counts.
- [ ] Keep search at the top of the library browser.
- [ ] Keep light media filter chips at the top of the library browser.
- [ ] Keep import as a top-level library action.
- [ ] Keep large-preview viewing in a modal rather than a permanent side inspector.
- [ ] Preserve keyboard navigation for the viewer and browser.
- [ ] Keep uniform thumbnail sizing and cropping rules.
- [ ] Ensure thumbnailing is lazy and cached.
- [ ] Ensure video playback is not instantiated in grid tiles.
- [ ] Keep imported assets immediately reusable in the composer.
- [ ] Attach imports to the active project or explicitly choose a target project.
- [ ] Keep favorites lightweight and secondary.
- [ ] Keep collections as secondary data, not primary navigation.
- [ ] Ensure project detail and project browser share the same overall library shell.
- [ ] Ensure search can match project titles, prompt headlines, model names, and useful filenames.

## Stage 5: Home, Models, And Activity

- [ ] Simplify `Home` into a launchpad rather than a dashboard.
- [ ] Remove heavy explanatory cards that repeat what the UI should make obvious.
- [ ] Keep `Home` focused on starter actions, recent projects, and starter-model setup.
- [ ] Keep the composer visible on `Home` when it is actually usable.
- [ ] Hide or disable the composer on `Home` when no runnable model exists yet.
- [ ] Keep `Models` as the install and removal surface.
- [ ] Preserve first-run starter-model setup behavior in `Models`.
- [ ] Keep model install phases explicit and truthful.
- [ ] Keep `Activity` in the rail footer.
- [ ] Ensure `Activity` shows only runtime-accepted running or queued work plus install operations.
- [ ] Keep failures sticky until dismissed.
- [ ] Keep old successful work in `Library`, not `Activity`.
- [ ] Mirror runtime/install state qualitatively in the composer.
- [ ] Keep runtime status calm and legible rather than metric-heavy.

## Stage 6: Cutover And Cleanup

- [ ] Remove visible `Studio` from the nav when `Home` + `Library` + global composer are fully proven.
- [ ] Retire `StudioScreen` when its responsibilities are fully absorbed elsewhere.
- [ ] Retire `StudioSidebarView` when workflow picking and source management move to the new surfaces.
- [ ] Retire `StudioInspectorView` when primary creation controls live in the composer and tune popover.
- [ ] Delete old workflow-list concepts from the visible product surface.
- [ ] Delete the old library filter rail once project-first browsing is stable.
- [ ] Remove dead code paths left by the temporary migration seam.
- [ ] Keep one clear creation story across the whole app.
- [ ] Update the product docs and current-status docs to match the new visible IA.

## SwiftUI-Pro Hard Checks

- [ ] Review shell files against `views.md`.
- [ ] Review shell files against `data.md`.
- [ ] Review composer files against `views.md`.
- [ ] Review composer files against `data.md`.
- [ ] Review composer files against `performance.md`.
- [ ] Review project-canvas files against `design.md`.
- [ ] Review project-canvas files against `accessibility.md`.
- [ ] Review library files against `navigation.md`.
- [ ] Review library files against `performance.md`.
- [ ] Review library files against `accessibility.md`.
- [ ] Review app-model and shared-state files against `swift.md`.
- [ ] Review all touched files against `hygiene.md`.

## Validation And Regression Checklist

- [ ] `swift test --package-path packages/clients/mlxr-mac-app`
- [ ] targeted Python/runtime tests for changed shared contracts
- [ ] `uv run python scripts/dev.py verify`
- [ ] staged dev `.app` run through `uv run python scripts/dev.py mac-app`
- [ ] Peekaboo validation on the staged app
- [ ] runtime log inspection after creation/install behavior changes
- [ ] confirm idle app does not spam the runtime
- [ ] confirm no duplicate daemon spawning
- [ ] confirm failed planning does not create ghost activity items
- [ ] confirm only runtime-accepted work appears in `Activity`

## Required Live End-To-End Flows

- [ ] first launch with no starter models installed
- [ ] install a starter model from `Models`
- [ ] create from `Home` using the global composer
- [ ] create from `Library` or project detail using the same composer
- [ ] switch between image and video without losing the draft
- [ ] use a generated asset as a reference
- [ ] open and close the modal viewer with mouse and keyboard
- [ ] confirm grouped results remain attached to the current project
- [ ] confirm imported assets land in the right project and are immediately reusable

## Sign-Off Rule

Do not call the redesign done until:

- the implementation matches the design-thinking docs
- the runtime remains the only truth source for readiness and run acceptance
- the SwiftUI-pro review is recorded for the touched stage
- validation passes locally
- the live app behaves correctly under Peekaboo and log inspection

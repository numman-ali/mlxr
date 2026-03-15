# Unified Studio Performance And Responsiveness Plan

Status: active performance tranche with hot-path relief landed; lightweight instrumentation remains the main follow-up

This plan covers the Mac app slowdown seen when:

- switching between Home, Library, Models, and Settings
- entering or leaving a project
- opening an asset in the viewer

The runtime is not the primary suspect here. The current evidence points to app-side UI and local-state work.

## Core Diagnosis

The slowdown has four main causes:

1. Local presentation state is still persisted too eagerly relative to user interaction.
2. Shell-level derived state is still expensive enough to make route changes heavier than they should be.
3. Library and project presentation work still sits too close to the view layer.
4. Viewer image loading has been too expensive in the presentation path.

The intended ownership split remains:

- the runtime owns jobs, installs, capabilities, and output truth
- the app owns local project organization, selection state, recents, favorites, and draft persistence

The fix is not to move more into the runtime. The fix is to make app-owned state cheaper, more explicit, and less synchronous.

## Tranche Order

### Tranche 1: Hot-Path Relief

- move presentation-state persistence fully off the UI hot path
- keep persistence debounced and coalesced
- cache shell-level derived state in `MLXRAppModel`
- move viewer image decode off the render path
- confirm the app switches faster before deeper refactors

### Tranche 2: Library And Project Projection

- keep `LibraryPresentationModel` as the place where grouping and filtering happen
- make its construction explicit and stable rather than repeatedly implicit from a computed property
- remove duplicated helper logic that can drift
- keep project selection, viewer selection, and visible-id pruning cheap

### Tranche 3: Thumbnail And Materialization Pipeline

- make thumbnail generation cancellation-safe
- ensure visible-grid thumbnail work is bounded
- add lightweight resolved-URL reuse so the same visible assets are not repeatedly re-materialized
- keep thumbnail generation off the main actor for heavy decode work

### Tranche 4: Models Entry Cost

- stop eager preview fan-out when entering `Models`
- only prefetch the small visible set needed for the first screenful
- keep model-detail loading on demand

### Tranche 5: Instrumentation And Guardrails

- add lightweight timing or signpost instrumentation around route switch, project open, asset open, and state persistence
- validate each tranche with the staged app and Peekaboo
- keep `swiftui-pro` review as the standing performance/style gate

## Concrete To-Do List

### Shell And State

- [x] Debounce and coalesce presentation-state persistence.
- [x] Ensure presentation-state persistence never blocks route changes on the main actor.
- [x] Cache `libraryEntries`, `libraryAssets`, and recent-library slices in `MLXRAppModel`.
- [x] Cache activity-run-group projections instead of rebuilding them from jobs inside hot render paths.
- [x] Make root-shell consumers read cached values rather than triggering fresh rebuilds.
- [x] Re-check whether `workspaces` and workspace-selection changes invalidate all caches correctly.

### Library And Project Detail

- [x] Remove any remaining `LibraryPresentationModel` regressions from the in-progress optimization patch.
- [x] Ensure project-summary lookup still works even when top-level filters hide a project from the browser list.
- [x] Ensure group ordering for `Newest` uses the newest relevant timestamp, not the oldest one in a set.
- [x] Keep `LibraryWorkspaceView` from reconstructing presentation work more often than necessary.
- [x] Prefer stable view models or snapshots over repeated computed-property reconstruction when practical.

### Viewer

- [x] Keep viewer presentation immediate and lightweight.
- [x] Load large image previews asynchronously.
- [x] Downsample viewer images before decode.
- [x] Compute viewer dominant hue off the render path.
- [x] Ensure viewer close/reopen does not leave stale preview work running.

### Thumbnail Pipeline

- [x] Make thumbnail generation cancellation-safe.
- [x] Keep thumbnail decode off the main actor for heavy work.
- [x] Avoid spawning more work than the visible grid needs.
- [x] Add small reuse for repeated asset materialization where the same visible assets are requested often.

### Models

- [x] Limit preview prefetch to the first useful set of visible rows.
- [x] Keep `Models` entry cheap even when many installable rows exist.
- [x] Use lazy detail loading only when the detail surface opens.

### Validation

- [x] `swift test --package-path packages/clients/mlxr-mac-app`
- [x] `uv run python scripts/dev.py verify`
- [x] `uv run python scripts/dev.py mac-app --quit-existing`
- [x] live route switching checks with Peekaboo
- [x] live project-entry checks with Peekaboo
- [x] live asset-open checks with Peekaboo
- [x] runtime log inspection after each tranche

## Remaining Follow-Up

- Add lightweight signpost or timing instrumentation around route switching, project open, and viewer open so future regressions are measured rather than felt anecdotally.

## Acceptance Criteria

This tranche is not done until all of the following are true:

- Library to Models switching feels immediate rather than visibly sticky.
- Entering and leaving a project no longer pauses on route change.
- Opening an image feels like an instant viewer open followed by preview fill, not a blocked click.
- The app does not regress project selection, filter behavior, or viewer navigation correctness.
- Thumbnail work cancels cleanly when the relevant surface disappears.
- The runtime remains the source of job/install/capability truth, while the app remains the owner of local presentation state.

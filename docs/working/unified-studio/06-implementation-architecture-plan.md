# Implementation Architecture Plan

Status: working implementation architecture plan

## Purpose

Map the redesign onto the current Swift packages and app model without inventing a second architecture.

## Core Principle

Favor re-composition over reinvention.

The redesign should mostly:

- move ownership upward into `RootView` and `MLXRAppModel`
- split the current `StudioScreen` into reusable composer and canvas pieces
- refactor `GalleryScreen` into project browser plus project detail

The redesign should not:

- replace runtime planning
- create a second persistence model
- push workflow logic into view-local helpers

## Package-Level Responsibilities

### `MLXRAppShell`

Own:

- top-level destination routing
- active project selection
- global composer placement
- activity overlay presentation
- launch routing and restoration

Primary changes:

- keep `Destination.studio` as a temporary fallback seam only during migration
- introduce shell state for active project detail
- host one shared composer overlay

### `MLXRFeatureCreate`

Refactor the current create surface into reusable pieces:

- `GlobalComposerView`
- `ComposerReferenceStrip`
- `ComposerSettingsRow`
- `TunePopover`
- action helpers that mutate the shared draft

This package should stop owning a full-screen destination.

### `MLXRFeatureGallery`

Own the library body in two states:

- `ProjectBrowserView`
- `ProjectDetailCanvasView`

Keep and reuse:

- thumbnail caching
- modal viewer
- asset materialization helpers

Remove the old permanent filter rail from the default layout.

### `MLXRFeatureHome`

Simplify the home surface around:

- onboarding or setup state
- recent projects
- quick actions
- shared composer host slot

### `MLXRFeatureToolkit`

Keep model management largely as-is, with incremental polish rather than redesign.

### `MLXRFeatureSettings`

Keep runtime diagnostics, prompt-helper settings, and advanced import tools. No composer.

### `MLXRActivityStrip`

Keep the overlay model. Adjust copy and grouping only as needed for the new shell.

## App Model Changes

`MLXRAppModel` remains the central state owner.

Add or clarify responsibilities for:

- active project creation and switching
- project summaries derived from `WorkspaceRecord` and `RunGroupRecord`
- project auto-titling from the first accepted prompt
- import-to-project behavior
- run-group actions such as recreate, delete, and reopen in composer
- routing helpers that move from `Home` into project detail after acceptance

Do not move planning or readiness logic out of the app model into views.

## Domain Model Decisions

Preserve:

- `StudioWorkspaceDraft` for the editable draft
- `WorkspaceRecord` for project persistence
- `RunGroupRecord` for grouped outputs
- `CollectionRecord` as secondary metadata only

Add only thin presentation types where needed, such as:

- project summary view models
- project detail presentation models
- user-facing sub-workflow labels

Do not add a new durable project entity.

## Delivery Order

Implement in this order:

1. governance and seam prep from the plan set
2. runtime-led composer contract
3. shell routing and shared composer extraction
4. project model and project detail canvas
5. library browser refactor
6. home cleanup and models/activity polish
7. final cutover removing the temporary `Studio` seam
8. test hardening and state-restoration cleanup

## Testing Focus

The redesign needs coverage at three layers:

- domain and presentation-model tests for project grouping and filters
- app-model tests for draft continuity, planning, and routing
- SwiftUI integration tests for shell behavior and shared composer visibility

Keep runtime-backed validation in the existing repo harness after the Swift package tests pass.

## Stage Exit Criteria

- The redesign is implemented through existing package seams.
- State ownership is clearer than before, not more scattered.
- The app still behaves like a thin runtime client after the UI changes land.

## SwiftUI-Pro Review Checklist

Files to review:

- `packages/clients/mlxr-mac-app/Sources/MLXRAppShell/AppModel.swift`
- `packages/clients/mlxr-mac-app/Sources/MLXRAppShell/AppModel+Workspace.swift`
- `packages/clients/mlxr-mac-app/Sources/MLXRAppShell/AppModel+Composer.swift`
- all new feature files added in the redesign tranche

Reference categories:

- `data.md` for observable ownership and binding structure
- `views.md` for file decomposition and removal of oversized multi-role files
- `swift.md` for concurrency and modern Foundation use
- `hygiene.md` for tests, comments where logic is non-obvious, and clean build state

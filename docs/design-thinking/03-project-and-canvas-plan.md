# Project And Canvas Plan

## Purpose

Define the continuity surface for work after `Studio` is removed as a separate destination.

## Project Model

`Project` is the user-facing name for `WorkspaceRecord`.

This resolves the open question in the screen brief:

- `WorkspaceRecord` is the right persistence home for projects
- `RunGroupRecord` remains the per-iteration history inside a project
- `CollectionRecord` stays optional and secondary for later saved sets or favorites workflows

No new database type is introduced for the redesign.

## Project Lifecycle

- The app always has one active project id.
- The default initial record may still exist internally as `default-workspace`, but it should not be surfaced to users as "Current Workspace."
- If the active project still has its untouched default title, the first accepted run renames it from the first prompt headline.
- `New Project` creates a fresh `WorkspaceRecord`, switches the shell to it, clears project-specific references, and preserves only safe composer defaults such as last modality and last model family.
- Users can rename the project from the project header.

## Where The Canvas Lives

The full results canvas lives in `Library` project detail.

That means:

- `Home` is a lightweight landing and quick-start surface
- `Library` top level is the project browser
- `Library` detail is the real "canvas" view for progress, history, and iteration

When the user submits from `Home`, the app transitions into the active project's detail view after runtime acceptance so the user lands in the continuity surface immediately.

## Canvas Structure

Project detail shows, in order:

1. project header
2. current run group card if work is running
3. latest hero result or latest run group
4. recent run groups
5. earlier run groups
6. imported source assets section when the project has imported references

The canvas is run-group aware, not raw-job aware.

## Run Group Presentation

Each run group shows:

- media thumbnails
- prompt headline
- compact metadata line
- source-reference count when relevant
- state when queued, running, failed, or completed
- inline actions

Primary inline actions are:

- `Re-create`
- `Edit`
- `Animate`
- `Use as reference`
- `Reveal in Finder`
- `Delete`

These actions mutate the shared composer draft rather than navigating to a separate creation screen.

## Hero Behavior

- Clicking a thumbnail promotes that asset into the hero preview.
- The hero preview is scoped to the currently opened project.
- Only the hero and visible thumbnails should instantiate heavy media players.
- The hero may reuse the existing modal viewer for larger playback and image preview.

## Imported Asset Behavior

Imported assets belong to the current project when imported from project detail.

When import starts from the top-level library browser, the app must either:

- attach the import to the active project, or
- ask the user to create or choose a target project before finalizing

Imports should not silently float outside project context.

## Explicit Non-Decision

Do not add a permanent right-edge filmstrip in the first redesign pass.

The project canvas already has:

- hero preview
- grouped result history
- direct actions back into the composer

That is enough to ship the new mental model before adding more chrome.

## Stage Exit Criteria

- A user can understand "where their work lives" as a project without learning repo terms.
- Results, progress, and reuse actions stay attached to the active project.
- Creating from `Home` lands in project detail instead of an orphaned temporary canvas.
- The implementation uses `WorkspaceRecord` plus `RunGroupRecord`, not a parallel project persistence model.

## SwiftUI-Pro Review Checklist

Files to review:

- project-detail canvas views under `packages/clients/mlxr-mac-app/Sources/MLXRFeatureGallery/`
- any project-summary presentation model in `MLXRAppDomain` or `MLXRAppShell`
- any replacement for `StudioCanvasView`

Reference categories:

- `views.md` for canvas decomposition and action extraction
- `data.md` for project and hero selection ownership
- `design.md` for adaptive layout and consistent control density
- `accessibility.md` for keyboard focus, labels, and reduced-motion handling

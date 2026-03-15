# Project And Canvas Plan

Status: working stage plan

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
- Manual project renaming is outside this compact tranche. The visible header stays dense and browse-first.

## Where The Canvas Lives

The full results canvas lives in `Library` project detail.

That means:

- `Home` is a lightweight landing and quick-start surface
- `Library` top level is the project browser
- `Library` detail is the real "canvas" view for progress, history, and iteration

When the user submits from `Home`, the app transitions into the active project's detail view after runtime acceptance so the user lands in the continuity surface immediately.

## Canvas Structure

Project detail shows, in order:

1. compact project header
2. inline pending-run strip for queued, running, or failed accepted work that has not materialized into visible assets yet
3. grouped square-tile result grid
4. modal viewer for full-aspect preview and action-heavy detail

The canvas is run-group aware, not raw-job aware, but the dense grid is now the primary browsing surface rather than a hero-preview stack.

## Run Group Presentation

Each run group shows:

- square media thumbnails
- prompt/title context through the tile label or viewer metadata
- count badge when a run group contains multiple outputs
- state badge when the accepted work is queued, running, or failed
- viewer actions instead of a heavy inline action row

Primary shipped actions are:

- `Edit`
- `Animate`
- `Use as reference`
- `Reveal in Finder`

A dedicated `Re-create` button remains later polish, and generated result-group deletion is still deferred. The current shipped reuse actions mutate the shared composer draft, expand the floating composer, and keep the user inside the current project context. Imported assets may still be removed from the library from the viewer.

## Focus And Viewer Behavior

- Single-clicking a thumbnail opens the modal viewer.
- The grid keeps square density; full aspect ratio belongs in the viewer.
- The project cover can be updated from the viewer using the currently focused asset.
- Only the visible viewer surface should instantiate heavy media playback.

## Imported Asset Behavior

Imported assets belong to the current project when imported from project detail.

When import starts from the top-level library browser, the app must not visibly
jump into a project before the file picker succeeds.

After the user actually chooses source files, the app may:

- attach the import to the currently opened project, or
- import against a deferred project id and only materialize/select that project after assets really exist

Imports should not silently float outside project context.

## Explicit Non-Decision

Do not add a permanent right-edge filmstrip in the first redesign pass.

The project canvas already has:

- compact header
- pending strip for accepted work that has not rendered yet
- grouped result history
- direct viewer actions back into the composer

That is enough to ship the new mental model before adding more chrome.

## Stage Exit Criteria

- A user can understand "where their work lives" as a project without learning repo terms.
- Results, progress, and reuse actions stay attached to the active project.
- Creating from `Home` lands in project detail instead of an orphaned temporary canvas.
- Accepted work that has not produced visible assets yet still appears inline in project detail.
- The implementation uses `WorkspaceRecord` plus `RunGroupRecord`, not a parallel project persistence model.

## SwiftUI-Pro Review Checklist

Files to review:

- project-detail canvas views under `packages/clients/mlxr-mac-app/Sources/MLXRFeatureGallery/`
- any project-summary presentation model in `MLXRAppDomain` or `MLXRAppShell`
- the project-detail canvas views under `MLXRFeatureGallery`

Reference categories:

- `views.md` for canvas decomposition and action extraction
- `data.md` for project and hero selection ownership
- `design.md` for adaptive layout and consistent control density
- `accessibility.md` for keyboard focus, labels, and reduced-motion handling

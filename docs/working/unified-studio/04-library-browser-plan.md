# Library Browser Plan

Status: working stage plan

## Purpose

Define the top-level `Library` experience as a project browser first and an asset browser second.

## Top-Level Library Shape

The top level of `Library` shows projects, not a flat asset wall.

The main header contains:

- search
- media filter chips: `All`, `Video`, `Image`, `Audio`
- favorites toggle
- `New Project`
- import action

The old side filter rail is removed.

## Project Cards

Each project card shows:

- title
- hero thumbnail
- latest prompt or asset subtitle
- quick counts for assets and sets
- active badge when the project is the current continuity context

Project cards should be quick to scan and easy to reopen, more like dense folders than inspector summaries.

## Project Detail Transition

Selecting a project opens project detail in place, not a separate destination.

Project detail reuses the shared shell:

- same nav rail
- same composer
- same activity affordance

Only the library body changes from project browser to project canvas.

## Search And Filter Behavior

Top-level search should match:

- project title
- prompt headlines
- model names
- source filenames when useful

Media filter chips at the top level filter projects by whether they contain matching assets.
Inside project detail, the same chips filter the visible assets and run groups for that project.

Model and task filters are still useful, but they live inline in the project-detail toolbar instead of occupying a permanent side rail.

## Collections And Favorites

Collections are no longer a primary browser affordance in the redesign.

Decision:

- keep favorites as a lightweight toolbar filter
- keep `CollectionRecord` data intact
- remove collection-first browsing from the main library layout

Collections can return later as optional saved sets if they still earn their complexity after the project browser ships.

## Import Behavior

`Import` remains a first-class library action because source assets are part of creative work, not a separate admin flow.

Rules:

- importing while a project is open adds the asset to that project
- importing from top-level library must not visibly jump into a project before the file picker succeeds
- after top-level file selection succeeds, the app may import against a deferred project id and only materialize/select that project after assets actually exist
- imported assets become eligible references in the shared composer immediately after import

## Viewer Behavior

The existing modal preview remains the right viewer pattern.

Keep:

- large in-window preview
- keyboard navigation
- quick reuse into the composer
- responsive preview layout so narrow windows stack preview and metadata instead of clipping
- single-click open from the project-detail grid
- composer hidden while the viewer is open

Do not reintroduce a persistent right inspector for library browsing.

## Stage Exit Criteria

- `Library` feels project-first at first glance.
- The flat asset browser and filter rail are no longer the primary mental model.
- Imported and generated assets remain reusable without losing project context.
- The same library shell supports both browse and continue flows.
- Project browser and project detail share the same toolbar rhythm and density language.

## SwiftUI-Pro Review Checklist

Files to review:

- `packages/clients/mlxr-mac-app/Sources/MLXRFeatureGallery/LibraryWorkspaceView.swift`
- new project-browser and modal-viewer files in `MLXRFeatureGallery`
- thumbnail and preview services used by the library

Reference categories:

- `navigation.md` for modal viewer and keyboard-driven presentation
- `performance.md` for lazy grids, thumbnailing, and avoiding eager players
- `accessibility.md` for keyboard and VoiceOver support
- `hygiene.md` for test coverage on browser state and modal behavior

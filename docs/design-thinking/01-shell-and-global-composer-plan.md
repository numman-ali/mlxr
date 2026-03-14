# Shell And Global Composer Plan

## Purpose

Define the app shell for the migration and the end-state after the composer becomes a shared bottom overlay.

## Shell Structure

The end-state primary navigation becomes:

- `Home`
- `Library`
- `Models`
- `Settings`

The rail footer contains:

- runtime status pill
- activity button with queued/running/failure badge state

`Activity` is not a top-level destination.

During migration, a temporary `Studio` destination may remain visible as a fallback route. If it exists, it is explicitly transitional:

- it may not own its own prompt or submit path
- it may not gain new workflow-specific behaviors
- it exists only until the global composer plus project canvas are fully proven

## Composer Placement

The composer is owned by `RootView` and rendered once as a shared overlay.

It is visible on:

- `Home`
- `Library` top level
- `Library` project detail

It is hidden on:

- `Models`
- `Settings`
- blocking bootstrap and starter-model overlays

This keeps the app feeling like one continuous studio without forcing management screens to carry creation chrome.

## Composer Scope

The composer replaces the current split between:

- workflow picker
- prompt card
- model/settings inspector

The composer contains:

- primary modality switcher: `Video` and `Image`
- plain-language sub-workflow picker under each modality
- prompt field
- adaptive reference strip
- model, aspect, quality, and duration or variation pills
- runtime status line
- generate action
- tune popover trigger

The composer does not contain:

- a persistent inspector column
- a separate workflow sidebar
- model install management

## Canonical Mode Model

The shell owns a single global draft, backed by `StudioWorkspaceDraft`, with this interpretation:

- `task` is the canonical runtime task
- `Video` and `Image` are the primary user-facing modes
- sub-workflows are thin labels over concrete `ProductTask` values

User-facing labels should be:

- `Video`
  - `Generate from text`
  - `Animate an image`
  - `Use audio as a guide`
  - `Use video as a guide`
  - `Blend between frames`
  - `Retake a clip`
- `Image`
  - `Generate from text`
  - `Edit an existing image`

Raw pipeline names do not appear in the main shell.

## Navigation Behavior

- Choosing `Create image` or `Create video` from `Home` focuses the composer and sets the matching mode.
- Opening a result action such as `Edit`, `Animate`, or `Use as reference` updates the global draft in place and keeps the user on the current screen.
- Submitting from `Home` creates a new project by default and transitions into that project detail after runtime acceptance.
- Submitting from the top-level `Library` browser creates a new project after runtime acceptance.
- Submitting from `Library` project detail keeps the user in the current project detail and appends into that project.
- Switching between `Home` and `Library` preserves the draft.

## Launch Routing

Initial routing becomes:

1. If no recommended starter model is installed, show `Models` setup.
2. If models exist but there is no meaningful project history, show `Home`.
3. If project history exists, show `Library` focused on the most recently opened project or the top-level project browser.

The shell should not auto-route into the temporary `Studio` fallback during migration.

## State Restoration

Persist and restore:

- last destination among `Home`, `Library`, `Models`, `Settings`
- active project id
- global draft
- last opened project detail if one was active

Do not restore transient activity overlay presentation.

## Stage Exit Criteria

- `Destination.studio` is removed from the app shell, or explicitly marked fallback-only until Stage 6.
- The rail still supports runtime status and activity without growing a new primary destination.
- One composer instance serves both `Home` and `Library`.
- The user can start on `Home`, `Library`, or `Models` without losing draft continuity.

## SwiftUI-Pro Review Checklist

Files to review:

- `packages/clients/mlxr-mac-app/Sources/MLXRAppShell/RootView.swift`
- `packages/clients/mlxr-mac-app/Sources/MLXRDesignSystem/NavComponents.swift`
- any new root-shell composer host view

Reference categories:

- `views.md` for shell decomposition and removal of oversized `body` logic
- `data.md` for shell-owned state and draft ownership
- `navigation.md` for overlay, sheet, and route behavior
- `performance.md` for avoiding repeated body work and unnecessary `AnyView`

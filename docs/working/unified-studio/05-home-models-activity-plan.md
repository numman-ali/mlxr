# Home, Models, And Activity Plan

Status: working stage plan

## Purpose

Define the three non-canvas surfaces that support the unified studio shell.

## Home

`Home` is orientation plus quick start. It is not a second studio, dashboard, or analytics wall.

### Home states

When no starter model is installed:

- show the recommended model setup path
- keep the page calm and explanatory
- hide the composer and replace it with a direct `Go to Models` or starter-install CTA until the user has at least one runnable image or video model

When models are installed but there is little or no project history:

- show a lightweight welcome state
- keep the composer ready
- show a small set of recent or recommended project starters if helpful

When the user already has project history:

- show recent projects
- show quick actions such as `Make a video`, `Make an image`, and `Continue project`
- keep the composer visible and ready

### Home behavior

- Submitting from `Home` creates or reuses the active project.
- After runtime acceptance, `Home` transitions into that project's library detail view.
- `Home` does not own persistent result history; project detail does.

## Models

`Models` remains the management surface and keeps its current density.

It should continue to own:

- recommended installs
- starter setup
- install queue visibility
- installed model detail
- removal

It should add or keep:

- example imagery or video stills
- machine fitness messaging
- disk usage
- "Get more models" return path from the composer

The composer stays hidden on this screen.

## Activity

`Activity` stays a rail-footer overlay, not a full destination.

This is an intentional refinement from the original screen sketch and matches the existing runtime-parity direction.

It shows:

- running jobs
- queued jobs
- install operations
- failures until dismissed
- a short recent-completed section when helpful

It does not become:

- a permanent history browser
- a place where old successful work lives

Old successful work belongs in `Library`.

## Runtime Status

Runtime presence appears in two places:

- a compact rail-footer status pill
- the composer status line on creation surfaces

Do not rebuild the old metric-heavy home summary just to expose runtime facts.

## Stage Exit Criteria

- `Home` orients without feeling like a dashboard.
- `Models` remains the install and runtime-management surface.
- `Activity` stays secondary, truthful, and easy to reach.
- `Home` does not render a live composer unless at least one runnable model exists.
- The app presents one coherent hierarchy instead of a mix of primary destinations and side utilities.

## SwiftUI-Pro Review Checklist

Files to review:

- `packages/clients/mlxr-mac-app/Sources/MLXRFeatureHome/HomeScreen.swift`
- `packages/clients/mlxr-mac-app/Sources/MLXRFeatureToolkit/ToolkitScreen.swift`
- activity footer and overlay views in `MLXRAppShell` / `MLXRActivityStrip`

Reference categories:

- `design.md` for calm hierarchy and token consistency
- `accessibility.md` for status communication that does not rely only on color
- `views.md` for avoiding control-panel sprawl

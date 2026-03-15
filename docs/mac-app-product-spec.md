# MLXR Mac App Product Spec

Status: canonical product spec for the first-party Mac app

## Summary

The first-party `MLXR` Mac app is the unified native consumer shell over the
shared runtime.

Current implementation status:

- native Swift package scaffold exists in `packages/clients/mlxr-mac-app/`
- runtime bridge, feature shells, and early tests are real
- the app now has a shared global composer, a project-first `Library`, rail-
  based `Activity`, guided starter-model setup, and no separate `Studio`
  destination in the visible or internal shell
- polish, onboarding, packaging, and release hardening still remain

Its job is to make the current runtime accessible to non-technical Mac users
without creating a second architecture or a host-owned inference stack.

## Product Goals

- zero-complication local use on a Mac
- clear generate, edit, and iteration flows
- straightforward progress, outputs, and errors
- one thin client over the same runtime used by the CLI

## Technology Direction

- native macOS app
- SwiftUI first, with AppKit interop only where needed
- shared runtime access over the local UDS-backed runtime contract
- on-demand runtime startup or attachment handled by the app

Do not build the first app as:

- a second local inference implementation
- a Tauri or Electron shell with duplicated runtime logic
- a special-case path that bypasses the shared runtime

## Development Path

Use a real dev `.app` bundle as the normal local workflow.

The canonical repo command is:

```bash
uv run python scripts/dev.py mac-app
```

That command should stay the default because it gives the app a stable macOS
identity, which is important for windowing, debugging, and Peekaboo-driven UI
automation. Direct `swift run` is still useful for package-only debugging, but
it should not be the main product-development path.

## Product Boundary

The first app should expose real and supported runtime capabilities on day one.

That means:

- promoted defaults are easy to find
- supported-but-unpromoted rows are visible, but clearly marked as advanced or
  preview
- blocked or planned rows are not exposed as runnable features

## Information Architecture

Top-level app areas:

- `Home`
- `Library`
- `Models`
- `Settings`
- `Activity` in the left rail footer

Creation is shared through one global composer rather than a top-level `Studio`
destination.

### Models

- curated recommended installs
- install queue visibility and status
- installed model details
- remove model from the MLXR-managed install home
- first-run starter-model setup flow for new users

### Library

- project-first browsing rather than a flat job log
- one continuity surface for progress, outputs, and reuse
- generated outputs plus imported source assets
- run-group-aware asset sets instead of a raw job list
- uniform square asset tiles in the browser
- large in-window modal preview instead of a sidebar inspector
- quick reveal in Finder
- fast reuse into the shared composer for edit, animate, guide, and retake
  flows

### Composer

- one shared creation surface for image and video workflows
- visible on `Home` and `Library`
- runtime-led planning and submission through the shared workflow seam
- model-aware presets and compact control pills in the floating bar
- current progress stays in context instead of pushing the user to a separate
  jobs page

### Settings

- runtime status
- runtime home and log discovery
- advanced import and runtime diagnostics
- prompt-helper settings

## Capability Presentation

Use three levels in the UI:

- `Recommended`
- `Advanced`
- `Unavailable`

Map them like this:

- `Recommended`: promoted rows
- `Advanced`: supported-but-unpromoted rows that are real and runnable
- `Unavailable`: blocked or planned rows, not interactive

## Prompt Enhancement

Prompt enhancement belongs in the app, not in the core runtime contract.

The product model keeps it as an optional helper with three modes:

- `Off`
- `Suggest`
- `Auto`

Default: `Suggest`

Rules:

- the current shipped app exposes prompt-helper mode in `Settings`
- inline composer suggestion UI is later polish, not current product truth
- original prompt text stays visible whenever inline suggestions are surfaced
- the user can accept or reject the suggestion when inline suggestions are surfaced
- core generation or edit must still work when the helper is unavailable
- the helper is a host-side pipeline, not a required inference dependency

First local helper investigation target:

- Qwen 3.5 local prompt-helper models, if they fit the machine and produce
  worthwhile guidance

If the helper is not ready in time, keep the core app launch unblocked and ship
without making it part of the hard runtime dependency chain.

## Runtime Relationship

The app should:

- discover or start the local runtime
- plan drafts through the shared workflow planning seam
- submit the same job-oriented requests as the CLI
- import local media through trusted local helper flows
- consume runtime job events and artifacts

The app should not:

- own scheduling
- own inference logic
- own family-specific stage sequencing
- redefine capability truth

The durable boundary rules for that split live in:

- [mac-app-runtime-contract.md](./mac-app-runtime-contract.md)

The current redesign implementation plan lives in:

- [working/unified-studio/00-master-plan.md](./working/unified-studio/00-master-plan.md)

## Repo Placement

The preferred repo home for the first app is:

- `packages/clients/mlxr-mac-app/`

This keeps it aligned with the runtime-first client model and distinct from:

- `packages/adapters/ltx-desktop/` for compatibility work
- future adapter or embedded host work that still stays thin over the same
  runtime

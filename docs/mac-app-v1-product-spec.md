# MLXR Mac App v1 Product Spec

## Summary

The first `MLXR` Mac app is a simple native macOS client over the shared
runtime.

It is not `MLXR Studio`.

Its job is to make the current runtime accessible to non-technical Mac users
without creating a second architecture or a host-owned inference stack.

## Product Goals

- zero-complication local use on a Mac
- clear generate and edit flows
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

## Product Boundary

The first app should expose real + supported runtime capabilities on day one.

That means:

- promoted defaults are easy to find
- supported-but-unpromoted rows are visible, but clearly marked as advanced or
  preview
- blocked or planned rows are not exposed as runnable features

## Information Architecture

Top-level app areas:

- `Images`
- `Video`
- `Library`
- `Jobs`
- `Settings`

### Images

- prompt-only generation
- image editing
- model selection
- optional advanced section for supported-but-unpromoted rows

### Video

- text-to-video
- image-to-video
- conditioned-audio
- advanced section for reference-video, interpolation, retake, and other
  supported-but-unpromoted rows

### Library

- exported outputs
- quick reveal in Finder
- rerun from prior settings later, if the runtime job metadata already supports
  it cleanly

### Jobs

- current progress
- recent failures
- stage status and output artifact links

### Settings

- runtime status
- runtime home and log discovery
- installed model visibility once the runtime model-management surface is ready
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

The v1 UX should expose it as an optional helper with three modes:

- `Off`
- `Suggest`
- `Auto`

Default: `Suggest`

Rules:

- original prompt text stays visible
- the user can accept or reject the suggestion
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
- submit the same job-oriented requests as the CLI
- import local media through trusted local helper flows
- consume runtime job events and artifacts

The app should not:

- own scheduling
- own inference logic
- own family-specific stage sequencing
- redefine capability truth

## Repo Placement

The preferred repo home for the first app is:

- `packages/clients/mlxr-mac-app/`

This keeps it aligned with the runtime-first client model and distinct from:

- `packages/adapters/ltx-desktop/` for compatibility work
- the later `MLXR Studio` product track

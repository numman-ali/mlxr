# Runtime-Led Composer Contract Plan

Status: working stage plan

## Purpose

Define how the global composer stays fully driven by runtime planning truth instead of drifting into host-owned validation.

## Core Rule

The app may shape language, layout, and defaults, but the runtime remains the authority for:

- whether a draft is runnable
- which references are required
- which output formats are allowed
- which model-task combinations are real

The composer is a presentation layer over `/v1/workflows/plan` and `/v1/workflows/run`, not a second workflow engine.

## Draft To Runtime Mapping

The composer keeps using `StudioWorkspaceDraft` as its editable state.

On every meaningful change, the app derives a `WorkflowIntent` from:

- selected model id
- accepted prompt text
- selected `ProductTask`
- resolved reference handles
- resolved settings
- selected output format
- family-local pack extensions

Meaningful changes include:

- prompt edits
- model changes
- task or mode changes
- reference add, remove, or reorder
- quality, aspect, duration, or variation changes
- tune-popover custom setting changes

Planning stays debounced in app state exactly once, not per view.

## What The Composer Reads From Planning

`WorkflowPlanResult.readiness` is the only source for:

- generate button enabled or disabled state
- blocking copy shown to the user
- warnings
- visible reference requirements
- allowed output formats

`WorkflowPlanResult.capability` remains the source for:

- task support
- model-level constraints
- fixed or clamped settings

The app may summarize this truth into simpler UI language, but it may not replace it.

## Required Planning Presentation Payload

The shared contract for the composer is `WorkflowPlanResult.presentation`.

The runtime returns a typed presentation block with:

- `primary_mode`
- `selected_task`
- `subworkflows`
- `reference_slots`
- `controls`

`subworkflows` carries:

- runtime task id
- user-facing label
- primary mode
- default flag

`reference_slots` carries:

- stable slot id
- user-facing label
- accepted reference kind
- optional runtime description
- required flag
- minimum count
- maximum count
- accepted roles
- whether multiple values are allowed

`controls` carries only phase-1 composer pills:

- quality presets
- aspect presets
- duration presets
- variation counts

This payload is required for the Mac app and CLI planning flows. It is not an optional metadata side channel.

## Runtime Ownership

The runtime owns:

- presentation defaults
- which sub-workflows are available
- which reference slots exist
- which composer controls are meaningful for the selected model/task

Families implement this through their workflow strategy, and the shared planner includes the resulting presentation block in `WorkflowPlanResult`.

The host owns only:

- visual arrangement
- icons and chrome
- simple wording adjustments that do not change capability truth

## Reference Slot Rules

Reference slot rendering is derived from runtime requirements first and UI language second.

The app uses these presentational labels where possible:

- image `start_frame`
- image `end_frame`
- audio `guide_audio`
- video `guide_video`
- video `source_video`

If the runtime returns a role or description outside the known label set, the app surfaces the runtime description directly instead of guessing.

Reference acceptance rules stay runtime-led:

- asset actions may preselect a task
- drag and drop may suggest a task
- final legality still comes from planning readiness

## Packs And Family Extensions

`PackRecord` remains optional UX sugar on top of runtime-supported models.

Packs may:

- apply namespaced extensions already supported by the selected family
- supply a friendlier label such as `Motion Track` or `Lightning`

Packs may not:

- invent unsupported tasks
- bypass readiness failures
- become a second capability registry

## Fallback Behavior

The host should treat a missing or partial presentation block as a runtime bug during this tranche, not as a reason to infer support locally.

If presentation data is missing or incomplete:

- keep submission blocked unless readiness is explicitly usable
- surface the runtime error plainly
- do not synthesize hidden sub-workflow or slot behavior from host heuristics

## Prompt Helper Boundary

Prompt helper behavior stays host-side only.

- `Off`, `Suggest`, and `Auto` remain presentation choices.
- The runtime only receives the prompt text the user actually chose to submit.
- Helper-generated text never mutates capability truth, selected task, or required references.

## Submission Contract

Run submission happens in this order:

1. plan the current draft
2. require `readiness.ready == true`
3. submit through `/v1/workflows/run`
4. record project or run-group activity only after runtime acceptance

This preserves the existing repo rule that the app reflects accepted runtime work, not optimistic local work.

## Planned UI Constraints

The first redesign pass keeps output choices intentionally narrow:

- image outputs stay on runtime-allowed still formats, with `png` as the default
- video outputs stay on runtime-allowed video formats, with `mp4` as the default
- negative prompt, seed, width, height, steps, and guidance stay behind tune

If the runtime disallows a format or fixes a numeric value, the composer reflects that instead of pretending the control is still open.

## Stage Exit Criteria

- The shell can render a fully adaptive composer without adding a host-owned validation table.
- Reference requirements and blocking issues come only from planning.
- Packs and prompt helper remain layered presentation features instead of drifting into runtime truth.
- A failed or unsupported draft is explained through runtime readiness, not custom host rules.

## SwiftUI-Pro Review Checklist

Files to review:

- `packages/clients/mlxr-mac-app/Sources/MLXRAppShell/AppModel+Composer.swift`
- `packages/clients/mlxr-mac-app/Sources/MLXRFeatureCreate/GlobalComposerBar.swift`
- `packages/clients/mlxr-mac-app/Sources/MLXRAppDomain/ExecutionSchemas.swift`
- `packages/core/runtime-workflows/src/mlxr/core/workflows/presentation.py`
- family workflow strategy files under `packages/families/*/src/mlxr/families/*/workflows.py`

Reference categories:

- `data.md` for shared state ownership and binding discipline
- `views.md` for composer decomposition and action extraction
- `performance.md` for avoiding expensive derived work in the view layer
- `swift.md` for modern concurrency and clean async submission flow

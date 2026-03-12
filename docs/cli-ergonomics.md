# CLI Ergonomics And Access Surfaces

## Summary

The CLI is the current public face of `MLXR` for humans, agents, and local
automation.

It should stay:

- thin over the runtime
- stable in shape
- machine-readable by default
- simple for common local generation tasks

## Current Real Surface

What is real today:

- `mlxr generate`
- local import of trusted image, video, audio, and LoRA inputs
- `--plan-only`
- `--wait`
- `--export-path`

What is intentionally still growing:

- `mlxr serve`
- `mlxr models *`
- `mlxr jobs *`
- later adapter or reusable-asset flows

## Who The CLI Serves

### 1. Agents and scripts

They need:

- predictable flags
- JSON output
- stable failure behavior
- thin access to the runtime without reimplementing logic

### 2. Local creators

They need:

- simple task-oriented commands
- sane defaults
- clear progress and failure reasons
- explicit export behavior

## Ergonomic Rules

- keep the current one-prompt plus optional refs model
- keep shared runtime tasks simple and typed
- do not let one-family parity flags take over the shared surface
- keep family-specific controls family-local unless they earn a shared design
- preserve the runtime as the canonical source of scheduling, provenance, and
  capability truth

## Intended Public Growth

### `mlxr generate`

Keep it as the main job-oriented entrypoint.

It should remain the easiest way to:

- submit a job
- wait for completion
- export the first artifact
- keep results scriptable

### `mlxr models *`

This should become the public install and inspection surface for:

- install
- update
- list
- doctor
- prune

The app should reuse this logic rather than inventing a separate model-install
contract.

### `mlxr jobs *`

This should become the public operational surface for:

- inspect
- list
- follow events
- export outputs

### `mlxr serve`

This should become the thin service-management path for local runtime startup
and status.

## Relationship To Other Surfaces

| Surface | Role |
| --- | --- |
| CLI | trusted local thin client for humans and agents |
| daemon API | canonical runtime contract |
| Mac app | native client over the same runtime |
| desktop adapter | compatibility seam for an existing product shell |
| embedded first-party host | later Apple-native host boundary |

## What The CLI Must Not Become

- a second inference engine
- a host-owned place for family-specific orchestration logic
- a dumping ground for every parity knob needed by one family
- a replacement for the runtime capability model

# Workflow Orchestration Design

Status: working platform doctrine for the workflow layer. Implement against this now, but treat the exact workflow schema as provisional until more than one family has validated it.

## Purpose

This doc defines the workflow layer that sits between the scheduler, model-family adapters, and host adapters.

`MLXR` now treats this layer as part of the core runtime, not as incidental glue inside one family or one host.

The immediate pressure came from `LTX-2.3 Fast`, but the point of the layer is broader:

- keep stage sequencing and lifecycle truth in one place
- let families declare truthful workflows without becoming mini-schedulers
- keep hosts thin even when tasks are multi-stage and long-running

## Mental Model

```text
Host request
  -> control plane resolves source, artifact, and capability
  -> core workflow layer selects a workflow template
  -> scheduler reserves resources per stage
  -> worker runs family-owned stage implementations
  -> core workflow layer records state, telemetry, artifacts, and failures
  -> host consumes job events and output handles
```

The workflow layer is where `MLXR` decides what happens next, what resources are reserved, what evidence is recorded, and what the host is allowed to claim about progress.

## Why This Is Core MLXR

Without a core workflow layer, the same mistakes repeat in different forms:

- hosts start owning stage order, warm-state assumptions, and failure handling
- families hide reusable lifecycle logic inside private adapter code
- scheduler decisions drift away from the actual execution graph
- job progress becomes a UI story instead of a runtime truth

The workflow layer exists to stop that drift.

It is the runtime-owned bridge between:

- scheduler truth
- family-specific execution stages
- host-visible lifecycle events

## Responsibility Split

### Core workflow layer owns

- workflow-template selection from family, task, and profile
- stage ordering, optional-branch rules, and lifecycle transitions
- scheduler reservations, admission, cancellation, and retries where supported
- runtime-managed inputs, outputs, state handoff, and artifact ownership
- generic telemetry, failure taxonomy, and event emission
- the rule that hosts consume workflows, not invent them

### Family adapters own

- truthful workflow templates for their tasks and profiles
- family-specific stage implementations and worker-local state
- source-role requirements and artifact payload expectations
- family constraints, capability declarations, and fail-closed compatibility checks
- explicit naming of what is not yet supported

### Host adapters own

- user-facing request shaping and ergonomics
- import and export helpers for trusted local UX
- capability-driven UI decisions and progress presentation
- mapping runtime outputs into the host product surface

Hosts do not own inference logic, scheduler policy, or reusable stage sequencing.

## Workflow Template Contract

Each family should be able to declare a workflow template that the core runtime can interpret without host-specific knowledge.

At minimum, a workflow template should name:

- `workflow_id`, `family_id`, task, and profile
- required artifact roles and accepted input-handle kinds
- stage order and optional branches
- state handoff points between stages
- scheduler reservation hints per stage
- output artifact kinds and completion conditions
- failure categories or fail-closed boundaries that matter to the host

The workflow layer interprets the template. The family adapter supplies the family truth inside it.

## Stage Vocabulary

Not every family needs the same stage names, but the runtime should reuse shared labels where the meaning is genuinely the same.

Examples of core-relevant stages:

- `inspect`
- `convert`
- `load`
- `condition_inputs`
- `prompt_encode`
- `generate`
- `decode`
- `encode`
- `export`
- `unload`

Families can add narrower stage names when needed, such as `denoise_stage1` or `latent_upscale`, but they should not hide a reusable shared stage behind a family-private alias unless the semantics are materially different.

## Bring-Up Default

For a new family, start with the smallest truthful workflow that proves the platform seam.

That usually means:

1. name the first task and profile worth supporting
2. declare the minimum required source roles
3. define the smallest honest workflow template
4. expose fail-closed boundaries early
5. add optional branches only after the first slice is validated

Use [docs/family-bringup/README.md](/Users/numman/Repos/mlxr/docs/family-bringup/README.md) for the repeated bring-up workflow and `.agents/skills/family-bringup/` when the task is specifically family onboarding or workflow reshaping.

## Non-Goals

This layer is not:

- a generic workflow engine for arbitrary user-authored graphs
- a place for host-specific UX branching
- a reason to flatten real family differences into fake uniformity
- a replacement for family validation ladders or benchmark evidence

## Review Gate

Before commit, review workflow changes from a fresh-eyes posture:

- did reusable sequencing land in core, not a host
- did the family declare truth instead of silently approximating it
- did the host stay thin
- did the docs and validation plan move with the new workflow truth

If the answer is no, the change is not done yet.

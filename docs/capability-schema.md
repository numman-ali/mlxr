# Capability Schema

Status: provisional schema `v0`. This is concrete enough to drive implementation and host integration, but it should not be versioned as stable until the freeze criteria pass.

## Purpose

The capability descriptor tells hosts what the runtime can actually do with a given portable artifact on the current machine and under the current policy state.

It is not just model metadata. It is:

- a validation surface
- a host UX surface
- a scheduler hint surface
- a policy and provenance surface

## Required Top-Level Fields

| Field | Purpose |
| --- | --- |
| `model_id` | Stable runtime identifier for the artifact instance. |
| `artifact_digest` | Identifies the portable artifact, not the build cache. |
| `family` | High-level family such as `ltx`, `sdxl`, `whisper`, `qwen-vlm`. |
| `family_variant` | Variant such as `fast`, `distilled`, or provider-converted subtypes. |
| `tasks` | Supported task IDs. |
| `modalities_in` | Input modalities. |
| `modalities_out` | Output modalities. |
| `constraints` | Hard input constraints and formulas. |
| `conditioning` | Supported conditioning types and high-level semantics. |
| `profiles_by_task` | Supported profiles by task. |
| `streaming` | Event and partial-output support. |
| `artifacts_out` | Runtime-managed artifact formats that can be emitted. |
| `scheduler_class` | Scheduler class used for admission and stage policy. |
| `hardware_tiers` | Machine-fit guidance and limitations. |
| `dependencies` | External or platform-managed dependencies such as encoders or native helpers. |
| `policy` | License, access state, and remote-code facts. |
| `extensions_schema` | Family-specific extension namespace and version. |

## Constraints

`constraints` must express machine-checkable rules instead of vague notes.

Examples:

- multiples of 32
- frame-count formulas such as `8n+1`
- maximum number of reference images
- supported aspect-ratio profiles
- whether audio duration must match video duration

If the runtime cannot validate a constraint before execution, it should say so explicitly.

## Profiles By Task

Profiles are task-specific, not model-global. A model may support:

- `fp16` for one task
- `q8` for another
- degraded low-memory profiles only on smaller machines

Hosts should not guess profile names. The runtime declares them.

## Scheduler Class

The capability schema must expose the scheduler class because hosts need to know whether they are asking for:

- a heavy video DiT job
- an image diffusion batch
- a streaming speech workload
- an interactive VLM or text session

This field is also how the control plane explains admission decisions.

## Hardware Tiers

Hardware guidance must be part of capabilities because portability is not the same thing as fitness on the current machine.

Each tier should include:

- tier name such as `minimum`, `degraded`, `recommended`
- memory requirement
- notes about limited profiles or output formats
- known exclusions if a feature is disabled on that tier

## Policy Block

`policy` must include:

- `license`
- `access_state`
- `remote_code_required`
- `remote_code_approved`
- `redistribution_state` when known

Hosts should never need to reverse-engineer this from provider metadata.

## Example

```json
{
  "model_id": "ltx-2.3-fast-local",
  "artifact_digest": "sha256:example",
  "family": "ltx",
  "family_variant": "fast",
  "tasks": ["video.generate", "video.condition.image"],
  "modalities_in": ["text", "image"],
  "modalities_out": ["video", "audio"],
  "constraints": {
    "width": { "multiple_of": 32 },
    "height": { "multiple_of": 32 },
    "num_frames": { "formula": "8n+1" }
  },
  "conditioning": {
    "image": true,
    "video": false,
    "audio": false,
    "lora": true
  },
  "profiles_by_task": {
    "video.generate": ["fp16", "q8"],
    "video.condition.image": ["fp16", "q8"]
  },
  "streaming": {
    "progress_events": true,
    "partial_artifacts": true,
    "token_deltas": false,
    "segment_events": true
  },
  "artifacts_out": ["mp4", "mov", "wav"],
  "scheduler_class": "media_video_dit",
  "hardware_tiers": [
    {
      "tier": "recommended",
      "memory_gb": 64,
      "notes": "full fast profile target"
    },
    {
      "tier": "degraded",
      "memory_gb": 32,
      "notes": "reduced profiles only; benchmark-gated"
    }
  ],
  "dependencies": {
    "media_encode": {
      "required": true,
      "policy": "runtime-managed"
    }
  },
  "policy": {
    "license": "ltx-2-community-license-agreement",
    "access_state": "public",
    "remote_code_required": false,
    "remote_code_approved": false
  },
  "extensions_schema": {
    "namespace": "ltx",
    "version": "1"
  }
}
```

## Rules

- This schema describes portable artifact behavior plus current-machine execution facts.
- It must not report machine-local build-cache contents as portable capabilities.
- Hosts may extend UI using family-specific `extensions_schema`, but they must not depend on undeclared fields for basic correctness.

# Family Bring-Up Template

Copy this structure when starting a new family note, ADR input, or implementation plan.

```md
# <Family Name> Bring-Up Note

Status: <proposed | in progress | first truthful slice | under validation>

## Why This Family

- Product reason:
- Platform reason:
- Why this family pressures `MLXR` differently from current families:

## First Truthful Slice

- First task:
- First profile:
- What is explicitly in scope:
- What is explicitly out of scope:

## Upstream And Source Map

| Role | Example source | Required? | Notes |
| --- | --- | --- | --- |
| checkpoint |  | yes/no |  |
| tokenizer / text encoder |  | yes/no |  |
| auxiliary component |  | yes/no |  |

Notes:

- license and access state:
- remote-code expectation:
- preflight inspection plan:

## Portable Artifact Contract

- Required payload roles:
- Required metadata:
- Load-time fail-closed checks:
- Output artifact kinds:

## Workflow Templates

### <task/profile/workflow_id>

- Inputs:
- Core-owned orchestration:
- Family-owned stages:
- Host-owned concerns:
- Optional branches:
- Completion condition:

## Capability Contract

- Tasks and modalities:
- Constraints:
- Profiles:
- Hardware expectations:
- Unsupported or deferred capability:

## Validation Ladder

- Smoke rung:
- Meaningful semantic or quality rung:
- Recommended-profile rung:
- Required logs or telemetry:

## Open Questions

- Question:
- What evidence would close it:

## Fresh-Eyes Review Notes

- Boundary leaks checked:
- Overclaims checked:
- Docs updated:
```

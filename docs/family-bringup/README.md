# Family Bring-Up

Status: default documentation pack for bringing up a new model family or re-basing an existing family onto the current `MLXR` platform contract.

## Purpose

This folder exists so family onboarding does not start from a blank page every time.

Use it when we need to decide:

- what belongs in the core workflow layer
- what belongs in the family adapter
- what belongs in a host adapter
- what the smallest truthful slice really is

## What To Read First

1. [workflow-orchestration-design.md](/Users/numman/Repos/mlxr/docs/workflow-orchestration-design.md)
2. [technical-design.md](/Users/numman/Repos/mlxr/docs/technical-design.md)
3. [phased-delivery-plan.md](/Users/numman/Repos/mlxr/docs/phased-delivery-plan.md)
4. [09-open-questions-and-validation-plan.md](/Users/numman/Repos/mlxr/docs/research/09-open-questions-and-validation-plan.md)

Then use:

- [checklist.md](/Users/numman/Repos/mlxr/docs/family-bringup/checklist.md)
- [template.md](/Users/numman/Repos/mlxr/docs/family-bringup/template.md)
- [evaluation-framework.md](/Users/numman/Repos/mlxr/docs/family-bringup/evaluation-framework.md) when deciding whether a base model, adapter, or quality claim is actually proven
- [debugging-decision-tree.md](/Users/numman/Repos/mlxr/docs/family-bringup/debugging-decision-tree.md) when the family runs but the result is wrong, unstable, or mismatched between probe and CLI/runtime paths

## Default Sequence

1. Name the family, the first proving task, and the first truthful profile.
2. Record source roles, provider pressure, license, remote-code, and artifact expectations.
3. Separate core workflow responsibilities from family-stage responsibilities and host concerns.
4. Write the capability contract and fail-closed boundaries before claiming support.
5. Define the validation ladder from smoke rung to meaningful quality rung.
6. Update the open-questions doc if the family exposes a new platform uncertainty.
7. Do the fresh-eyes review pass before commit.

## Output Expectation

A family bring-up should leave behind:

- a clear first-slice claim
- an explicit workflow template or stage graph
- a documented source and artifact contract
- a validation ladder
- a short list of open questions that still block freeze

If the family still depends on hand-wavy host logic or undocumented workflow assumptions, the bring-up is not finished.

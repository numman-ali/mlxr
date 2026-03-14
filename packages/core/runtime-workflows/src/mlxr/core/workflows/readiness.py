from __future__ import annotations

from mlxr.core.schemas import (
    CapabilityDescriptor,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowPlanReadiness,
    WorkflowReference,
    WorkflowReferenceRequirement,
)


def build_plan_readiness(
    *,
    capability: CapabilityDescriptor,
    intent: WorkflowIntent,
    plan: WorkflowPlan,
    requirements: list[WorkflowReferenceRequirement],
    extra_blocking_issues: list[str] | None = None,
) -> WorkflowPlanReadiness:
    blocking_issues: list[str] = []
    if not intent.prompt.strip():
        blocking_issues.append("Add a prompt so MLXR knows what to make or change.")

    allowed_output_formats = list(capability.artifacts_out)
    requested_format = intent.output.artifact_format
    if requested_format is not None and requested_format not in allowed_output_formats:
        blocking_issues.append(
            f"Output format '{requested_format}' is not supported for this model."
        )

    blocking_issues.extend(
        _constraint_blocking_issues(
            constraints=capability.constraints,
            params=intent.params,
        )
    )

    counts = _reference_counts(intent.references)
    blocking_issues.extend(
        _reference_requirement_issues(counts=counts, requirements=requirements)
    )
    if extra_blocking_issues:
        blocking_issues.extend(extra_blocking_issues)

    return WorkflowPlanReadiness(
        ready=not blocking_issues,
        blocking_issues=blocking_issues,
        warnings=list(plan.warnings),
        reference_requirements=requirements,
        allowed_output_formats=allowed_output_formats,
    )


def _constraint_blocking_issues(
    *,
    constraints: dict[str, object],
    params: dict[str, object],
) -> list[str]:
    blocking_issues: list[str] = []
    for name in ("width", "height", "num_frames", "num_inference_steps"):
        blocking_issues.extend(
            _integer_constraint_issues(
                name=name,
                value=params.get(name),
                constraints=constraints,
            )
        )
    blocking_issues.extend(
        _numeric_constraint_issues(
            name="guidance_scale",
            value=params.get("guidance_scale"),
            constraints=constraints,
        )
    )
    return blocking_issues


def _integer_constraint_issues(
    *,
    name: str,
    value: object,
    constraints: dict[str, object],
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, int) or isinstance(value, bool):
        return [f"{name} must be an integer."]

    constraint = constraints.get(name)
    if not isinstance(constraint, dict):
        return []

    issues: list[str] = []
    fixed = constraint.get("fixed")
    if isinstance(fixed, int) and value != fixed:
        issues.append(f"{name} must be exactly {fixed}.")

    minimum = constraint.get("minimum")
    if isinstance(minimum, int) and value < minimum:
        issues.append(f"{name} must be >= {minimum}.")

    maximum = constraint.get("maximum")
    if isinstance(maximum, int) and value > maximum:
        issues.append(f"{name} must be <= {maximum}.")

    multiple_of = constraint.get("multiple_of")
    if isinstance(multiple_of, int) and multiple_of > 0 and value % multiple_of != 0:
        issues.append(f"{name} must be an integer multiple of {multiple_of}.")

    formula = constraint.get("formula")
    if formula == "8n+1" and (value < 1 or (value - 1) % 8 != 0):
        issues.append(f"{name} must satisfy the current 8n+1 rule.")

    return issues


def _numeric_constraint_issues(
    *,
    name: str,
    value: object,
    constraints: dict[str, object],
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return [f"{name} must be numeric."]

    numeric_value = float(value)
    constraint = constraints.get(name)
    if not isinstance(constraint, dict):
        return []

    issues: list[str] = []
    fixed = constraint.get("fixed")
    if isinstance(fixed, (int, float)) and not isinstance(fixed, bool):
        if numeric_value != float(fixed):
            issues.append(f"{name} must be exactly {fixed}.")

    minimum = constraint.get("minimum")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        if numeric_value < float(minimum):
            issues.append(f"{name} must be >= {minimum}.")

    maximum = constraint.get("maximum")
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        if numeric_value > float(maximum):
            issues.append(f"{name} must be <= {maximum}.")

    return issues


def _reference_counts(references: list[WorkflowReference]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reference in references:
        counts[reference.kind] = counts.get(reference.kind, 0) + 1
    return counts


def _reference_requirement_issues(
    *,
    counts: dict[str, int],
    requirements: list[WorkflowReferenceRequirement],
) -> list[str]:
    issues: list[str] = []
    for requirement in requirements:
        actual_count = counts.get(requirement.kind, 0)
        if actual_count < requirement.minimum_count:
            issues.append(requirement.description)
            continue
        if (
            requirement.maximum_count is not None
            and actual_count > requirement.maximum_count
        ):
            maximum = requirement.maximum_count
            issues.append(
                f"Use no more than {maximum} {requirement.kind} reference"
                + ("s" if maximum != 1 else "")
                + " for this workflow."
            )
    return issues

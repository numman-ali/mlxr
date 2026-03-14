from __future__ import annotations

from dataclasses import dataclass, field

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobRequest,
    ModelRecord,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowPlanResult,
)

from .contracts import (
    FamilyWorkflowStrategy,
    WorkflowNotSupportedError,
    WorkflowPlanningContext,
)


@dataclass(slots=True)
class WorkflowStrategyRegistry:
    _strategies: dict[str, FamilyWorkflowStrategy] = field(default_factory=dict)

    def register(self, strategy: FamilyWorkflowStrategy) -> None:
        self._strategies[strategy.family_id] = strategy

    def get(self, family_id: str) -> FamilyWorkflowStrategy:
        try:
            return self._strategies[family_id]
        except KeyError as exc:
            raise WorkflowNotSupportedError(
                f"No workflow strategy is registered for family '{family_id}'"
            ) from exc

    def has(self, family_id: str) -> bool:
        return family_id in self._strategies


class WorkflowPlanner:
    def __init__(self, registry: WorkflowStrategyRegistry) -> None:
        self.registry = registry

    def plan_for_model(
        self, *, model: ModelRecord, intent: WorkflowIntent
    ) -> WorkflowPlanResult:
        capability = self._capability_for_model(model)
        strategy = self.registry.get(model.family)
        context = WorkflowPlanningContext(model=model, capability=capability)
        plan = strategy.plan(context, intent)
        readiness = strategy.readiness(context, intent, plan)
        return WorkflowPlanResult(
            capability=capability,
            plan=plan,
            readiness=readiness,
        )

    def job_request_for_plan(
        self,
        *,
        model: ModelRecord,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
    ) -> JobRequest:
        capability = self._capability_for_model(model)
        strategy = self.registry.get(model.family)
        context = WorkflowPlanningContext(model=model, capability=capability)
        return strategy.to_job_request(context, intent, plan)

    def _capability_for_model(self, model: ModelRecord) -> CapabilityDescriptor:
        capability = model.capability or (
            model.artifact.capability if model.artifact else None
        )
        if capability is None:
            raise ValueError(f"Model '{model.model_id}' has no capability descriptor")
        return capability

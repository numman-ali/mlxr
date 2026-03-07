from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mlx_runtime_schemas import (
    CapabilityDescriptor,
    JobRequest,
    ModelRecord,
    WorkflowIntent,
    WorkflowPlan,
)


@dataclass(slots=True)
class WorkflowPlanningContext:
    model: ModelRecord
    capability: CapabilityDescriptor


class FamilyWorkflowStrategy(Protocol):
    family_id: str

    def plan(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
    ) -> WorkflowPlan: ...

    def to_job_request(
        self,
        context: WorkflowPlanningContext,
        intent: WorkflowIntent,
        plan: WorkflowPlan,
    ) -> JobRequest: ...

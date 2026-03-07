from __future__ import annotations

from dataclasses import dataclass

from mlx_runtime_core import RuntimeCatalog
from mlx_runtime_schemas import (
    JobSubmitResult,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowPlanResult,
    WorkflowRunResult,
)
from mlx_runtime_workflows import WorkflowPlanner

from .jobs import JobManager


@dataclass
class WorkflowService:
    catalog: RuntimeCatalog
    planner: WorkflowPlanner
    job_manager: JobManager

    def plan(self, intent: WorkflowIntent) -> WorkflowPlanResult:
        model = self.catalog.get_model(intent.model_id)
        return self.planner.plan_for_model(model=model, intent=intent)

    def run(
        self, intent: WorkflowIntent, plan: WorkflowPlan | None = None
    ) -> WorkflowRunResult:
        model = self.catalog.get_model(intent.model_id)
        resolved_plan = (
            plan or self.planner.plan_for_model(model=model, intent=intent).plan
        )
        job_request = self.planner.job_request_for_plan(
            model=model,
            intent=intent,
            plan=resolved_plan,
        )
        record = self.job_manager.submit(job_request)
        return WorkflowRunResult(
            plan=resolved_plan,
            submit=JobSubmitResult(job_id=record.job_id, record=record),
        )

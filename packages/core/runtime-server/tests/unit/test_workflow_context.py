from __future__ import annotations

import unittest
from types import SimpleNamespace

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobOutputPolicy,
    JobRecord,
    JobRequest,
    JobState,
    PolicyDescriptor,
    WorkflowContextMetadata,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowPlanResult,
)
from mlxr.core.server.workflows import WorkflowService


class _FakeCatalog:
    def get_model(self, model_id: str) -> object:
        return SimpleNamespace(model_id=model_id)


class _FakePlanner:
    def plan_for_model(
        self, model: object, intent: WorkflowIntent
    ) -> WorkflowPlanResult:
        return WorkflowPlanResult(
            capability=CapabilityDescriptor(
                model_id=getattr(model, "model_id", "test-model"),
                artifact_digest="digest",
                family="test",
                tasks=["image.generate"],
                modalities_in=["text"],
                modalities_out=["image"],
                scheduler_class="test.scheduler",
                policy=PolicyDescriptor(access_state="public"),
                metadata={},
            ),
            plan=WorkflowPlan(
                model_id=getattr(model, "model_id", "test-model"),
                family="test",
                selected_task=intent.task or "image.generate",
                resolved_prompt=intent.prompt,
            ),
        )

    def job_request_for_plan(
        self, model: object, intent: WorkflowIntent, plan: WorkflowPlan
    ) -> JobRequest:
        return JobRequest(
            model_id=getattr(model, "model_id", "test-model"),
            task=plan.selected_task,
            inputs={"prompt": intent.prompt},
            params={},
            output=JobOutputPolicy(),
        )


class _FakeJobManager:
    def __init__(self) -> None:
        self.submitted_request: JobRequest | None = None

    def submit(self, request: JobRequest):
        self.submitted_request = request
        return JobRecord(
            job_id="job-1",
            request=request,
            state=JobState.ACCEPTED,
            error=None,
            artifacts=[],
        )


class _FakeInputStore:
    def get(self, handle_id: str):  # pragma: no cover - not used in this test
        return None


class WorkflowContextTests(unittest.TestCase):
    def test_run_carries_workflow_context_into_job_request(self) -> None:
        manager = _FakeJobManager()
        service = WorkflowService(
            catalog=_FakeCatalog(),
            planner=_FakePlanner(),
            job_manager=manager,
            input_store=_FakeInputStore(),
        )

        intent = WorkflowIntent(
            model_id="z-image-turbo-local",
            prompt="cinematic portrait",
            task="image.generate",
            context=WorkflowContextMetadata(
                workspace_id="workspace-1",
                run_group_id="group-1",
                source_asset_ids=["asset-a"],
                intent_label="Make Image",
                preset_id="square-standard",
            ),
        )

        result = service.run(intent)

        self.assertEqual(result.submit.job_id, "job-1")
        self.assertIsNotNone(manager.submitted_request)
        assert manager.submitted_request is not None
        self.assertIsNotNone(manager.submitted_request.context)
        assert manager.submitted_request.context is not None
        self.assertEqual(manager.submitted_request.context.workspace_id, "workspace-1")
        self.assertEqual(manager.submitted_request.context.run_group_id, "group-1")
        self.assertEqual(
            manager.submitted_request.context.source_asset_ids, ["asset-a"]
        )

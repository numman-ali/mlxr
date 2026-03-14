from __future__ import annotations

import unittest
from types import SimpleNamespace

from mlxr.core.schemas import (
    CapabilityDescriptor,
    InputHandleRecord,
    JobOutputPolicy,
    JobRecord,
    JobRequest,
    JobState,
    PolicyDescriptor,
    WorkflowContextMetadata,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowPlanPresentation,
    WorkflowPlanReadiness,
    WorkflowPlanResult,
    WorkflowPresentationSubworkflow,
    WorkflowReference,
    WorkflowReferenceRequirement,
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
            readiness=WorkflowPlanReadiness(
                ready=True,
                warnings=["test warning"],
                reference_requirements=[
                    WorkflowReferenceRequirement(
                        kind="image",
                        minimum_count=0,
                        maximum_count=1,
                        description="Optional image reference",
                    )
                ],
                allowed_output_formats=["png"],
            ),
            presentation=WorkflowPlanPresentation(
                primary_mode="image",
                selected_task=intent.task or "image.generate",
                subworkflows=[
                    WorkflowPresentationSubworkflow(
                        task="image.generate",
                        label="Generate from text",
                        mode="image",
                        default=True,
                    )
                ],
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
    def __init__(self, records: dict[str, object] | None = None) -> None:
        self.records = records or {}

    def get(self, handle_id: str):  # pragma: no cover - not used in this test
        return self.records.get(handle_id)


class _BlockingPlanner(_FakePlanner):
    def plan_for_model(
        self, model: object, intent: WorkflowIntent
    ) -> WorkflowPlanResult:
        result = super().plan_for_model(model, intent)
        return result.model_copy(
            update={
                "readiness": WorkflowPlanReadiness(
                    ready=False,
                    blocking_issues=["Choose at least one reference image."],
                    reference_requirements=[
                        WorkflowReferenceRequirement(
                            kind="image",
                            minimum_count=1,
                            maximum_count=1,
                            description="One source image is required",
                        )
                    ],
                    allowed_output_formats=["png"],
                )
            }
        )


class WorkflowContextTests(unittest.TestCase):
    def test_plan_returns_readiness_metadata(self) -> None:
        service = WorkflowService(
            catalog=_FakeCatalog(),
            planner=_FakePlanner(),
            job_manager=_FakeJobManager(),
            input_store=_FakeInputStore(),
        )

        plan = service.plan(
            WorkflowIntent(
                model_id="z-image-turbo-local",
                prompt="cinematic portrait",
                task="image.generate",
            )
        )

        self.assertTrue(plan.readiness.ready)
        self.assertEqual(plan.readiness.warnings, ["test warning"])
        self.assertEqual(plan.readiness.allowed_output_formats, ["png"])
        self.assertEqual(len(plan.readiness.reference_requirements), 1)
        self.assertEqual(plan.readiness.reference_requirements[0].kind, "image")
        self.assertEqual(plan.presentation.primary_mode, "image")
        self.assertEqual(plan.presentation.selected_task, "image.generate")

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

    def test_run_rejects_blocked_plan_before_submit(self) -> None:
        manager = _FakeJobManager()
        service = WorkflowService(
            catalog=_FakeCatalog(),
            planner=_BlockingPlanner(),
            job_manager=manager,
            input_store=_FakeInputStore(),
        )

        with self.assertRaisesRegex(ValueError, "Choose at least one reference image."):
            service.run(
                WorkflowIntent(
                    model_id="z-image-turbo-local",
                    prompt="cinematic portrait",
                    task="image.edit",
                )
            )

        self.assertIsNone(manager.submitted_request)

    def test_plan_rejects_image_reference_with_non_image_media_type(self) -> None:
        service = WorkflowService(
            catalog=_FakeCatalog(),
            planner=_FakePlanner(),
            job_manager=_FakeJobManager(),
            input_store=_FakeInputStore(
                {
                    "inp_bad": InputHandleRecord(
                        handle_id="inp_bad",
                        role="image",
                        media_type="video/mp4",
                        filename="clip.mp4",
                        storage_key="inputs/inp_bad/clip.mp4",
                        size_bytes=3,
                    )
                }
            ),
        )

        with self.assertRaisesRegex(ValueError, "must use an image media type"):
            service.plan(
                WorkflowIntent(
                    model_id="z-image-turbo-local",
                    prompt="cinematic portrait",
                    references=[
                        WorkflowReference(kind="image", input_handle="inp_bad")
                    ],
                )
            )

    def test_run_rejects_lora_reference_without_safetensors_payload(self) -> None:
        service = WorkflowService(
            catalog=_FakeCatalog(),
            planner=_FakePlanner(),
            job_manager=_FakeJobManager(),
            input_store=_FakeInputStore(
                {
                    "inp_lora": InputHandleRecord(
                        handle_id="inp_lora",
                        role="lora",
                        media_type="application/octet-stream",
                        filename="weights.bin",
                        storage_key="inputs/inp_lora/weights.bin",
                        size_bytes=3,
                    )
                }
            ),
        )

        with self.assertRaisesRegex(ValueError, "must use a safetensors payload"):
            service.run(
                WorkflowIntent(
                    model_id="z-image-turbo-local",
                    prompt="cinematic portrait",
                    references=[
                        WorkflowReference(kind="lora", input_handle="inp_lora")
                    ],
                )
            )

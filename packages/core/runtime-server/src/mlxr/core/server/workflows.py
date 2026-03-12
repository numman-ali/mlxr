from __future__ import annotations

from dataclasses import dataclass

from mlxr.core.runtime import RuntimeCatalog
from mlxr.core.schemas import (
    InputHandleRecord,
    JobSubmitResult,
    WorkflowIntent,
    WorkflowPlanResult,
    WorkflowRunResult,
)
from mlxr.core.workflows import WorkflowPlanner

from .jobs import JobManager
from .store import InputStore


@dataclass
class WorkflowService:
    catalog: RuntimeCatalog
    planner: WorkflowPlanner
    job_manager: JobManager
    input_store: InputStore

    def plan(self, intent: WorkflowIntent) -> WorkflowPlanResult:
        self._validate_reference_bindings(intent, require_handles=False)
        model = self.catalog.get_model(intent.model_id)
        return self.planner.plan_for_model(model=model, intent=intent)

    def run(self, intent: WorkflowIntent) -> WorkflowRunResult:
        self._validate_reference_bindings(intent, require_handles=True)
        model = self.catalog.get_model(intent.model_id)
        resolved_plan = self.planner.plan_for_model(model=model, intent=intent).plan
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

    def _validate_reference_bindings(
        self, intent: WorkflowIntent, *, require_handles: bool
    ) -> None:
        for reference in intent.references:
            if reference.input_handle is None:
                if require_handles:
                    raise ValueError(
                        "Workflow execution requires bound input handles for every reference"
                    )
                continue
            record = self.input_store.get(reference.input_handle)
            if record is None:
                raise ValueError(
                    f"Unknown workflow input handle '{reference.input_handle}'"
                )
            _validate_reference_media_type(reference.kind, record)


def _validate_reference_media_type(kind: str, record: InputHandleRecord) -> None:
    media_type = record.media_type or "application/octet-stream"
    if kind == "image" and not media_type.startswith("image/"):
        raise ValueError(
            f"Workflow image reference '{record.handle_id}' must use an image media type"
        )
    if kind == "audio" and not media_type.startswith("audio/"):
        raise ValueError(
            f"Workflow audio reference '{record.handle_id}' must use an audio media type"
        )
    if kind == "video" and not media_type.startswith("video/"):
        raise ValueError(
            f"Workflow video reference '{record.handle_id}' must use a video media type"
        )
    if kind == "lora":
        filename = record.filename or ""
        if media_type == "application/x-safetensors" or filename.endswith(
            ".safetensors"
        ):
            return
        raise ValueError(
            f"Workflow LoRA reference '{record.handle_id}' must use a safetensors payload"
        )

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

from mlxr.core.schemas import (
    JobOutputPolicy,
    JobRecord,
    JobState,
    WorkflowIntent,
    WorkflowPreferences,
    WorkflowReference,
)

from .daemon import RuntimeDaemonError
from .parser import _workflow_quality
from .runtime import (
    RuntimeApiError,
    RuntimeClient,
    RuntimeUnavailableError,
    _runtime_client_for_args,
    _runtime_unavailable_message,
)

_ReferenceKind = Literal["image", "video", "audio", "lora"]


def _run_generate_entry(args: argparse.Namespace) -> int:
    try:
        with _runtime_client_for_args(args, auto_start=True) as client:
            return _run_generate_command(client, args)
    except RuntimeDaemonError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeUnavailableError as exc:
        print(_runtime_unavailable_message(args, exc), file=sys.stderr)
        return 1
    except RuntimeApiError as exc:
        print(_generate_api_error_message(args, exc), file=sys.stderr)
        return 1


def _run_generate_command(client: RuntimeClient, args: argparse.Namespace) -> int:
    references = _references_from_args(client, args)
    intent = WorkflowIntent(
        model_id=str(args.model_id),
        prompt=str(args.prompt),
        task=str(args.task) if args.task is not None else None,
        negative_prompt=(
            str(args.negative_prompt) if args.negative_prompt is not None else None
        ),
        references=references,
        params=_generation_params(args),
        preferences=WorkflowPreferences(
            quality=_workflow_quality(str(args.quality)),
        ),
        output=JobOutputPolicy(artifact_format=str(args.artifact_format)),
        extensions=_extensions_from_args(args),
    )
    if args.plan_only:
        plan = client.plan(intent)
        print(json.dumps(plan.model_dump(mode="json"), indent=2))
        return 0
    result = client.run(intent)
    payload = {
        "job_id": result.submit.job_id,
        "task": result.plan.selected_task,
        "family": result.plan.family,
        "warnings": result.plan.warnings,
    }
    if args.wait:
        terminal = _wait_for_terminal_job(
            client,
            result.submit.job_id,
            timeout_seconds=float(args.timeout_seconds),
            poll_interval_seconds=float(args.poll_interval_seconds),
        )
        payload["state"] = terminal.state.value
        payload["artifact_ids"] = [
            artifact.artifact_id for artifact in terminal.artifacts
        ]
        if args.export_path is not None and terminal.artifacts:
            export_result = client.export_output(
                terminal.artifacts[0].artifact_id,
                destination_path=Path(args.export_path).expanduser().resolve(),
                overwrite=bool(args.overwrite_export),
            )
            payload["export_path"] = export_result.destination_path
    print(json.dumps(payload, indent=2))
    if not args.wait:
        return 0
    return 0 if terminal.state == JobState.COMPLETED else 1


def _generation_params(args: argparse.Namespace) -> dict[str, int | float]:
    params: dict[str, int | float] = {}
    for key in ("width", "height", "fps", "seed"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    if args.num_frames is not None:
        params["num_frames"] = int(args.num_frames)
    if args.num_inference_steps is not None:
        params["num_inference_steps"] = int(args.num_inference_steps)
    if args.guidance_scale is not None:
        params["guidance_scale"] = float(args.guidance_scale)
    if args.window_start_seconds is not None:
        params["window_start_seconds"] = float(args.window_start_seconds)
    if args.window_end_seconds is not None:
        params["window_end_seconds"] = float(args.window_end_seconds)
    if args.no_regenerate_video:
        params["regenerate_video"] = False
    if args.no_regenerate_audio:
        params["regenerate_audio"] = False
    return params


def _references_from_args(
    client: RuntimeClient, args: argparse.Namespace
) -> list[WorkflowReference]:
    references: list[WorkflowReference] = []
    for image_path in args.image or []:
        references.append(
            _reference_from_path_or_plan(
                client=client,
                path=Path(image_path),
                kind="image",
                plan_only=bool(args.plan_only),
                metadata={
                    "frame_index": int(args.image_frame_index),
                    "strength": float(args.image_strength),
                },
            )
        )
    for keyframe in args.keyframe_image:
        references.append(
            _reference_from_path_or_plan(
                client=client,
                path=Path(keyframe.path),
                kind="image",
                plan_only=bool(args.plan_only),
                metadata={
                    "frame_index": int(keyframe.frame_index),
                    "strength": float(keyframe.strength),
                },
            )
        )
    if args.audio is not None:
        references.append(
            _reference_from_path_or_plan(
                client=client,
                path=Path(args.audio),
                kind="audio",
                plan_only=bool(args.plan_only),
                metadata={
                    "start_time_seconds": float(args.audio_start_seconds),
                    "max_duration_seconds": (
                        float(args.audio_max_duration_seconds)
                        if args.audio_max_duration_seconds is not None
                        else None
                    ),
                },
            )
        )
    for video_path in args.video or []:
        references.append(
            _reference_from_path_or_plan(
                client=client,
                path=Path(video_path),
                kind="video",
                plan_only=bool(args.plan_only),
                metadata={"strength": 1.0},
            )
        )
    for lora in args.lora:
        references.append(
            _reference_from_path_or_plan(
                client=client,
                path=Path(lora.path),
                kind="lora",
                plan_only=bool(args.plan_only),
                metadata={"strength": float(lora.strength)},
            )
        )
    return references


def _reference_from_path_or_plan(
    *,
    client: RuntimeClient,
    path: Path,
    kind: _ReferenceKind,
    plan_only: bool,
    metadata: dict[str, int | float | bool | None],
) -> WorkflowReference:
    if plan_only:
        return WorkflowReference(
            input_handle=None,
            kind=kind,
            role="reference",
            metadata=metadata,
        )
    record = client.import_file(path, kind=kind)
    return WorkflowReference(
        input_handle=record.handle_id,
        kind=kind,
        role="reference",
        metadata=metadata,
    )


def _extensions_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.extensions_json is None:
        return {}
    return _parse_extensions_json(str(args.extensions_json))


def _parse_extensions_json(raw_value: str) -> dict[str, object]:
    source = raw_value
    if raw_value.startswith("@"):
        source = Path(raw_value[1:]).expanduser().resolve().read_text("utf-8")
    payload = json.loads(source)
    if not isinstance(payload, dict):
        raise ValueError("--extensions-json must resolve to a JSON object")
    return dict(payload)


def _wait_for_terminal_job(
    client: RuntimeClient,
    job_id: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> JobRecord:
    deadline = time.monotonic() + timeout_seconds
    terminal_states = {
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.CANCELLED,
    }
    while True:
        record = client.get_job(job_id)
        if record.state in terminal_states:
            return record
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for job '{job_id}' to reach a terminal state"
            )
        time.sleep(poll_interval_seconds)


def _generate_api_error_message(
    args: argparse.Namespace, error: RuntimeApiError
) -> str:
    if error.status_code == 404 and "Unknown model" in error.detail:
        return (
            f"Model '{args.model_id}' is not installed or is unknown to the runtime.\n"
            f"Run `mlxr models install {args.model_id}` or `mlxr models list`."
        )
    if error.status_code == 403:
        return f"{error.detail}\nIf this model comes from Hugging Face, run `hf auth login`."
    return error.detail

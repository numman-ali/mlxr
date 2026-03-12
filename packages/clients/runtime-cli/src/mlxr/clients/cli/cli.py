from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, overload

import httpx
from mlxr.core.runtime import RuntimeHome
from mlxr.core.schemas import (
    ArtifactExportResult,
    InputHandleRecord,
    JobOutputPolicy,
    JobRecord,
    JobState,
    WorkflowIntent,
    WorkflowPlanResult,
    WorkflowPreferences,
    WorkflowReference,
    WorkflowRunRequest,
    WorkflowRunResult,
)

WorkflowQuality = Literal["auto", "fast", "balanced", "high"]
_NamespaceT = TypeVar("_NamespaceT")


@dataclass(frozen=True, slots=True)
class _KeyframeImageSpec:
    path: Path
    frame_index: int
    strength: float


@dataclass(frozen=True, slots=True)
class _LoraSpec:
    path: Path
    strength: float


class _KeyframeImageAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        raw_values = _coerce_action_values(
            self, values, option_string, expected="2 or 3"
        )
        if len(raw_values) not in (2, 3):
            msg = (
                f"{option_string} requires 2 or 3 arguments (PATH FRAME_IDX [STRENGTH])"
            )
            raise argparse.ArgumentError(self, msg)
        current = list(getattr(namespace, self.dest) or [])
        current.append(
            _KeyframeImageSpec(
                path=Path(raw_values[0]),
                frame_index=int(raw_values[1]),
                strength=float(raw_values[2]) if len(raw_values) == 3 else 1.0,
            )
        )
        setattr(namespace, self.dest, current)


class _LoraAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        raw_values = _coerce_action_values(
            self, values, option_string, expected="1 or 2"
        )
        if len(raw_values) not in (1, 2):
            msg = f"{option_string} requires 1 or 2 arguments (PATH [STRENGTH])"
            raise argparse.ArgumentError(self, msg)
        current = list(getattr(namespace, self.dest) or [])
        current.append(
            _LoraSpec(
                path=Path(raw_values[0]),
                strength=float(raw_values[1]) if len(raw_values) == 2 else 1.0,
            )
        )
        setattr(namespace, self.dest, current)


class _ArgumentParser(argparse.ArgumentParser):
    @overload
    def parse_args(
        self, args: Sequence[str] | None = None, namespace: None = None
    ) -> argparse.Namespace: ...

    @overload
    def parse_args(
        self, args: Sequence[str] | None, namespace: _NamespaceT
    ) -> _NamespaceT: ...

    @overload
    def parse_args(self, *, namespace: _NamespaceT) -> _NamespaceT: ...

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: _NamespaceT | None = None,
    ) -> argparse.Namespace | _NamespaceT:
        parsed = super().parse_args(args, namespace)
        if isinstance(parsed, argparse.Namespace):
            _validate_cli_args(self, parsed)
        return parsed


def _coerce_action_values(
    action: argparse.Action,
    values: str | Sequence[object] | None,
    option_string: str | None,
    *,
    expected: str,
) -> list[str]:
    if values is None:
        msg = f"{option_string} requires {expected} arguments"
        raise argparse.ArgumentError(action, msg)
    if isinstance(values, str):
        return [values]
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            msg = f"{option_string} received a non-string argument"
            raise argparse.ArgumentError(action, msg)
        normalized.append(value)
    return normalized


def _workflow_quality(value: str) -> WorkflowQuality:
    if value == "auto":
        return "auto"
    if value == "fast":
        return "fast"
    if value == "balanced":
        return "balanced"
    if value == "high":
        return "high"
    raise ValueError(f"Unsupported workflow quality: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description="First-party thin CLI for MLXR")
    parser.add_argument(
        "--runtime-url",
        default=None,
        help="Optional loopback HTTP base URL. If omitted, the CLI uses the default UDS runtime path.",
    )
    parser.add_argument(
        "--uds-path",
        type=Path,
        default=None,
        help="Optional explicit Unix-domain socket path for the runtime daemon.",
    )
    parser.add_argument(
        "--http-token",
        default=None,
        help="Optional bearer token for HTTP mode.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="Plan and run one workflow-oriented generation request"
    )
    _add_generation_arguments(generate_parser)
    generate_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the planned workflow instead of submitting it.",
    )
    generate_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the submitted job to reach a terminal state.",
    )
    generate_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for a terminal job state when --wait is used.",
    )
    generate_parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval in seconds when --wait is used.",
    )
    generate_parser.add_argument(
        "--export-path",
        type=Path,
        help="Optional trusted local export path for the first output artifact when --wait is used.",
    )
    generate_parser.add_argument(
        "--overwrite-export",
        action="store_true",
        help="Allow overwriting an existing export path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = RuntimeClient(
        base_url=str(args.runtime_url) if args.runtime_url else None,
        uds_path=Path(args.uds_path) if args.uds_path is not None else None,
        http_token=args.http_token,
    )
    if args.command == "generate":
        return _run_generate_command(client, args)
    parser.error(f"Unknown command '{args.command}'")
    return 2


class RuntimeClient:
    def __init__(
        self,
        *,
        base_url: str | None,
        uds_path: Path | None,
        http_token: str | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if http_token:
            headers["Authorization"] = f"Bearer {http_token}"
        resolved_base_url = (
            base_url.rstrip("/") if base_url else None
        ) or "http://mlxr"
        transport: httpx.BaseTransport | None = None
        if base_url is None:
            resolved_uds_path = (uds_path or _default_uds_path()).expanduser().resolve()
            transport = httpx.HTTPTransport(uds=str(resolved_uds_path))
        self._client = httpx.Client(
            base_url=resolved_base_url,
            headers=headers,
            transport=transport,
            timeout=None,
        )

    def close(self) -> None:
        self._client.close()

    def import_file(self, path: Path, *, kind: str) -> InputHandleRecord:
        media_type = _media_type_for_path(path, kind)
        response = self._client.post(
            "/v1/inputs/import-file",
            params={
                "filename": path.name,
                "media_type": media_type,
                "role": kind,
            },
            content=_file_chunks(path),
        )
        response.raise_for_status()
        return InputHandleRecord.model_validate(response.json())

    def plan(self, intent: WorkflowIntent) -> WorkflowPlanResult:
        response = self._client.post(
            "/v1/workflows/plan",
            json=intent.model_dump(mode="json"),
        )
        response.raise_for_status()
        return WorkflowPlanResult.model_validate(response.json())

    def run(self, intent: WorkflowIntent) -> WorkflowRunResult:
        response = self._client.post(
            "/v1/workflows/run",
            json=WorkflowRunRequest(intent=intent).model_dump(mode="json"),
        )
        response.raise_for_status()
        return WorkflowRunResult.model_validate(response.json())

    def get_job(self, job_id: str) -> JobRecord:
        response = self._client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        return JobRecord.model_validate(response.json())

    def export_output(
        self, artifact_id: str, *, destination_path: Path, overwrite: bool
    ) -> ArtifactExportResult:
        response = self._client.post(
            f"/v1/outputs/{artifact_id}/export",
            json={
                "destination_path": str(destination_path),
                "overwrite": overwrite,
            },
        )
        response.raise_for_status()
        return ArtifactExportResult.model_validate(response.json())


def _run_generate_command(client: RuntimeClient, args: argparse.Namespace) -> int:
    try:
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
    finally:
        client.close()


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--task",
        help="Optional explicit workflow task. If omitted, the runtime infers the task from the provided references.",
    )
    parser.add_argument(
        "--negative-prompt",
        help="Optional negative prompt passed through directly to the selected family workflow.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        help="Optional trusted local image reference. This remains the simple first-frame shorthand.",
    )
    parser.add_argument(
        "--keyframe-image",
        dest="keyframe_image",
        action=_KeyframeImageAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help="Optional keyed image reference as PATH FRAME_IDX [STRENGTH]. Repeat to provide multiple keyed images such as first and last frames.",
    )
    parser.add_argument(
        "--image-frame-index",
        type=int,
        default=0,
        help="Optional frame index for the image reference. Non-zero values use later-frame keyframe guidance.",
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for the image reference.",
    )
    parser.add_argument(
        "--audio", type=Path, help="Optional trusted local audio reference"
    )
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        help="Optional trusted local video reference. Repeat to provide multiple video refs when the selected family supports them.",
    )
    parser.add_argument(
        "--lora",
        dest="lora",
        action=_LoraAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help="Optional trusted local LoRA reference as PATH [STRENGTH]. Repeat when the selected family supports multiple LoRAs.",
    )
    parser.add_argument(
        "--audio-start-seconds",
        type=float,
        default=0.0,
        help="Optional start offset for the audio reference.",
    )
    parser.add_argument(
        "--audio-max-duration-seconds",
        type=float,
        default=None,
        help="Optional maximum duration for the audio reference.",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--window-start-seconds",
        type=float,
        help="Optional task window start in seconds for editing tasks such as video.retake.",
    )
    parser.add_argument(
        "--window-end-seconds",
        type=float,
        help="Optional task window end in seconds for editing tasks such as video.retake.",
    )
    parser.add_argument(
        "--no-regenerate-video",
        action="store_true",
        help="Editing-task option to preserve the original video content inside the selected window.",
    )
    parser.add_argument(
        "--no-regenerate-audio",
        action="store_true",
        help="Editing-task option to preserve the original audio content inside the selected window.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        help="Optional generic inference-step override for families that expose it.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="Optional generic guidance-scale override for families that expose it.",
    )
    parser.add_argument(
        "--artifact-format",
        default="mp4",
        choices=("mp4", "wav", "png", "jpg"),
    )
    parser.add_argument(
        "--quality",
        default="auto",
        choices=("auto", "fast", "balanced", "high"),
    )
    parser.add_argument(
        "--extensions-json",
        help="Optional JSON object or @path-to-json file for advanced family-specific extensions.",
    )


def _validate_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if getattr(args, "command", None) != "generate":
        return
    if bool(args.plan_only) and bool(args.wait):
        parser.error("--wait cannot be used with --plan-only")
    if args.export_path is not None and not bool(args.wait):
        parser.error("--export-path requires --wait")
    if bool(args.overwrite_export) and args.export_path is None:
        parser.error("--overwrite-export requires --export-path")
    if float(args.timeout_seconds) <= 0:
        parser.error("--timeout-seconds must be greater than 0")
    if float(args.poll_interval_seconds) < 0:
        parser.error("--poll-interval-seconds must be non-negative")
    if args.keyframe_image and (
        int(args.image_frame_index) != 0 or float(args.image_strength) != 1.0
    ):
        parser.error(
            "--keyframe-image cannot be mixed with non-default --image-frame-index "
            "or --image-strength values"
        )
    if int(args.image_frame_index) < 0:
        parser.error("--image-frame-index must be non-negative")
    if not (0.0 <= float(args.image_strength) <= 1.0):
        parser.error("--image-strength must be between 0.0 and 1.0")
    for keyframe in args.keyframe_image:
        if keyframe.frame_index < 0:
            parser.error("--keyframe-image frame index must be non-negative")
        if not (0.0 <= keyframe.strength <= 1.0):
            parser.error("--keyframe-image strength must be between 0.0 and 1.0")
    for lora in args.lora:
        if lora.strength <= 0.0:
            parser.error("--lora strength must be greater than 0.0")
    if float(args.audio_start_seconds) < 0.0:
        parser.error("--audio-start-seconds must be non-negative")
    if (
        args.audio_max_duration_seconds is not None
        and float(args.audio_max_duration_seconds) <= 0.0
    ):
        parser.error("--audio-max-duration-seconds must be greater than 0")
    if args.num_inference_steps is not None and int(args.num_inference_steps) <= 0:
        parser.error("--num-inference-steps must be greater than 0")
    if args.guidance_scale is not None and float(args.guidance_scale) < 0.0:
        parser.error("--guidance-scale must be non-negative")
    if (args.window_start_seconds is None) != (args.window_end_seconds is None):
        parser.error(
            "--window-start-seconds and --window-end-seconds must be provided together"
        )
    if args.window_start_seconds is not None and float(args.window_start_seconds) < 0.0:
        parser.error("--window-start-seconds must be non-negative")
    if args.window_end_seconds is not None and float(args.window_end_seconds) <= float(
        args.window_start_seconds
    ):
        parser.error("--window-end-seconds must be greater than --window-start-seconds")
    if args.extensions_json is not None:
        try:
            _parse_extensions_json(str(args.extensions_json))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))


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
        if args.plan_only:
            references.append(
                WorkflowReference(
                    input_handle=None,
                    kind="image",
                    role="reference",
                    metadata={
                        "frame_index": int(args.image_frame_index),
                        "strength": float(args.image_strength),
                    },
                )
            )
        else:
            record = client.import_file(Path(image_path), kind="image")
            references.append(
                WorkflowReference(
                    input_handle=record.handle_id,
                    kind="image",
                    role="reference",
                    metadata={
                        "frame_index": int(args.image_frame_index),
                        "strength": float(args.image_strength),
                    },
                )
            )
    for keyframe in args.keyframe_image:
        if args.plan_only:
            references.append(
                WorkflowReference(
                    input_handle=None,
                    kind="image",
                    role="reference",
                    metadata={
                        "frame_index": int(keyframe.frame_index),
                        "strength": float(keyframe.strength),
                    },
                )
            )
        else:
            record = client.import_file(Path(keyframe.path), kind="image")
            references.append(
                WorkflowReference(
                    input_handle=record.handle_id,
                    kind="image",
                    role="reference",
                    metadata={
                        "frame_index": int(keyframe.frame_index),
                        "strength": float(keyframe.strength),
                    },
                )
            )
    if args.audio is not None:
        if args.plan_only:
            references.append(
                WorkflowReference(
                    input_handle=None,
                    kind="audio",
                    role="reference",
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
        else:
            record = client.import_file(Path(args.audio), kind="audio")
            references.append(
                WorkflowReference(
                    input_handle=record.handle_id,
                    kind="audio",
                    role="reference",
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
        if args.plan_only:
            references.append(
                WorkflowReference(
                    input_handle=None,
                    kind="video",
                    role="reference",
                    metadata={"strength": 1.0},
                )
            )
        else:
            record = client.import_file(Path(video_path), kind="video")
            references.append(
                WorkflowReference(
                    input_handle=record.handle_id,
                    kind="video",
                    role="reference",
                    metadata={"strength": 1.0},
                )
            )
    for lora in args.lora:
        if args.plan_only:
            references.append(
                WorkflowReference(
                    input_handle=None,
                    kind="lora",
                    role="reference",
                    metadata={"strength": float(lora.strength)},
                )
            )
        else:
            record = client.import_file(Path(lora.path), kind="lora")
            references.append(
                WorkflowReference(
                    input_handle=record.handle_id,
                    kind="lora",
                    role="reference",
                    metadata={"strength": float(lora.strength)},
                )
            )
    return references


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


def _media_type_for_path(path: Path, kind: str) -> str:
    suffix = path.suffix.lower()
    if kind == "image":
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".ppm": "image/x-portable-pixmap",
            ".webp": "image/webp",
        }.get(suffix, "application/octet-stream")
    if kind == "audio":
        return {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }.get(suffix, "application/octet-stream")
    if kind == "video":
        return {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".m4v": "video/x-m4v",
            ".webm": "video/webm",
            ".avi": "video/x-msvideo",
        }.get(suffix, "application/octet-stream")
    if kind == "lora":
        return {
            ".safetensors": "application/x-safetensors",
        }.get(suffix, "application/octet-stream")
    return "application/octet-stream"


def _default_uds_path() -> Path:
    raw = os.environ.get("MLX_RUNTIME_UDS_PATH")
    if raw:
        return Path(raw)
    runtime_home = RuntimeHome.from_env()
    return runtime_home.temp_dir / "control-plane.sock"


def _file_chunks(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


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


if __name__ == "__main__":
    raise SystemExit(main())

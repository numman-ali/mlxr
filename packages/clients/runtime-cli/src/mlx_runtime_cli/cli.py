from __future__ import annotations

import argparse
import base64
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeVar, overload

import httpx
from mlx_runtime_core import RuntimeHome
from mlx_runtime_schemas import (
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
        )

    def close(self) -> None:
        self._client.close()

    def import_file(self, path: Path, *, kind: str) -> InputHandleRecord:
        media_type = _media_type_for_path(path, kind)
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        response = self._client.post(
            "/v1/inputs/import",
            json={
                "content_base64": payload,
                "media_type": media_type,
                "role": kind,
                "filename": path.name,
            },
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
            video_prompt=str(args.video_prompt) if args.video_prompt else None,
            audio_prompt=str(args.audio_prompt) if args.audio_prompt else None,
            references=references,
            params=_generation_params(args),
            preferences=WorkflowPreferences(
                natural_audio=bool(args.natural_audio),
                no_music=bool(args.no_music),
                enhance_prompt=bool(args.enhance_prompt),
                quality=_workflow_quality(str(args.quality)),
            ),
            output=JobOutputPolicy(artifact_format=str(args.artifact_format)),
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
    parser.add_argument("--video-prompt")
    parser.add_argument("--audio-prompt")
    parser.add_argument(
        "--image", type=Path, help="Optional trusted local image reference"
    )
    parser.add_argument(
        "--audio", type=Path, help="Optional trusted local audio reference"
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--artifact-format",
        default="mp4",
        choices=("mp4", "wav"),
    )
    parser.add_argument("--natural-audio", action="store_true")
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument(
        "--quality",
        default="auto",
        choices=("auto", "fast", "balanced", "high"),
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


def _generation_params(args: argparse.Namespace) -> dict[str, int]:
    params: dict[str, int] = {}
    for key in ("width", "height", "fps", "seed"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    if args.num_frames is not None:
        params["num_frames"] = int(args.num_frames)
    return params


def _references_from_args(
    client: RuntimeClient, args: argparse.Namespace
) -> list[WorkflowReference]:
    references: list[WorkflowReference] = []
    if args.image is not None:
        if args.plan_only:
            references.append(
                WorkflowReference(
                    input_handle=None,
                    kind="image",
                    role="reference",
                )
            )
        else:
            record = client.import_file(Path(args.image), kind="image")
            references.append(
                WorkflowReference(
                    input_handle=record.handle_id,
                    kind="image",
                    role="reference",
                )
            )
    if args.audio is not None:
        if args.plan_only:
            references.append(
                WorkflowReference(
                    input_handle=None,
                    kind="audio",
                    role="reference",
                )
            )
        else:
            record = client.import_file(Path(args.audio), kind="audio")
            references.append(
                WorkflowReference(
                    input_handle=record.handle_id,
                    kind="audio",
                    role="reference",
                )
            )
    return references


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
    return "application/octet-stream"


def _default_uds_path() -> Path:
    raw = os.environ.get("MLX_RUNTIME_UDS_PATH")
    if raw:
        return Path(raw)
    runtime_home = RuntimeHome.from_env()
    return runtime_home.temp_dir / "control-plane.sock"


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

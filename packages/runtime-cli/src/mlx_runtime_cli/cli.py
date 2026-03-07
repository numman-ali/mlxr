from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from mlx_runtime_core import RuntimeHome
from mlx_runtime_schemas import (
    InputHandleRecord,
    JobOutputPolicy,
    WorkflowIntent,
    WorkflowPlanResult,
    WorkflowPreferences,
    WorkflowReference,
    WorkflowRunRequest,
    WorkflowRunResult,
)

WorkflowQuality = Literal["auto", "fast", "balanced", "high"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="First-party thin CLI for MLXR")
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
                quality=cast(WorkflowQuality, args.quality),
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
        print(json.dumps(payload, indent=2))
        return 0
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


def _generation_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {}
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
            return references
        record = client.import_file(Path(args.image), kind="image")
        references.append(
            WorkflowReference(
                input_handle=record.handle_id,
                kind="image",
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
            ".webp": "image/webp",
        }.get(suffix, "application/octet-stream")
    return "application/octet-stream"


def _default_uds_path() -> Path:
    raw = os.environ.get("MLX_RUNTIME_UDS_PATH")
    if raw:
        return Path(raw)
    runtime_home = RuntimeHome.from_env()
    return runtime_home.temp_dir / "control-plane.sock"

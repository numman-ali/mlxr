from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, overload

from .generation_args import _validate_generate_args
from .helptext import _generate_help_epilog, _top_level_help_epilog

WorkflowQuality = Literal["auto", "fast", "balanced", "high"]
_NamespaceT = TypeVar("_NamespaceT")
_CONNECTION_NAMES = ("runtime_url", "uds_path", "http_token")


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
            _merge_connection_args(parsed)
            _validate_cli_args(self, parsed)
        return parsed


def _add_connection_arguments(parser: argparse.ArgumentParser, *, prefix: str) -> None:
    parser.add_argument(
        "--runtime-url",
        dest=f"{prefix}runtime_url",
        default=None,
        metavar="RUNTIME_URL",
        help="Optional loopback HTTP base URL for an explicit runtime.",
    )
    parser.add_argument(
        "--uds-path",
        dest=f"{prefix}uds_path",
        type=Path,
        default=None,
        metavar="UDS_PATH",
        help="Optional explicit Unix-domain socket path for the local runtime daemon.",
    )
    parser.add_argument(
        "--http-token",
        dest=f"{prefix}http_token",
        default=None,
        metavar="HTTP_TOKEN",
        help="Optional bearer token when --runtime-url uses HTTP mode.",
    )


def _merge_connection_args(args: argparse.Namespace) -> None:
    for name in _CONNECTION_NAMES:
        command_value = getattr(args, f"command_{name}", None)
        global_value = getattr(args, f"global_{name}", None)
        setattr(
            args,
            name,
            command_value if command_value is not None else global_value,
        )


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
    global_connection_parent = _ArgumentParser(add_help=False)
    _add_connection_arguments(global_connection_parent, prefix="global_")
    command_connection_parent = _ArgumentParser(add_help=False)
    _add_connection_arguments(command_connection_parent, prefix="command_")

    parser = _ArgumentParser(
        description="Human-first local CLI for MLXR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[global_connection_parent],
        epilog=_top_level_help_epilog(),
    )
    subparsers = parser.add_subparsers(dest="command")
    parser_map: dict[str, argparse.ArgumentParser] = {}

    generate_parser = subparsers.add_parser(
        "generate",
        help="Plan and run one workflow-oriented generation request",
        description="Plan and run one workflow-oriented generation request.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[command_connection_parent],
        epilog=_generate_help_epilog(),
    )
    parser_map["generate"] = generate_parser
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

    _build_non_generate_parsers(
        subparsers=subparsers,
        parser_map=parser_map,
        connection_parent=command_connection_parent,
    )
    setattr(parser, "_mlxr_parser_map", parser_map)
    return parser


def _build_non_generate_parsers(
    *,
    subparsers: argparse._SubParsersAction[_ArgumentParser],
    parser_map: dict[str, argparse.ArgumentParser],
    connection_parent: _ArgumentParser,
) -> None:
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start, inspect, or stop the reusable local daemon",
        description="Start, inspect, or stop the reusable local daemon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[connection_parent],
    )
    parser_map["serve"] = serve_parser
    serve_parser.add_argument(
        "--json",
        action="store_true",
        help="Print daemon status as JSON.",
    )
    serve_subparsers = serve_parser.add_subparsers(dest="serve_command")
    serve_status_parser = serve_subparsers.add_parser(
        "status",
        help="Show local daemon health and paths",
        description="Show local daemon health and paths.",
        parents=[connection_parent],
    )
    parser_map["serve status"] = serve_status_parser
    serve_status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print daemon status as JSON.",
    )
    serve_stop_parser = serve_subparsers.add_parser(
        "stop",
        help="Stop the reusable local daemon",
        description="Stop the reusable local daemon.",
        parents=[connection_parent],
    )
    parser_map["serve stop"] = serve_stop_parser
    serve_stop_parser.add_argument(
        "--json",
        action="store_true",
        help="Print daemon status as JSON.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check runtime home, daemon health, auth, and install readiness",
        description="Check runtime home, daemon health, auth, and install readiness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[connection_parent],
    )
    parser_map["doctor"] = doctor_parser
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Print doctor results as JSON.",
    )

    models_parser = subparsers.add_parser(
        "models",
        help="List and install supported MLXR model ids",
        description="List and install supported MLXR model ids.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[connection_parent],
    )
    parser_map["models"] = models_parser
    models_parser.add_argument(
        "--json",
        action="store_true",
        help="Print model information as JSON.",
    )
    models_subparsers = models_parser.add_subparsers(dest="models_command")
    models_list_parser = models_subparsers.add_parser(
        "list",
        help="List recommended and advanced supported model ids",
        description="List recommended and advanced supported model ids.",
        parents=[connection_parent],
    )
    parser_map["models list"] = models_list_parser
    models_list_parser.add_argument(
        "--json",
        action="store_true",
        help="Print model information as JSON.",
    )
    models_install_parser = models_subparsers.add_parser(
        "install",
        help="Install one supported model id",
        description="Install one supported model id.",
        parents=[connection_parent],
    )
    parser_map["models install"] = models_install_parser
    models_install_parser.add_argument("model_id")
    models_install_parser.add_argument(
        "--json",
        action="store_true",
        help="Print install result as JSON.",
    )

    feedback_parser = subparsers.add_parser(
        "feedback",
        help="Open or print the GitHub issues page",
        description="Open or print the GitHub issues page.",
    )
    parser_map["feedback"] = feedback_parser
    feedback_parser.add_argument(
        "--print-url",
        action="store_true",
        help="Print the issues URL instead of opening a browser.",
    )

    help_parser = subparsers.add_parser(
        "help",
        help="Show help for the CLI or one command",
        description="Show help for the CLI or one command.",
    )
    parser_map["help"] = help_parser
    help_parser.add_argument("command_path", nargs="*")


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
    parser.add_argument("--audio-start-seconds", type=float, default=0.0)
    parser.add_argument("--audio-max-duration-seconds", type=float, default=None)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--window-start-seconds", type=float)
    parser.add_argument("--window-end-seconds", type=float)
    parser.add_argument("--no-regenerate-video", action="store_true")
    parser.add_argument("--no-regenerate-audio", action="store_true")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
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
    parser.add_argument("--extensions-json")


def _validate_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    runtime_url = getattr(args, "runtime_url", None)
    uds_path = getattr(args, "uds_path", None)
    http_token = getattr(args, "http_token", None)
    command = getattr(args, "command", None)
    if runtime_url is not None and uds_path is not None:
        parser.error("--runtime-url and --uds-path cannot be used together")
    if http_token is not None and runtime_url is None:
        parser.error("--http-token requires --runtime-url")
    if command == "serve" and runtime_url is not None:
        parser.error(
            "`mlxr serve` manages the local daemon and does not accept --runtime-url"
        )
    if command == "generate":
        _validate_generate_args(parser, args)

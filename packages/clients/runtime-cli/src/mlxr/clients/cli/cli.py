from __future__ import annotations

import sys

from .commands import (
    _models_list_payload,
    _run_doctor_command,
    _run_feedback_command,
    _run_help_command,
    _run_models_command,
    _run_serve_command,
)
from .generation import (
    _extensions_from_args,
    _generation_params,
    _parse_extensions_json,
    _references_from_args,
    _run_generate_command,
    _run_generate_entry,
    _wait_for_terminal_job,
)
from .helptext import _top_level_brief_help
from .io_utils import _default_uds_path, _file_chunks, _media_type_for_path
from .parser import (
    WorkflowQuality,
    _coerce_action_values,
    _workflow_quality,
    build_parser,
)
from .runtime import RuntimeClient, _runtime_client_for_args

__all__ = [
    "RuntimeClient",
    "WorkflowQuality",
    "build_parser",
    "main",
    "_coerce_action_values",
    "_default_uds_path",
    "_extensions_from_args",
    "_file_chunks",
    "_generation_params",
    "_media_type_for_path",
    "_models_list_payload",
    "_parse_extensions_json",
    "_references_from_args",
    "_run_generate_command",
    "_runtime_client_for_args",
    "_wait_for_terminal_job",
    "_workflow_quality",
]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list:
        print(_top_level_brief_help(), end="")
        return 0
    args = parser.parse_args(argv_list)
    if getattr(args, "command", None) is None:
        print(_top_level_brief_help(), end="")
        return 0
    if args.command == "help":
        return _run_help_command(parser, args)
    if args.command == "generate":
        return _run_generate_entry(args)
    if args.command == "serve":
        return _run_serve_command(args)
    if args.command == "doctor":
        return _run_doctor_command(args)
    if args.command == "models":
        return _run_models_command(args)
    if args.command == "feedback":
        return _run_feedback_command(args)
    parser.error(f"Unknown command '{args.command}'")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

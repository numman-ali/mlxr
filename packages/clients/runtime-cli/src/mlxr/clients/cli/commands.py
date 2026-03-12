from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import asdict, dataclass

from huggingface_hub import get_token
from mlxr.core.runtime import RuntimeHome
from mlxr.core.schemas import ModelRecord, SupportedModelDescriptor

from .constants import DEFAULT_STARTUP_TIMEOUT_SECONDS, DOCS_URL, FEEDBACK_URL
from .daemon import (
    DaemonStatus,
    RuntimeDaemonError,
    daemon_socket_path,
    daemon_status,
    ensure_runtime_daemon,
    stop_runtime_daemon,
)
from .runtime import (
    RuntimeApiError,
    RuntimeUnavailableError,
    _runtime_client_for_args,
    _runtime_connection_options_from_args,
    _runtime_unavailable_message,
)


@dataclass(frozen=True, slots=True)
class _DoctorCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class _ModelListEntry:
    model_id: str
    display_name: str
    family: str
    support_level: str
    tasks: tuple[str, ...]
    installed: bool

    def to_json(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "family": self.family,
            "support_level": self.support_level,
            "tasks": list(self.tasks),
            "installed": self.installed,
        }


@dataclass(frozen=True, slots=True)
class _ModelListPayload:
    recommended: tuple[_ModelListEntry, ...]
    advanced: tuple[_ModelListEntry, ...]
    installed_local: tuple[_ModelListEntry, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "recommended": [entry.to_json() for entry in self.recommended],
            "advanced": [entry.to_json() for entry in self.advanced],
            "installed_local": [entry.to_json() for entry in self.installed_local],
        }


def _run_help_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    parser_map = getattr(parser, "_mlxr_parser_map")
    command_path = " ".join(str(part) for part in args.command_path)
    if not command_path:
        parser.print_help()
        return 0
    target = parser_map.get(command_path)
    if target is None:
        print(
            f"Unknown help target '{command_path}'. See `mlxr --help`.",
            file=sys.stderr,
        )
        return 2
    target.print_help()
    return 0


def _run_serve_command(args: argparse.Namespace) -> int:
    runtime_home = RuntimeHome.from_env()
    socket_path = daemon_socket_path(
        runtime_home, _runtime_connection_options_from_args(args).uds_path
    )
    if args.serve_command == "status":
        status = daemon_status(runtime_home=runtime_home, socket_path=socket_path)
        _print_status_payload(status, json_output=bool(args.json))
        return 0 if status.healthy else 1
    if args.serve_command == "stop":
        status = stop_runtime_daemon(runtime_home=runtime_home, socket_path=socket_path)
        _print_status_payload(status, json_output=bool(args.json))
        return 0 if status.status == "stopped" else 1
    try:
        status = ensure_runtime_daemon(
            runtime_home=runtime_home,
            socket_path=socket_path,
            startup_timeout_seconds=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        )
    except RuntimeDaemonError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_status_payload(status, json_output=bool(args.json))
    return 0


def _run_doctor_command(args: argparse.Namespace) -> int:
    runtime_home = RuntimeHome.from_env()
    checks: list[_DoctorCheck] = []
    try:
        runtime_home.ensure_layout()
        checks.append(_DoctorCheck("runtime home", "ok", str(runtime_home.root)))
    except OSError as exc:
        checks.append(_DoctorCheck("runtime home", "fail", str(exc)))
        _print_doctor_checks(
            checks,
            json_output=bool(args.json),
            installed_models=None,
            supported_models=None,
        )
        return 1

    if args.runtime_url is None:
        socket_path = daemon_socket_path(
            runtime_home, _runtime_connection_options_from_args(args).uds_path
        )
        status = daemon_status(runtime_home=runtime_home, socket_path=socket_path)
        daemon_level = "ok" if status.healthy else "warn"
        checks.append(
            _DoctorCheck("daemon", daemon_level, status.message or status.status)
        )
        installed_models: int | None = None
        supported_models: int | None = None
        if status.healthy:
            try:
                with _runtime_client_for_args(args, auto_start=False) as client:
                    installed_models = len(client.list_models())
                    supported_models = len(client.list_supported_models())
            except (RuntimeApiError, RuntimeUnavailableError):
                checks.append(
                    _DoctorCheck(
                        "runtime api",
                        "warn",
                        "Daemon is running but model introspection failed.",
                    )
                )
    else:
        installed_models = None
        supported_models = None
        try:
            with _runtime_client_for_args(args, auto_start=False) as client:
                client.health()
                checks.append(
                    _DoctorCheck("runtime", "ok", f"reachable at {args.runtime_url}")
                )
                installed_models = len(client.list_models())
                supported_models = len(client.list_supported_models())
        except (RuntimeApiError, RuntimeUnavailableError) as exc:
            checks.append(
                _DoctorCheck("runtime", "warn", _runtime_unavailable_message(args, exc))
            )

    token = get_token()
    if token:
        checks.append(
            _DoctorCheck("huggingface auth", "ok", "Local auth token is available.")
        )
    else:
        checks.append(
            _DoctorCheck(
                "huggingface auth",
                "warn",
                "No local Hugging Face auth token found. Run `hf auth login` for gated models.",
            )
        )

    _print_doctor_checks(
        checks,
        json_output=bool(args.json),
        installed_models=installed_models,
        supported_models=supported_models,
    )
    return 1 if any(check.status == "fail" for check in checks) else 0


def _run_models_command(args: argparse.Namespace) -> int:
    command = args.models_command or "list"
    try:
        with _runtime_client_for_args(args, auto_start=True) as client:
            if command == "install":
                result = client.install_model(str(args.model_id))
                if args.json:
                    print(json.dumps(result.model_dump(mode="json"), indent=2))
                else:
                    action = (
                        "Installed"
                        if result.status == "installed"
                        else "Already installed"
                    )
                    print(
                        f"{action} {result.model.model_id} ({result.supported_model.display_name})."
                    )
                    if result.model.artifact is not None:
                        print(f"Artifact: {result.model.artifact.artifact_digest}")
                    print(f"Tasks: {', '.join(result.supported_model.tasks)}")
                return 0

            payload = _models_list_payload(
                client.list_supported_models(),
                client.list_models(),
            )
            if args.json:
                print(json.dumps(payload.to_json(), indent=2))
            else:
                _print_models_payload(payload)
            return 0
    except RuntimeDaemonError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeUnavailableError as exc:
        print(_runtime_unavailable_message(args, exc), file=sys.stderr)
        return 1
    except RuntimeApiError as exc:
        message = exc.detail
        if exc.status_code == 404 and command == "install":
            message = (
                f"Unknown supported model '{args.model_id}'. "
                "Run `mlxr models list` to see recommended model ids."
            )
        elif exc.status_code == 403:
            message = f"{exc.detail}\nIf this model comes from Hugging Face, run `hf auth login`."
        print(message, file=sys.stderr)
        return 1


def _run_feedback_command(args: argparse.Namespace) -> int:
    if bool(args.print_url):
        print(FEEDBACK_URL)
        return 0
    try:
        opened = webbrowser.open(FEEDBACK_URL)
    except (OSError, webbrowser.Error):
        opened = False
    if opened:
        print(f"Opened {FEEDBACK_URL}")
        return 0
    print(FEEDBACK_URL)
    return 0


def _print_status_payload(status: DaemonStatus, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(asdict(status), indent=2))
        return
    print(f"Status: {status.status}")
    print(f"Managed: {'yes' if status.managed else 'no'}")
    print(f"Healthy: {'yes' if status.healthy else 'no'}")
    print(f"Runtime home: {status.runtime_home}")
    print(f"Socket: {status.socket_path}")
    print(f"Log: {status.stdio_log_path}")
    if status.pid is not None:
        print(f"PID: {status.pid}")
    if status.message:
        print(status.message)


def _print_doctor_checks(
    checks: list[_DoctorCheck],
    *,
    json_output: bool,
    installed_models: int | None,
    supported_models: int | None,
) -> None:
    if json_output:
        payload = {
            "checks": [asdict(check) for check in checks],
            "installed_models": installed_models,
            "supported_models": supported_models,
            "docs_url": DOCS_URL,
            "feedback_url": FEEDBACK_URL,
        }
        print(json.dumps(payload, indent=2))
        return
    print("MLXR doctor")
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")
    if installed_models is not None:
        print(f"Installed models: {installed_models}")
    if supported_models is not None:
        print(f"Supported model entries: {supported_models}")
    print(f"Docs: {DOCS_URL}")
    print(f"Feedback: {FEEDBACK_URL}")


def _models_list_payload(
    supported_models: list[SupportedModelDescriptor],
    installed_models: list[ModelRecord],
) -> _ModelListPayload:
    recommended: list[_ModelListEntry] = []
    advanced: list[_ModelListEntry] = []
    for descriptor in supported_models:
        entry = _ModelListEntry(
            model_id=descriptor.model_id,
            display_name=descriptor.display_name,
            family=descriptor.family,
            support_level=descriptor.support_level,
            tasks=tuple(descriptor.tasks),
            installed=descriptor.installed,
        )
        if descriptor.recommendation_tier == "recommended":
            recommended.append(entry)
        else:
            advanced.append(entry)

    supported_model_ids = {descriptor.model_id for descriptor in supported_models}
    installed_local: list[_ModelListEntry] = []
    for record in installed_models:
        if record.model_id in supported_model_ids:
            continue
        tasks = tuple(record.capability.tasks) if record.capability is not None else ()
        installed_local.append(
            _ModelListEntry(
                model_id=record.model_id,
                display_name=record.model_id,
                family=record.family,
                support_level="supported",
                tasks=tasks,
                installed=True,
            )
        )

    return _ModelListPayload(
        recommended=tuple(recommended),
        advanced=tuple(advanced),
        installed_local=tuple(installed_local),
    )


def _print_models_payload(payload: _ModelListPayload) -> None:
    print("Recommended models")
    for entry in payload.recommended:
        print(
            f"- {entry.model_id} ({entry.display_name}) | family={entry.family} | support={entry.support_level} | installed={'yes' if entry.installed else 'no'} | tasks={', '.join(entry.tasks)}"
        )
    if payload.advanced:
        print("\nSupported advanced models")
        for entry in payload.advanced:
            print(
                f"- {entry.model_id} ({entry.display_name}) | family={entry.family} | support={entry.support_level} | installed={'yes' if entry.installed else 'no'} | tasks={', '.join(entry.tasks)}"
            )
    if payload.installed_local:
        print("\nInstalled local models")
        for entry in payload.installed_local:
            print(
                f"- {entry.model_id} ({entry.display_name}) | family={entry.family} | tasks={', '.join(entry.tasks)}"
            )
    print("\nInstall one with: mlxr models install <model_id>")

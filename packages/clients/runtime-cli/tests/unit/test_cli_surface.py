from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mlxr.clients.cli.cli import (
    _models_list_payload,
    _runtime_client_for_args,
    build_parser,
    main,
)
from mlxr.clients.cli.daemon import DaemonStatus
from mlxr.core.runtime import RuntimeHome
from mlxr.core.schemas import (
    ModelInstallResult,
    ModelRecord,
    SupportedModelDescriptor,
)


def _supported_model(
    *,
    model_id: str,
    display_name: str,
    family: str,
    recommendation_tier: str,
    support_level: str,
    tasks: list[str],
    installed: bool = False,
) -> SupportedModelDescriptor:
    return SupportedModelDescriptor(
        model_id=model_id,
        display_name=display_name,
        family=family,
        family_variant=None,
        recommendation_tier=recommendation_tier,
        support_level=support_level,
        tasks=tasks,
        provider="huggingface",
        source_summary=model_id,
        license="other",
        access_state="public",
        installed=installed,
        installable=True,
    )


def _model_record(model_id: str, family: str) -> ModelRecord:
    return ModelRecord(model_id=model_id, family=family)


class _RecordingRuntimeClient:
    def __init__(
        self,
        *,
        base_url: str | None,
        uds_path: Path | None,
        http_token: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.uds_path = uds_path
        self.http_token = http_token
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ModelsClient:
    def __init__(
        self,
        *,
        supported_models: list[SupportedModelDescriptor] | None = None,
        installed_models: list[ModelRecord] | None = None,
        install_result: ModelInstallResult | None = None,
    ) -> None:
        self.supported_models = supported_models or []
        self.installed_models = installed_models or []
        self.install_result = install_result
        self.install_calls: list[str] = []

    def close(self) -> None:
        return None

    def list_supported_models(self) -> list[SupportedModelDescriptor]:
        return self.supported_models

    def list_models(self) -> list[ModelRecord]:
        return self.installed_models

    def install_model(self, model_id: str) -> ModelInstallResult:
        self.install_calls.append(model_id)
        if self.install_result is None:
            raise AssertionError("install_result must be provided for install tests")
        return self.install_result


class RuntimeCliSurfaceTests(unittest.TestCase):
    def test_main_without_args_prints_brief_help(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main([])
        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("MLXR is a local-first MLX runtime", output)
        self.assertIn("mlxr models list", output)

    def test_help_command_prints_nested_subcommand_help(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["help", "models", "install"])
        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("Install one supported model id", output)
        self.assertIn("--json", output)

    def test_parser_accepts_transport_flags_before_and_after_subcommand(self) -> None:
        parser = build_parser()
        before = parser.parse_args(
            [
                "--uds-path",
                "/tmp/mlxr-before.sock",
                "models",
                "list",
            ]
        )
        after = parser.parse_args(
            [
                "models",
                "list",
                "--uds-path",
                "/tmp/mlxr-after.sock",
            ]
        )
        self.assertEqual(before.command, "models")
        self.assertEqual(before.models_command, "list")
        self.assertEqual(before.uds_path, Path("/tmp/mlxr-before.sock"))
        self.assertEqual(after.command, "models")
        self.assertEqual(after.models_command, "list")
        self.assertEqual(after.uds_path, Path("/tmp/mlxr-after.sock"))

    def test_models_list_payload_includes_supported_advanced_catalog_rows(self) -> None:
        payload = _models_list_payload(
            [
                _supported_model(
                    model_id="ltx-2.3-fast-local",
                    display_name="LTX 2.3 Fast",
                    family="ltx",
                    recommendation_tier="recommended",
                    support_level="promoted",
                    tasks=["video.generate"],
                    installed=True,
                ),
                _supported_model(
                    model_id="qwen-image-edit-local",
                    display_name="Qwen-Image Edit 2511",
                    family="qwen_image",
                    recommendation_tier="advanced",
                    support_level="supported",
                    tasks=["image.edit"],
                ),
            ],
            [
                _model_record("ltx-2.3-fast-local", "ltx"),
                _model_record("local-experimental-image", "z_image"),
            ],
        )
        self.assertEqual(
            [entry.model_id for entry in payload.recommended],
            ["ltx-2.3-fast-local"],
        )
        self.assertEqual(
            [entry.model_id for entry in payload.advanced],
            ["qwen-image-edit-local"],
        )
        self.assertEqual(
            [entry.model_id for entry in payload.installed_local],
            ["local-experimental-image"],
        )

    def test_models_list_command_prints_recommended_and_advanced_sections(self) -> None:
        client = _ModelsClient(
            supported_models=[
                _supported_model(
                    model_id="ltx-2.3-fast-local",
                    display_name="LTX 2.3 Fast",
                    family="ltx",
                    recommendation_tier="recommended",
                    support_level="promoted",
                    tasks=["video.generate"],
                ),
                _supported_model(
                    model_id="qwen-image-edit-local",
                    display_name="Qwen-Image Edit 2511",
                    family="qwen_image",
                    recommendation_tier="advanced",
                    support_level="supported",
                    tasks=["image.edit"],
                ),
            ],
            installed_models=[_model_record("local-experimental-image", "z_image")],
        )
        stdout = io.StringIO()
        with patch(
            "mlxr.clients.cli.commands._runtime_client_for_args",
            return_value=nullcontext(client),
        ):
            with redirect_stdout(stdout):
                exit_code = main(["models", "list"])
        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("Recommended models", output)
        self.assertIn("Supported advanced models", output)
        self.assertIn("qwen-image-edit-local", output)
        self.assertIn("Installed local models", output)

    def test_models_install_command_prints_friendly_summary(self) -> None:
        supported_model = _supported_model(
            model_id="flux2-klein-9b-local",
            display_name="FLUX.2 Klein 9B",
            family="flux2",
            recommendation_tier="recommended",
            support_level="promoted",
            tasks=["image.generate", "image.edit"],
            installed=True,
        )
        install_result = ModelInstallResult(
            status="installed",
            model=_model_record("flux2-klein-9b-local", "flux2"),
            supported_model=supported_model,
        )
        client = _ModelsClient(install_result=install_result)
        stdout = io.StringIO()
        with patch(
            "mlxr.clients.cli.commands._runtime_client_for_args",
            return_value=nullcontext(client),
        ):
            with redirect_stdout(stdout):
                exit_code = main(["models", "install", "flux2-klein-9b-local"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(client.install_calls, ["flux2-klein-9b-local"])
        output = stdout.getvalue()
        self.assertIn("Installed flux2-klein-9b-local", output)
        self.assertIn("Tasks: image.generate, image.edit", output)

    def test_doctor_json_reports_local_runtime_and_auth_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            socket_path = runtime_home.temp_dir / "control-plane.sock"
            status = DaemonStatus(
                status="stopped",
                runtime_home=str(runtime_home.root),
                socket_path=str(socket_path),
                stdio_log_path=str(runtime_home.logs_dir / "control-plane-stdio.log"),
                metadata_path=str(runtime_home.temp_dir / "control-plane-daemon.json"),
                managed=False,
                healthy=False,
                message="No healthy local daemon is running.",
            )
            stdout = io.StringIO()
            with (
                patch(
                    "mlxr.clients.cli.commands.RuntimeHome.from_env",
                    return_value=runtime_home,
                ),
                patch(
                    "mlxr.clients.cli.commands.daemon_status",
                    return_value=status,
                ),
                patch(
                    "mlxr.clients.cli.commands.get_token",
                    return_value=None,
                ),
            ):
                with redirect_stdout(stdout):
                    exit_code = main(["doctor", "--json"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["installed_models"], None)
        self.assertEqual(payload["supported_models"], None)
        self.assertEqual(payload["checks"][0]["name"], "runtime home")
        self.assertEqual(payload["checks"][0]["status"], "ok")
        self.assertEqual(payload["checks"][1]["name"], "daemon")
        self.assertEqual(payload["checks"][1]["status"], "warn")
        self.assertEqual(payload["checks"][2]["name"], "huggingface auth")
        self.assertEqual(payload["checks"][2]["status"], "warn")

    def test_feedback_falls_back_to_printing_url(self) -> None:
        stdout = io.StringIO()
        with patch("mlxr.clients.cli.commands.webbrowser.open", return_value=False):
            with redirect_stdout(stdout):
                exit_code = main(["feedback"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue().strip(),
            "https://github.com/numman-ali/mlxr/issues/new/choose",
        )

    def test_runtime_client_for_args_autostart_prints_notice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            socket_path = (runtime_home.temp_dir / "control-plane.sock").resolve()
            started_status = DaemonStatus(
                status="started",
                runtime_home=str(runtime_home.root),
                socket_path=str(socket_path),
                stdio_log_path=str(runtime_home.logs_dir / "control-plane-stdio.log"),
                metadata_path=str(runtime_home.temp_dir / "control-plane-daemon.json"),
                managed=True,
                healthy=True,
                pid=12345,
                message="Started a reusable local daemon.",
            )
            args = argparse.Namespace(
                runtime_url=None,
                uds_path=None,
                http_token=None,
            )
            stderr = io.StringIO()
            with (
                patch(
                    "mlxr.clients.cli.runtime.RuntimeHome.from_env",
                    return_value=runtime_home,
                ),
                patch(
                    "mlxr.clients.cli.runtime.ensure_runtime_daemon",
                    return_value=started_status,
                ),
                patch(
                    "mlxr.clients.cli.runtime.RuntimeClient",
                    _RecordingRuntimeClient,
                ),
            ):
                with redirect_stderr(stderr):
                    with _runtime_client_for_args(args, auto_start=True) as client:
                        self.assertIsInstance(client, _RecordingRuntimeClient)
                        self.assertEqual(client.uds_path, socket_path)
                    self.assertTrue(client.closed)
        self.assertIn("Started MLXR runtime at", stderr.getvalue())

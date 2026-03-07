from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import mlx_runtime_cli.benchmark_ltx as benchmark_ltx
from fastapi.testclient import TestClient
from mlx_runtime_cli.benchmark_ltx import (
    DEFAULT_CONDITIONING_IMAGE,
    BenchmarkEnvironment,
    ConversionStepResult,
    InputImportStepResult,
    InspectStepResult,
    JobRunResult,
    RegisterStepResult,
    RuntimeApi,
    ScenarioBenchmarkResult,
    SourceSetupResult,
    _classify_job_failure,
    _classify_route_failure,
    _source_entries_for_mode,
    _unavailable_metrics,
    build_default_scenarios,
    run_scenario_with_api,
    write_benchmark_result,
)
from mlx_runtime_schemas import (
    ArtifactConversionRequest,
    JobRecord,
    JobState,
    RuntimeEvent,
    RuntimeEventKind,
    SourceRef,
)
from mlx_runtime_server.app import create_app

from tests.runtime_test_support import (
    make_local_bundle,
    make_state,
    patched_inline_job_process_context,
    patched_ltx_prompt_encoder,
    patched_ltx_video_generator,
)


class LTXBenchmarkHarnessTests(unittest.TestCase):
    def test_t2v_scenario_records_cold_and_warm_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            scenario = next(
                scenario
                for scenario in build_default_scenarios()
                if scenario.scenario_id == "t2v"
            )
            result_path = root / "results" / "t2v.json"

            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                patched_ltx_video_generator(),
                TestClient(create_app(state)) as client,
            ):
                result = run_scenario_with_api(
                    RuntimeApi(client.request, supports_timeout=False),
                    scenario=scenario,
                    source_mode="local-bundle",
                    runtime_home=state.runtime_home.root,
                    scenario_output_dir=root / "benchmarks" / "t2v",
                    local_bundle_path=source_dir,
                )

            self.assertEqual(result.status, "completed")
            self.assertIsNotNone(result.source_setup.conversion)
            assert result.source_setup.conversion is not None
            self.assertIsNotNone(result.source_setup.conversion.timings_ms)
            self.assertEqual(set(result.source_bindings), {"bundle"})
            self.assertEqual(len(result.runs), 2)

            cold_run = result.runs[0]
            warm_run = result.runs[1]
            self.assertEqual(cold_run.label, "cold")
            self.assertEqual(warm_run.label, "warm")
            self.assertEqual(
                cold_run.generation_backend, "ltx_test_distilled_generator"
            )
            self.assertTrue(
                any(metric.stage_id == "generate" for metric in cold_run.stage_metrics)
            )
            self.assertIsNotNone(cold_run.downloaded_path)
            assert cold_run.downloaded_path is not None
            self.assertIn(b"ftyp", Path(cold_run.downloaded_path).read_bytes()[:32])

            write_benchmark_result(result, result_path)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            unavailable_names = {
                item["name"] for item in payload["unavailable_metrics"]
            }
            self.assertIn("compile_count", unavailable_names)

    def test_i2v_scenario_imports_conditioning_and_records_stage_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            scenario = next(
                scenario
                for scenario in build_default_scenarios(
                    conditioning_image=DEFAULT_CONDITIONING_IMAGE
                )
                if scenario.scenario_id == "i2v"
            )

            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                patched_ltx_video_generator(),
                TestClient(create_app(state)) as client,
            ):
                result = run_scenario_with_api(
                    RuntimeApi(client.request, supports_timeout=False),
                    scenario=scenario,
                    source_mode="local-bundle",
                    runtime_home=state.runtime_home.root,
                    scenario_output_dir=root / "benchmarks" / "i2v",
                    local_bundle_path=source_dir,
                )

            self.assertEqual(result.status, "completed")
            self.assertIsNotNone(result.source_setup.input_import)
            assert result.source_setup.input_import is not None
            self.assertIsNotNone(result.source_setup.input_import.handle_id)
            self.assertTrue(
                any(
                    metric.stage_id == "condition_inputs"
                    for metric in result.runs[0].stage_metrics
                )
            )
            self.assertTrue(
                any(
                    metric.stage_id == "generate"
                    and metric.metrics.get("conditioning_count") == 1
                    for metric in result.runs[0].stage_metrics
                )
            )

    def test_source_modes_failure_mapping_and_unavailable_metrics_are_explicit(
        self,
    ) -> None:
        local_entries = _source_entries_for_mode(
            source_mode="local-bundle",
            local_bundle_path=Path("/tmp/example-bundle"),
        )
        self.assertEqual([entry.role for entry in local_entries], ["bundle"])

        hf_entries = _source_entries_for_mode(
            source_mode="huggingface",
            local_bundle_path=None,
        )
        self.assertEqual(
            [entry.role for entry in hf_entries],
            ["checkpoint", "spatial_upsampler", "text_encoder"],
        )
        self.assertTrue(
            all(entry.source_ref.auth.token_ref == "hf-default" for entry in hf_entries)
        )

        socket_path = benchmark_ltx._benchmark_socket_path(
            session_label="20260306T230000Z",
            scenario_id="t2v",
        )
        self.assertLess(len(str(socket_path)), 100)

        hf_env = benchmark_ltx._huggingface_runtime_env(
            scenario_output_dir=Path("/tmp/benchmarks/output/session/t2v")
        )
        self.assertIn("HF_HUB_CACHE", hf_env)
        self.assertIn("HF_XET_CACHE", hf_env)
        self.assertNotIn("HF_HOME", hf_env)

        unavailable_names = {metric.name for metric in _unavailable_metrics()}
        self.assertIn("compile_count", unavailable_names)
        self.assertIn("model_switch", unavailable_names)

        self.assertEqual(
            _classify_route_failure(
                step="submit",
                status_code=409,
                error="A heavy job is already running",
            ),
            "admission_rejection",
        )
        self.assertEqual(
            _classify_job_failure(
                events=[
                    RuntimeEvent(
                        job_id="job_1",
                        kind=RuntimeEventKind.JOB_PHASE_CHANGED,
                        phase="loading_model",
                        data={"stage_id": "load_model"},
                    )
                ],
                error="load failed",
            ),
            "load_failure",
        )

    def test_runtime_api_failure_paths_and_helper_branches(self) -> None:
        captured_calls: list[dict[str, object]] = []

        def requester(method: str, path: str, **kwargs: object) -> httpx.Response:
            captured_calls.append({"method": method, "path": path, **kwargs})
            return httpx.Response(200, json={"ok": True})

        api = RuntimeApi(requester)
        api.request("POST", "/v1/demo", json_payload={"prompt": "hi"}, timeout=3.0)
        self.assertEqual(captured_calls[0]["json"], {"prompt": "hi"})
        self.assertEqual(captured_calls[0]["timeout"], 3.0)

        no_timeout_api = RuntimeApi(requester, supports_timeout=False)
        no_timeout_api.request("GET", "/v1/demo", timeout=4.0)
        self.assertNotIn("timeout", captured_calls[1])

        parsed_events = benchmark_ltx._parse_sse_events(
            "\n".join(
                (
                    "event: job.metrics",
                    'data: {"job_id":"job_1","kind":"job.metrics","phase":"running","data":{"stage_id":"generate","duration_ms":12.5,"memory":{"telemetry_available":true},"metrics":{"backend":"preview"}}}',
                    "",
                    "event: job.completed",
                    'data: {"job_id":"job_1","kind":"job.completed","phase":"completed","data":{}}',
                )
            )
        )
        self.assertEqual(len(parsed_events), 2)
        stage_metrics = benchmark_ltx._stage_metrics_from_events(parsed_events)
        self.assertEqual(len(stage_metrics), 1)
        self.assertEqual(
            benchmark_ltx._generation_backend_from_stage_metrics(stage_metrics),
            "preview",
        )
        self.assertIsNone(benchmark_ltx._generation_backend_from_stage_metrics([]))
        self.assertIsNone(benchmark_ltx._last_stage_id([]))

        detail_response = httpx.Response(400, json={"detail": "bad request"})
        self.assertEqual(
            benchmark_ltx._response_error_text(detail_response), "bad request"
        )
        text_response = httpx.Response(500, text="plain failure")
        self.assertEqual(
            benchmark_ltx._response_error_text(text_response), "plain failure"
        )

        self.assertEqual(
            _classify_route_failure(
                step="inspect",
                status_code=403,
                error="repo requires authenticated access",
            ),
            "auth_or_access_failure",
        )
        self.assertEqual(
            _classify_route_failure(
                step="inspect", status_code=400, error="remote code approval missing"
            ),
            "remote_code_policy_failure",
        )
        self.assertEqual(
            _classify_route_failure(
                step="convert", status_code=400, error="bad convert"
            ),
            "conversion_failure",
        )
        self.assertEqual(
            _classify_route_failure(step="download", status_code=500, error="io"),
            "output_encode_or_export_failure",
        )
        self.assertEqual(
            _classify_job_failure(events=[], error="OOM while generating"),
            "runtime_oom_or_memory_pressure",
        )
        self.assertEqual(
            _classify_job_failure(
                events=[
                    RuntimeEvent(
                        job_id="job_2",
                        kind=RuntimeEventKind.JOB_PHASE_CHANGED,
                        phase="streaming_output",
                        data={"stage_id": "encode_output"},
                    )
                ],
                error="ffmpeg failed",
            ),
            "output_encode_or_export_failure",
        )

        with patch(
            "mlx_runtime_cli.benchmark_ltx.subprocess.run",
            return_value=SimpleNamespace(stdout="12345\n"),
        ):
            self.assertEqual(benchmark_ltx._system_memory_bytes(), 12345)
        with patch(
            "mlx_runtime_cli.benchmark_ltx.subprocess.run",
            side_effect=FileNotFoundError("sysctl missing"),
        ):
            self.assertIsNone(benchmark_ltx._system_memory_bytes())
        with patch(
            "mlx_runtime_cli.benchmark_ltx.importlib.metadata.version",
            side_effect=PackageNotFoundError,
        ):
            self.assertIsNone(benchmark_ltx._package_version("missing-package"))

    def test_run_scenario_surfaces_each_failure_boundary(self) -> None:
        scenario = next(
            scenario
            for scenario in build_default_scenarios(
                conditioning_image=DEFAULT_CONDITIONING_IMAGE
            )
            if scenario.scenario_id == "i2v"
        )
        runtime_home = Path("tmp/failure-runtime-home")
        output_dir = Path("tmp/failure-benchmark-output")
        source_ref = SourceRef(
            provider="local",
            locator={"path": "/tmp/example-bundle"},
            family_hint="ltx",
        )
        source_entry = benchmark_ltx.SourceEntry(role="bundle", source_ref=source_ref)
        inspect_ok = (
            InspectStepResult(
                role="bundle",
                source_ref=source_ref,
                route_wall_ms=1.0,
                status_code=200,
                resolved_ref="rev_bundle",
            ),
            SimpleNamespace(),
        )
        register_ok = (
            RegisterStepResult(
                role="bundle",
                source_ref=source_ref,
                route_wall_ms=1.0,
                status_code=200,
                source_id="src_bundle",
                resolved_ref="rev_bundle",
            ),
            SimpleNamespace(source_id="src_bundle"),
        )
        convert_ok = (
            ConversionStepResult(
                request=ArtifactConversionRequest(
                    source_id="src_bundle",
                    model_id="ltx-benchmark-local-bundle-i2v",
                ),
                route_wall_ms=1.0,
                status_code=200,
                artifact_digest="sha256:test",
            ),
            SimpleNamespace(artifact=SimpleNamespace(artifact_digest="sha256:test")),
        )
        import_ok = (
            InputImportStepResult(
                fixture_path=str(DEFAULT_CONDITIONING_IMAGE),
                route_wall_ms=1.0,
                status_code=200,
                handle_id="inp_1",
            ),
            SimpleNamespace(handle_id="inp_1"),
        )
        successful_run = JobRunResult(
            label="cold",
            submit_route_wall_ms=1.0,
            submit_status_code=200,
            job_id="job_1",
            terminal_state="completed",
        )
        failed_run = JobRunResult(
            label="cold",
            submit_route_wall_ms=1.0,
            submit_status_code=200,
            job_id="job_1",
            error="boom",
            failure_category="unknown_failure",
        )

        with patch(
            "mlx_runtime_cli.benchmark_ltx._source_entries_for_mode",
            return_value=(source_entry,),
        ):
            with patch(
                "mlx_runtime_cli.benchmark_ltx._inspect_source",
                return_value=(
                    InspectStepResult(
                        role="bundle",
                        source_ref=source_ref,
                        route_wall_ms=1.0,
                        status_code=404,
                        error="missing",
                    ),
                    None,
                ),
            ):
                inspect_failed = run_scenario_with_api(
                    RuntimeApi(lambda *_args, **_kwargs: httpx.Response(200)),
                    scenario=scenario,
                    source_mode="local-bundle",
                    runtime_home=runtime_home,
                    scenario_output_dir=output_dir,
                    local_bundle_path=Path("/tmp/example-bundle"),
                )
            self.assertEqual(
                inspect_failed.failure_category, "provider_resolution_failure"
            )

            with (
                patch(
                    "mlx_runtime_cli.benchmark_ltx._inspect_source",
                    return_value=inspect_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._register_source",
                    return_value=(
                        RegisterStepResult(
                            role="bundle",
                            source_ref=source_ref,
                            route_wall_ms=1.0,
                            status_code=400,
                            error="register failed",
                        ),
                        None,
                    ),
                ),
            ):
                register_failed = run_scenario_with_api(
                    RuntimeApi(lambda *_args, **_kwargs: httpx.Response(200)),
                    scenario=scenario,
                    source_mode="local-bundle",
                    runtime_home=runtime_home,
                    scenario_output_dir=output_dir,
                    local_bundle_path=Path("/tmp/example-bundle"),
                )
            self.assertEqual(register_failed.status, "failed")

            with (
                patch(
                    "mlx_runtime_cli.benchmark_ltx._inspect_source",
                    return_value=inspect_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._register_source",
                    return_value=register_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._convert_artifact",
                    return_value=(
                        ConversionStepResult(
                            request=ArtifactConversionRequest(
                                source_id="src_bundle",
                                model_id="ltx-benchmark-local-bundle-i2v",
                            ),
                            route_wall_ms=1.0,
                            status_code=400,
                            error="convert failed",
                        ),
                        None,
                    ),
                ),
            ):
                convert_failed = run_scenario_with_api(
                    RuntimeApi(lambda *_args, **_kwargs: httpx.Response(200)),
                    scenario=scenario,
                    source_mode="local-bundle",
                    runtime_home=runtime_home,
                    scenario_output_dir=output_dir,
                    local_bundle_path=Path("/tmp/example-bundle"),
                )
            self.assertEqual(convert_failed.failure_category, "conversion_failure")

            with (
                patch(
                    "mlx_runtime_cli.benchmark_ltx._inspect_source",
                    return_value=inspect_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._register_source",
                    return_value=register_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._convert_artifact",
                    return_value=convert_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._import_conditioning_input",
                    return_value=(
                        InputImportStepResult(
                            fixture_path=str(DEFAULT_CONDITIONING_IMAGE),
                            route_wall_ms=1.0,
                            status_code=400,
                            error="import failed",
                        ),
                        None,
                    ),
                ),
            ):
                import_failed = run_scenario_with_api(
                    RuntimeApi(lambda *_args, **_kwargs: httpx.Response(200)),
                    scenario=scenario,
                    source_mode="local-bundle",
                    runtime_home=runtime_home,
                    scenario_output_dir=output_dir,
                    local_bundle_path=Path("/tmp/example-bundle"),
                )
            self.assertEqual(import_failed.status, "failed")

            with (
                patch(
                    "mlx_runtime_cli.benchmark_ltx._inspect_source",
                    return_value=inspect_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._register_source",
                    return_value=register_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._convert_artifact",
                    return_value=convert_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._import_conditioning_input",
                    return_value=import_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._run_job_once",
                    side_effect=[failed_run],
                ),
            ):
                cold_failed = run_scenario_with_api(
                    RuntimeApi(lambda *_args, **_kwargs: httpx.Response(200)),
                    scenario=scenario,
                    source_mode="local-bundle",
                    runtime_home=runtime_home,
                    scenario_output_dir=output_dir,
                    local_bundle_path=Path("/tmp/example-bundle"),
                )
            self.assertEqual(cold_failed.failure_category, "unknown_failure")

            warm_failed_run = JobRunResult(
                label="warm",
                submit_route_wall_ms=1.0,
                submit_status_code=200,
                job_id="job_2",
                error="warm failed",
                failure_category="unknown_failure",
            )
            with (
                patch(
                    "mlx_runtime_cli.benchmark_ltx._inspect_source",
                    return_value=inspect_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._register_source",
                    return_value=register_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._convert_artifact",
                    return_value=convert_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._import_conditioning_input",
                    return_value=import_ok,
                ),
                patch(
                    "mlx_runtime_cli.benchmark_ltx._run_job_once",
                    side_effect=[successful_run, warm_failed_run],
                ),
            ):
                warm_failed = run_scenario_with_api(
                    RuntimeApi(lambda *_args, **_kwargs: httpx.Response(200)),
                    scenario=scenario,
                    source_mode="local-bundle",
                    runtime_home=runtime_home,
                    scenario_output_dir=output_dir,
                    local_bundle_path=Path("/tmp/example-bundle"),
                )
            self.assertEqual(warm_failed.failure_category, "unknown_failure")

    def test_run_job_once_helper_failure_modes(self) -> None:
        scenario = next(
            scenario
            for scenario in build_default_scenarios()
            if scenario.scenario_id == "t2v"
        )
        downloads_dir = Path("tmp/run-job-failures")
        downloads_dir.mkdir(parents=True, exist_ok=True)

        submit_failure = benchmark_ltx._run_job_once(
            RuntimeApi(
                lambda *_args, **_kwargs: httpx.Response(
                    409, json={"detail": "A heavy job is already running"}
                )
            ),
            scenario=scenario,
            label="cold",
            model_id="model",
            downloads_dir=downloads_dir,
            conditioning_handle_id=None,
            poll_interval_seconds=0.01,
            job_timeout_seconds=0.01,
            download_timeout_seconds=0.01,
        )
        self.assertEqual(submit_failure.failure_category, "admission_rejection")

        with patch(
            "mlx_runtime_cli.benchmark_ltx._wait_for_job_terminal_state",
            return_value=(None, None, "timed out"),
        ):
            timeout_result = benchmark_ltx._run_job_once(
                RuntimeApi(
                    lambda *_args, **_kwargs: httpx.Response(
                        200,
                        json={
                            "job_id": "job_1",
                            "record": {
                                "job_id": "job_1",
                                "request": {
                                    "model_id": "model",
                                    "task": "video.generate",
                                    "inputs": {},
                                    "params": {},
                                    "output": {
                                        "artifact_format": "mp4",
                                        "destination_mode": "runtime_managed",
                                        "stream": False,
                                    },
                                    "extensions": {},
                                },
                                "state": "accepted",
                                "artifacts": [],
                            },
                        },
                    )
                ),
                scenario=scenario,
                label="cold",
                model_id="model",
                downloads_dir=downloads_dir,
                conditioning_handle_id=None,
                poll_interval_seconds=0.01,
                job_timeout_seconds=0.01,
                download_timeout_seconds=0.01,
            )
        self.assertEqual(timeout_result.failure_category, "unknown_failure")

        completed_record = JobRecord.model_validate(
            {
                "job_id": "job_1",
                "request": {
                    "model_id": "model",
                    "task": "video.generate",
                    "inputs": {},
                    "params": {},
                    "output": {
                        "artifact_format": "mp4",
                        "destination_mode": "runtime_managed",
                        "stream": False,
                    },
                    "extensions": {},
                },
                "state": "completed",
                "artifacts": [],
            }
        )
        submit_ok = httpx.Response(
            200,
            json={
                "job_id": "job_1",
                "record": completed_record.model_dump(mode="json"),
            },
        )
        with (
            patch(
                "mlx_runtime_cli.benchmark_ltx._wait_for_job_terminal_state",
                return_value=(completed_record, 1.0, None),
            ),
            patch(
                "mlx_runtime_cli.benchmark_ltx._timed_request",
                side_effect=[
                    (submit_ok, 1.0),
                    (httpx.Response(500, text="events failed"), 1.0),
                ],
            ),
        ):
            events_failed = benchmark_ltx._run_job_once(
                RuntimeApi(lambda *_args, **_kwargs: httpx.Response(200)),
                scenario=scenario,
                label="cold",
                model_id="model",
                downloads_dir=downloads_dir,
                conditioning_handle_id=None,
                poll_interval_seconds=0.01,
                job_timeout_seconds=0.01,
                download_timeout_seconds=0.01,
            )
        self.assertEqual(events_failed.failure_category, "unknown_failure")

        failed_record = completed_record.model_copy(
            update={"state": JobState.FAILED, "error": "ffmpeg failed"}
        )
        with (
            patch(
                "mlx_runtime_cli.benchmark_ltx._wait_for_job_terminal_state",
                return_value=(failed_record, 1.0, None),
            ),
            patch(
                "mlx_runtime_cli.benchmark_ltx._timed_request",
                side_effect=[
                    (submit_ok, 1.0),
                    (
                        httpx.Response(
                            200,
                            text='event: job.phase_changed\ndata: {"job_id":"job_1","kind":"job.phase_changed","phase":"streaming_output","data":{"stage_id":"encode_output"}}\n\nevent: job.failed\ndata: {"job_id":"job_1","kind":"job.failed","phase":"failed","data":{"error":"ffmpeg failed"}}\n\n',
                        ),
                        1.0,
                    ),
                ],
            ),
        ):
            failed_terminal = benchmark_ltx._run_job_once(
                RuntimeApi(lambda *_args, **_kwargs: httpx.Response(200)),
                scenario=scenario,
                label="cold",
                model_id="model",
                downloads_dir=downloads_dir,
                conditioning_handle_id=None,
                poll_interval_seconds=0.01,
                job_timeout_seconds=0.01,
                download_timeout_seconds=0.01,
            )
        self.assertEqual(
            failed_terminal.failure_category, "output_encode_or_export_failure"
        )

    def test_running_runtime_daemon_and_main_cover_cli_flow(self) -> None:
        root = Path("tmp/benchmark-cli")
        bundle_dir = root / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "dummy.txt").write_text("bundle", encoding="utf-8")

        class FakeProcess:
            def __init__(self) -> None:
                self._terminated = False

            def poll(self) -> int | None:
                return None if not self._terminated else 0

            def terminate(self) -> None:
                self._terminated = True

            def wait(self, timeout: float) -> int:
                self._terminated = True
                return 0

            def kill(self) -> None:
                self._terminated = True

        fake_client = SimpleNamespace(
            request=lambda *_args, **_kwargs: httpx.Response(
                200, json={"status": "ok"}
            ),
            close=lambda: None,
        )
        with (
            patch(
                "mlx_runtime_cli.benchmark_ltx.subprocess.Popen",
                return_value=FakeProcess(),
            ),
            patch("mlx_runtime_cli.benchmark_ltx.httpx.HTTPTransport"),
            patch(
                "mlx_runtime_cli.benchmark_ltx.httpx.Client", return_value=fake_client
            ),
            patch("mlx_runtime_cli.benchmark_ltx._wait_for_health"),
        ):
            with benchmark_ltx.running_runtime_daemon(
                runtime_home=root / "runtime-home",
                server_stdio_log_path=root / "server.log",
            ) as (api, log_path):
                response = api.request("GET", "/health")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(str(log_path).endswith("server.log"))

        benchmark_result = ScenarioBenchmarkResult(
            scenario_id="t2v",
            task="video.generate",
            source_mode="local-bundle",
            model_id="model",
            status="completed",
            environment=BenchmarkEnvironment(
                timestamp_utc=datetime.now(timezone.utc),
                transport="uds",
                runtime_home=str(root / "runtime-home"),
                scenario_output_dir=str(root / "scenario"),
                control_plane_log_path=str(
                    root / "runtime-home" / "logs" / "control-plane.log"
                ),
                machine={},
                package_versions={},
            ),
            source_setup=SourceSetupResult(),
        )

        @contextmanager
        def fake_running_runtime_daemon(
            **_kwargs: object,
        ) -> Generator[tuple[RuntimeApi, Path], None, None]:
            yield (
                RuntimeApi(lambda *_args, **_kw: httpx.Response(200)),
                root / "server.log",
            )

        with (
            patch(
                "mlx_runtime_cli.benchmark_ltx.running_runtime_daemon",
                fake_running_runtime_daemon,
            ),
            patch(
                "mlx_runtime_cli.benchmark_ltx.run_scenario_with_api",
                return_value=benchmark_result,
            ),
            patch("mlx_runtime_cli.benchmark_ltx.write_benchmark_result"),
        ):
            self.assertEqual(
                benchmark_ltx.main(
                    [
                        "--bundle-path",
                        str(bundle_dir),
                        "--scenario",
                        "t2v",
                        "--results-dir",
                        str(root / "results"),
                        "--output-dir",
                        str(root / "output"),
                    ]
                ),
                0,
            )

        failing_result = benchmark_result.model_copy(update={"status": "failed"})
        with (
            patch(
                "mlx_runtime_cli.benchmark_ltx.running_runtime_daemon",
                fake_running_runtime_daemon,
            ),
            patch(
                "mlx_runtime_cli.benchmark_ltx.run_scenario_with_api",
                return_value=failing_result,
            ),
            patch("mlx_runtime_cli.benchmark_ltx.write_benchmark_result"),
        ):
            self.assertEqual(
                benchmark_ltx.main(
                    [
                        "--source-mode",
                        "huggingface",
                        "--scenario",
                        "t2v",
                        "--results-dir",
                        str(root / "results-hf"),
                        "--output-dir",
                        str(root / "output-hf"),
                    ]
                ),
                1,
            )
        with self.assertRaises(SystemExit):
            benchmark_ltx.main(["--source-mode", "local-bundle"])


if __name__ == "__main__":
    unittest.main()

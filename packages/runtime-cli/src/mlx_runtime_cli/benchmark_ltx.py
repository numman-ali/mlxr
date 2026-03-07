from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from mlx_runtime_core import RuntimeHome
from mlx_runtime_schemas import (
    ArtifactConversionRequest,
    ArtifactConversionResult,
    ArtifactConversionTimingsMs,
    InputHandleRecord,
    JobRecord,
    JobSubmitResult,
    OutputArtifactRecord,
    RuntimeEvent,
    RuntimeEventKind,
    SourceAuth,
    SourceInspectionResult,
    SourceInspectionTimingsMs,
    SourceRef,
    SourceRegistrationRecord,
)
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "output"
DEFAULT_CONDITIONING_IMAGE = REPO_ROOT / "benchmarks" / "fixtures" / "conditioning.ppm"
DEFAULT_UDS_SOCKET_ROOT = Path("/tmp/mlxr-bench-uds")
DEFAULT_SESSION_LABEL_FORMAT = "%Y%m%dT%H%M%SZ"
DEFAULT_POLL_INTERVAL_SECONDS = 0.05
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60.0
DEFAULT_CONVERSION_TIMEOUT_SECONDS = 1800.0
DEFAULT_JOB_TIMEOUT_SECONDS = 900.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 120.0
LTX_REPO_ID = "Lightricks/LTX-2.3"
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"

SourceMode = Literal["local-bundle", "huggingface"]
RunLabel = Literal["cold", "warm"]


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    scenario_id: str
    task: str
    prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    seed: int
    conditioning_image: Path | None = None
    conditioning_media_type: str = "image/x-portable-pixmap"


@dataclass(frozen=True, slots=True)
class SourceEntry:
    role: str
    source_ref: SourceRef


class UnavailableMetric(BaseModel):
    name: str
    reason: str


class BenchmarkEnvironment(BaseModel):
    timestamp_utc: datetime
    transport: str
    runtime_home: str
    scenario_output_dir: str
    control_plane_log_path: str
    server_stdio_log_path: str | None = None
    machine: dict[str, object]
    package_versions: dict[str, str | None]


class InspectStepResult(BaseModel):
    role: str
    source_ref: SourceRef
    route_wall_ms: float
    status_code: int
    resolved_ref: str | None = None
    access_state: str | None = None
    bytes_total: int | None = None
    timings_ms: SourceInspectionTimingsMs | None = None
    error: str | None = None


class RegisterStepResult(BaseModel):
    role: str
    source_ref: SourceRef
    route_wall_ms: float
    status_code: int
    source_id: str | None = None
    resolved_ref: str | None = None
    access_state: str | None = None
    error: str | None = None


class ConversionStepResult(BaseModel):
    request: ArtifactConversionRequest
    route_wall_ms: float
    status_code: int
    artifact_digest: str | None = None
    timings_ms: ArtifactConversionTimingsMs | None = None
    error: str | None = None


class InputImportStepResult(BaseModel):
    fixture_path: str
    route_wall_ms: float
    status_code: int
    handle_id: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    error: str | None = None


class StageMetricResult(BaseModel):
    timestamp: datetime
    phase: str | None = None
    stage_id: str
    duration_ms: float
    memory: dict[str, object]
    metrics: dict[str, object] = Field(default_factory=dict)


class JobRunResult(BaseModel):
    label: RunLabel
    submit_route_wall_ms: float
    submit_status_code: int
    job_id: str | None = None
    job_wait_ms: float | None = None
    terminal_state: str | None = None
    artifact_id: str | None = None
    artifact_format: str | None = None
    download_ms: float | None = None
    downloaded_path: str | None = None
    downloaded_size_bytes: int | None = None
    generation_backend: str | None = None
    stage_metrics: list[StageMetricResult] = Field(default_factory=list)
    events_count: int = 0
    error: str | None = None
    failure_category: str | None = None


class SourceSetupResult(BaseModel):
    inspect: list[InspectStepResult] = Field(default_factory=list)
    register_steps: list[RegisterStepResult] = Field(default_factory=list)
    conversion: ConversionStepResult | None = None
    input_import: InputImportStepResult | None = None


class ScenarioBenchmarkResult(BaseModel):
    scenario_id: str
    task: str
    source_mode: SourceMode
    model_id: str
    status: str
    environment: BenchmarkEnvironment
    source_setup: SourceSetupResult
    source_bindings: dict[str, str] = Field(default_factory=dict)
    artifact_digest: str | None = None
    resolved_refs_by_role: dict[str, str | None] = Field(default_factory=dict)
    runs: list[JobRunResult] = Field(default_factory=list)
    unavailable_metrics: list[UnavailableMetric] = Field(default_factory=list)
    error: str | None = None
    failure_category: str | None = None


class RuntimeApi:
    def __init__(
        self,
        requester: Callable[..., httpx.Response],
        *,
        supports_timeout: bool = True,
    ) -> None:
        self._requester = requester
        self._supports_timeout = supports_timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: object | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        request_kwargs: dict[str, object] = {}
        if json_payload is not None:
            request_kwargs["json"] = json_payload
        if timeout is not None and self._supports_timeout:
            request_kwargs["timeout"] = timeout
        return self._requester(method, path, **request_kwargs)


def build_default_scenarios(
    *, conditioning_image: Path = DEFAULT_CONDITIONING_IMAGE
) -> tuple[ScenarioConfig, ...]:
    return (
        ScenarioConfig(
            scenario_id="t2v",
            task="video.generate",
            prompt="Slow cinematic dolly through a neon-lit corridor with drifting haze",
            width=96,
            height=64,
            num_frames=9,
            fps=12,
            seed=7,
        ),
        ScenarioConfig(
            scenario_id="i2v",
            task="video.condition.image",
            prompt="Turn this still frame into a subtle forward camera move",
            width=96,
            height=64,
            num_frames=9,
            fps=12,
            seed=7,
            conditioning_image=conditioning_image,
        ),
    )


def run_scenario_with_api(
    api: RuntimeApi,
    *,
    scenario: ScenarioConfig,
    source_mode: SourceMode,
    runtime_home: Path,
    scenario_output_dir: Path,
    local_bundle_path: Path | None = None,
    model_id: str | None = None,
    server_stdio_log_path: Path | None = None,
    transport: str = "in_process",
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    conversion_timeout_seconds: float = DEFAULT_CONVERSION_TIMEOUT_SECONDS,
    job_timeout_seconds: float = DEFAULT_JOB_TIMEOUT_SECONDS,
    download_timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> ScenarioBenchmarkResult:
    runtime_home = runtime_home.resolve()
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = scenario_output_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    environment = BenchmarkEnvironment(
        timestamp_utc=datetime.now(timezone.utc),
        transport=transport,
        runtime_home=str(runtime_home),
        scenario_output_dir=str(scenario_output_dir),
        control_plane_log_path=str(runtime_home / "logs" / "control-plane.log"),
        server_stdio_log_path=(
            str(server_stdio_log_path.resolve())
            if server_stdio_log_path is not None
            else None
        ),
        machine=_machine_details(),
        package_versions=_package_versions(),
    )
    result = ScenarioBenchmarkResult(
        scenario_id=scenario.scenario_id,
        task=scenario.task,
        source_mode=source_mode,
        model_id=model_id or f"ltx-benchmark-{source_mode}-{scenario.scenario_id}",
        status="running",
        environment=environment,
        source_setup=SourceSetupResult(),
        unavailable_metrics=_unavailable_metrics(),
    )

    source_entries = _source_entries_for_mode(
        source_mode=source_mode, local_bundle_path=local_bundle_path
    )
    inspect_refs_by_role: dict[str, str | None] = {}
    registered_source_ids: dict[str, str] = {}
    for entry in source_entries:
        inspect_record, inspection = _inspect_source(
            api, role=entry.role, source_ref=entry.source_ref
        )
        result.source_setup.inspect.append(inspect_record)
        inspect_refs_by_role[entry.role] = inspect_record.resolved_ref
        if inspection is None:
            result.status = "failed"
            result.error = inspect_record.error
            result.failure_category = _classify_route_failure(
                step="inspect",
                status_code=inspect_record.status_code,
                error=inspect_record.error,
            )
            result.resolved_refs_by_role = inspect_refs_by_role
            return result

        register_record, registration = _register_source(
            api, role=entry.role, source_ref=entry.source_ref
        )
        result.source_setup.register_steps.append(register_record)
        if registration is None or register_record.source_id is None:
            result.status = "failed"
            result.error = register_record.error
            result.failure_category = _classify_route_failure(
                step="register",
                status_code=register_record.status_code,
                error=register_record.error,
            )
            result.resolved_refs_by_role = inspect_refs_by_role
            return result
        registered_source_ids[entry.role] = register_record.source_id

    result.resolved_refs_by_role = inspect_refs_by_role
    conversion_request = _conversion_request_for_source_mode(
        source_mode=source_mode,
        model_id=result.model_id,
        registered_source_ids=registered_source_ids,
    )
    conversion_record, conversion = _convert_artifact(
        api,
        conversion_request,
        timeout_seconds=conversion_timeout_seconds,
    )
    result.source_setup.conversion = conversion_record
    result.source_bindings = _source_bindings_for_result(
        source_mode=source_mode,
        registered_source_ids=registered_source_ids,
        conversion_request=conversion_request,
    )
    if conversion is None:
        result.status = "failed"
        result.error = conversion_record.error
        result.failure_category = _classify_route_failure(
            step="convert",
            status_code=conversion_record.status_code,
            error=conversion_record.error,
        )
        return result
    result.artifact_digest = conversion.artifact.artifact_digest

    conditioning_handle_id: str | None = None
    if scenario.conditioning_image is not None:
        import_record, input_record = _import_conditioning_input(
            api,
            fixture_path=scenario.conditioning_image,
            media_type=scenario.conditioning_media_type,
        )
        result.source_setup.input_import = import_record
        if input_record is None or input_record.handle_id is None:
            result.status = "failed"
            result.error = import_record.error
            result.failure_category = _classify_route_failure(
                step="import_input",
                status_code=import_record.status_code,
                error=import_record.error,
            )
            return result
        conditioning_handle_id = input_record.handle_id

    cold_run = _run_job_once(
        api,
        scenario=scenario,
        label="cold",
        model_id=result.model_id,
        downloads_dir=downloads_dir,
        conditioning_handle_id=conditioning_handle_id,
        poll_interval_seconds=poll_interval_seconds,
        job_timeout_seconds=job_timeout_seconds,
        download_timeout_seconds=download_timeout_seconds,
    )
    result.runs.append(cold_run)
    if cold_run.failure_category is not None:
        result.status = "failed"
        result.error = cold_run.error
        result.failure_category = cold_run.failure_category
        return result

    warm_run = _run_job_once(
        api,
        scenario=scenario,
        label="warm",
        model_id=result.model_id,
        downloads_dir=downloads_dir,
        conditioning_handle_id=conditioning_handle_id,
        poll_interval_seconds=poll_interval_seconds,
        job_timeout_seconds=job_timeout_seconds,
        download_timeout_seconds=download_timeout_seconds,
    )
    result.runs.append(warm_run)
    if warm_run.failure_category is not None:
        result.status = "failed"
        result.error = warm_run.error
        result.failure_category = warm_run.failure_category
        return result

    result.status = "completed"
    return result


def write_benchmark_result(result: ScenarioBenchmarkResult, result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
    result_path.write_text(f"{payload}\n", encoding="utf-8")


@contextmanager
def running_runtime_daemon(
    *,
    runtime_home: Path,
    server_stdio_log_path: Path,
    socket_path: Path | None = None,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> Generator[tuple[RuntimeApi, Path], None, None]:
    runtime_home = runtime_home.resolve()
    runtime = RuntimeHome(root=runtime_home)
    runtime.ensure_layout()
    server_stdio_log_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path = (
        socket_path.resolve()
        if socket_path is not None
        else (runtime.temp_dir / "control-plane.sock").resolve()
    )
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    process_env = os.environ.copy()
    process_env["MLX_RUNTIME_HOME"] = str(runtime_home)
    process_env["MLX_RUNTIME_UDS_PATH"] = str(socket_path)
    process_env.pop("MLX_RUNTIME_HTTP_HOST", None)
    process_env.pop("MLX_RUNTIME_HTTP_PORT", None)
    process_env.pop("MLX_RUNTIME_HTTP_TOKEN", None)
    process_env.pop("MLX_RUNTIME_ALLOWED_ORIGINS", None)
    if env is not None:
        process_env.update(env)

    with server_stdio_log_path.open("wb") as stdio_handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "mlx_runtime_server"],
            cwd=str(REPO_ROOT),
            env=process_env,
            stdout=stdio_handle,
            stderr=subprocess.STDOUT,
        )
        transport = httpx.HTTPTransport(
            uds=str(socket_path), retries=0, trust_env=False
        )
        client = httpx.Client(
            transport=transport,
            base_url="http://mlxr",
            timeout=httpx.Timeout(30.0),
        )
        try:
            _wait_for_health(
                client,
                process=process,
                startup_timeout_seconds=startup_timeout_seconds,
            )
            yield RuntimeApi(client.request), server_stdio_log_path.resolve()
        finally:
            client.close()
            _terminate_process(process)
            if socket_path.exists():
                socket_path.unlink()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run end-to-end LTX benchmarks")
    parser.add_argument(
        "--source-mode",
        choices=("local-bundle", "huggingface"),
        default="local-bundle",
        help="Model source mode to benchmark",
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        help="Trusted local bundle path for --source-mode=local-bundle",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("t2v", "i2v"),
        help="Scenario to run; repeat the flag to run multiple scenarios",
    )
    parser.add_argument(
        "--conditioning-image",
        type=Path,
        default=DEFAULT_CONDITIONING_IMAGE,
        help="Conditioning image fixture for the I2V benchmark",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for structured benchmark JSON outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for runtime homes, downloads, and daemon stdio logs",
    )
    parser.add_argument(
        "--session-label",
        type=str,
        default=datetime.now(timezone.utc).strftime(DEFAULT_SESSION_LABEL_FORMAT),
        help="Directory label for this benchmark session",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="Time to wait for the daemon health check",
    )
    parser.add_argument(
        "--conversion-timeout-seconds",
        type=float,
        default=DEFAULT_CONVERSION_TIMEOUT_SECONDS,
        help="Time to wait for artifact conversion, including provider download",
    )
    parser.add_argument(
        "--job-timeout-seconds",
        type=float,
        default=DEFAULT_JOB_TIMEOUT_SECONDS,
        help="Time to wait for each benchmark job to reach a terminal state",
    )
    args = parser.parse_args(argv)

    source_mode = args.source_mode
    bundle_path = args.bundle_path.resolve() if args.bundle_path is not None else None
    if source_mode == "local-bundle" and bundle_path is None:
        parser.error("--bundle-path is required when --source-mode=local-bundle")
    if bundle_path is not None and not bundle_path.exists():
        parser.error(f"Bundle path '{bundle_path}' does not exist")

    scenarios = {
        scenario.scenario_id: scenario
        for scenario in build_default_scenarios(
            conditioning_image=args.conditioning_image.resolve()
        )
    }
    selected_ids = args.scenario or ["t2v", "i2v"]
    selected_scenarios = [scenarios[scenario_id] for scenario_id in selected_ids]

    session_results_dir = args.results_dir.resolve() / args.session_label
    session_output_dir = args.output_dir.resolve() / args.session_label
    session_results_dir.mkdir(parents=True, exist_ok=True)
    session_output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for scenario in selected_scenarios:
        scenario_output_dir = session_output_dir / scenario.scenario_id
        runtime_home = scenario_output_dir / "runtime-home"
        stdio_log_path = scenario_output_dir / "server-stdio.log"
        socket_path = _benchmark_socket_path(
            session_label=args.session_label,
            scenario_id=scenario.scenario_id,
        )
        extra_env: dict[str, str] = {}
        if source_mode == "huggingface":
            extra_env.update(
                _huggingface_runtime_env(scenario_output_dir=scenario_output_dir)
            )
        with running_runtime_daemon(
            runtime_home=runtime_home,
            server_stdio_log_path=stdio_log_path,
            socket_path=socket_path,
            startup_timeout_seconds=args.startup_timeout_seconds,
            env=extra_env,
        ) as (api, resolved_stdio_log_path):
            result = run_scenario_with_api(
                api,
                scenario=scenario,
                source_mode=source_mode,
                runtime_home=runtime_home,
                scenario_output_dir=scenario_output_dir,
                local_bundle_path=bundle_path,
                server_stdio_log_path=resolved_stdio_log_path,
                transport="uds",
                poll_interval_seconds=DEFAULT_POLL_INTERVAL_SECONDS,
                conversion_timeout_seconds=args.conversion_timeout_seconds,
                job_timeout_seconds=args.job_timeout_seconds,
                download_timeout_seconds=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
            )
        result_path = session_results_dir / f"{scenario.scenario_id}.json"
        write_benchmark_result(result, result_path)
        print(f"[{scenario.scenario_id}] {result.status} -> {result_path}")
        if result.status != "completed":
            failures += 1

    return 0 if failures == 0 else 1


def _inspect_source(
    api: RuntimeApi, *, role: str, source_ref: SourceRef
) -> tuple[InspectStepResult, SourceInspectionResult | None]:
    response, route_wall_ms = _timed_request(
        api,
        "POST",
        "/v1/sources/inspect",
        json_payload=source_ref.model_dump(mode="json"),
    )
    if response.status_code != 200:
        return (
            InspectStepResult(
                role=role,
                source_ref=source_ref,
                route_wall_ms=route_wall_ms,
                status_code=response.status_code,
                error=_response_error_text(response),
            ),
            None,
        )
    inspection = SourceInspectionResult.model_validate(response.json())
    return (
        InspectStepResult(
            role=role,
            source_ref=source_ref,
            route_wall_ms=route_wall_ms,
            status_code=response.status_code,
            resolved_ref=inspection.provenance.resolved_ref,
            access_state=inspection.resolved_source.access_state,
            bytes_total=inspection.provider_inspection.bytes_total,
            timings_ms=inspection.timings_ms,
        ),
        inspection,
    )


def _register_source(
    api: RuntimeApi, *, role: str, source_ref: SourceRef
) -> tuple[RegisterStepResult, SourceRegistrationRecord | None]:
    response, route_wall_ms = _timed_request(
        api,
        "POST",
        "/v1/sources/register",
        json_payload=source_ref.model_dump(mode="json"),
    )
    if response.status_code != 200:
        return (
            RegisterStepResult(
                role=role,
                source_ref=source_ref,
                route_wall_ms=route_wall_ms,
                status_code=response.status_code,
                error=_response_error_text(response),
            ),
            None,
        )
    registration = SourceRegistrationRecord.model_validate(response.json())
    return (
        RegisterStepResult(
            role=role,
            source_ref=source_ref,
            route_wall_ms=route_wall_ms,
            status_code=response.status_code,
            source_id=registration.source_id,
            resolved_ref=registration.provenance.resolved_ref,
            access_state=registration.resolved_source.access_state,
        ),
        registration,
    )


def _convert_artifact(
    api: RuntimeApi,
    request: ArtifactConversionRequest,
    *,
    timeout_seconds: float,
) -> tuple[ConversionStepResult, ArtifactConversionResult | None]:
    response, route_wall_ms = _timed_request(
        api,
        "POST",
        "/v1/artifacts/convert",
        json_payload=request.model_dump(mode="json"),
        timeout=timeout_seconds,
    )
    if response.status_code != 200:
        return (
            ConversionStepResult(
                request=request,
                route_wall_ms=route_wall_ms,
                status_code=response.status_code,
                error=_response_error_text(response),
            ),
            None,
        )
    conversion = ArtifactConversionResult.model_validate(response.json())
    return (
        ConversionStepResult(
            request=request,
            route_wall_ms=route_wall_ms,
            status_code=response.status_code,
            artifact_digest=conversion.artifact.artifact_digest,
            timings_ms=conversion.timings_ms,
        ),
        conversion,
    )


def _import_conditioning_input(
    api: RuntimeApi, *, fixture_path: Path, media_type: str
) -> tuple[InputImportStepResult, InputHandleRecord | None]:
    payload = fixture_path.read_bytes()
    response, route_wall_ms = _timed_request(
        api,
        "POST",
        "/v1/inputs/import",
        json_payload={
            "content_base64": base64.b64encode(payload).decode("ascii"),
            "media_type": media_type,
            "filename": fixture_path.name,
        },
    )
    if response.status_code != 200:
        return (
            InputImportStepResult(
                fixture_path=str(fixture_path.resolve()),
                route_wall_ms=route_wall_ms,
                status_code=response.status_code,
                error=_response_error_text(response),
            ),
            None,
        )
    record = InputHandleRecord.model_validate(response.json())
    return (
        InputImportStepResult(
            fixture_path=str(fixture_path.resolve()),
            route_wall_ms=route_wall_ms,
            status_code=response.status_code,
            handle_id=record.handle_id,
            media_type=record.media_type,
            size_bytes=record.size_bytes,
        ),
        record,
    )


def _run_job_once(
    api: RuntimeApi,
    *,
    scenario: ScenarioConfig,
    label: RunLabel,
    model_id: str,
    downloads_dir: Path,
    conditioning_handle_id: str | None,
    poll_interval_seconds: float,
    job_timeout_seconds: float,
    download_timeout_seconds: float,
) -> JobRunResult:
    request_payload = _job_request_payload(
        scenario=scenario,
        model_id=model_id,
        conditioning_handle_id=conditioning_handle_id,
    )
    response, submit_route_wall_ms = _timed_request(
        api, "POST", "/v1/jobs", json_payload=request_payload
    )
    if response.status_code != 200:
        error = _response_error_text(response)
        return JobRunResult(
            label=label,
            submit_route_wall_ms=submit_route_wall_ms,
            submit_status_code=response.status_code,
            error=error,
            failure_category=_classify_route_failure(
                step="submit", status_code=response.status_code, error=error
            ),
        )

    submit = JobSubmitResult.model_validate(response.json())
    terminal_record, job_wait_ms, timeout_error = _wait_for_job_terminal_state(
        api,
        job_id=submit.job_id,
        timeout_seconds=job_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    if terminal_record is None:
        return JobRunResult(
            label=label,
            submit_route_wall_ms=submit_route_wall_ms,
            submit_status_code=response.status_code,
            job_id=submit.job_id,
            error=timeout_error,
            failure_category="unknown_failure",
        )

    run_result = JobRunResult(
        label=label,
        submit_route_wall_ms=submit_route_wall_ms,
        submit_status_code=response.status_code,
        job_id=submit.job_id,
        job_wait_ms=job_wait_ms,
        terminal_state=terminal_record.state.value,
        error=terminal_record.error,
    )
    events_response, _ = _timed_request(
        api, "GET", f"/v1/jobs/{submit.job_id}/events", timeout=download_timeout_seconds
    )
    if events_response.status_code != 200:
        error = _response_error_text(events_response)
        run_result.error = error
        run_result.failure_category = "unknown_failure"
        return run_result
    events = _parse_sse_events(events_response.text)
    run_result.events_count = len(events)
    run_result.stage_metrics = _stage_metrics_from_events(events)
    run_result.generation_backend = _generation_backend_from_stage_metrics(
        run_result.stage_metrics
    )

    if terminal_record.state.value != "completed":
        run_result.failure_category = _classify_job_failure(
            events=events, error=terminal_record.error
        )
        return run_result

    artifact_record = _first_artifact(terminal_record)
    if artifact_record is None:
        run_result.error = "Completed job produced no output artifact"
        run_result.failure_category = "output_encode_or_export_failure"
        return run_result
    run_result.artifact_id = artifact_record.artifact_id
    run_result.artifact_format = artifact_record.artifact_format

    download_response, download_ms = _timed_request(
        api,
        "GET",
        f"/v1/outputs/{artifact_record.artifact_id}/download",
        timeout=download_timeout_seconds,
    )
    if download_response.status_code != 200:
        error = _response_error_text(download_response)
        run_result.download_ms = download_ms
        run_result.error = error
        run_result.failure_category = "output_encode_or_export_failure"
        return run_result

    downloaded_path = downloads_dir / f"{label}.{artifact_record.artifact_format}"
    downloaded_path.write_bytes(download_response.content)
    run_result.download_ms = download_ms
    run_result.downloaded_path = str(downloaded_path.resolve())
    run_result.downloaded_size_bytes = len(download_response.content)
    return run_result


def _source_entries_for_mode(
    *, source_mode: SourceMode, local_bundle_path: Path | None
) -> tuple[SourceEntry, ...]:
    if source_mode == "local-bundle":
        if local_bundle_path is None:
            raise ValueError(
                "local_bundle_path is required for source_mode='local-bundle'"
            )
        return (
            SourceEntry(
                role="bundle",
                source_ref=SourceRef(
                    provider="local",
                    locator={"path": str(local_bundle_path.resolve())},
                    family_hint="ltx",
                ),
            ),
        )
    return (
        SourceEntry(
            role="checkpoint",
            source_ref=SourceRef(
                provider="huggingface",
                locator={
                    "repo": LTX_REPO_ID,
                    "revision": "main",
                    "role": "checkpoint",
                },
                auth=SourceAuth(token_ref="hf-default"),
                family_hint="ltx",
            ),
        ),
        SourceEntry(
            role="spatial_upsampler",
            source_ref=SourceRef(
                provider="huggingface",
                locator={
                    "repo": LTX_REPO_ID,
                    "revision": "main",
                    "role": "spatial_upsampler",
                },
                auth=SourceAuth(token_ref="hf-default"),
                family_hint="ltx",
            ),
        ),
        SourceEntry(
            role="text_encoder",
            source_ref=SourceRef(
                provider="huggingface",
                locator={
                    "repo": GEMMA_REPO_ID,
                    "revision": "main",
                    "role": "text_encoder",
                },
                auth=SourceAuth(token_ref="hf-default"),
                family_hint="ltx",
            ),
        ),
    )


def _conversion_request_for_source_mode(
    *,
    source_mode: SourceMode,
    model_id: str,
    registered_source_ids: dict[str, str],
) -> ArtifactConversionRequest:
    if source_mode == "local-bundle":
        source_id = registered_source_ids.get("bundle")
        if source_id is None:
            raise ValueError(
                "Missing registered bundle source for local-bundle conversion"
            )
        return ArtifactConversionRequest(source_id=source_id, model_id=model_id)
    return ArtifactConversionRequest(
        source_bindings={
            "checkpoint": registered_source_ids["checkpoint"],
            "spatial_upsampler": registered_source_ids["spatial_upsampler"],
            "text_encoder": registered_source_ids["text_encoder"],
        },
        family="ltx",
        model_id=model_id,
    )


def _source_bindings_for_result(
    *,
    source_mode: SourceMode,
    registered_source_ids: dict[str, str],
    conversion_request: ArtifactConversionRequest,
) -> dict[str, str]:
    if source_mode == "local-bundle":
        source_id = conversion_request.source_id
        return {"bundle": source_id} if source_id is not None else {}
    source_bindings = conversion_request.source_bindings
    return dict(source_bindings) if source_bindings is not None else {}


def _job_request_payload(
    *,
    scenario: ScenarioConfig,
    model_id: str,
    conditioning_handle_id: str | None,
) -> dict[str, object]:
    inputs: dict[str, object] = {"prompt": scenario.prompt}
    if conditioning_handle_id is not None:
        inputs["images"] = [
            {
                "input_handle": conditioning_handle_id,
                "frame_index": 0,
                "strength": 1.0,
            }
        ]
    return {
        "model_id": model_id,
        "task": scenario.task,
        "inputs": inputs,
        "params": {
            "width": scenario.width,
            "height": scenario.height,
            "num_frames": scenario.num_frames,
            "fps": scenario.fps,
            "seed": scenario.seed,
        },
        "output": {"artifact_format": "mp4"},
    }


def _timed_request(
    api: RuntimeApi,
    method: str,
    path: str,
    *,
    json_payload: object | None = None,
    timeout: float | None = None,
) -> tuple[httpx.Response, float]:
    started_at = time.perf_counter()
    response = api.request(method, path, json_payload=json_payload, timeout=timeout)
    return response, _elapsed_ms(started_at)


def _wait_for_job_terminal_state(
    api: RuntimeApi,
    *,
    job_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[JobRecord | None, float | None, str | None]:
    started_at = time.perf_counter()
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = api.request("GET", f"/v1/jobs/{job_id}")
        if response.status_code != 200:
            return None, None, _response_error_text(response)
        record = JobRecord.model_validate(response.json())
        if record.state.value in {"completed", "failed", "cancelled"}:
            return record, _elapsed_ms(started_at), None
        time.sleep(poll_interval_seconds)
    return None, None, f"Timed out waiting for job '{job_id}' to finish"


def _wait_for_health(
    client: httpx.Client,
    *,
    process: subprocess.Popen[bytes],
    startup_timeout_seconds: float,
) -> None:
    deadline = time.time() + startup_timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Runtime daemon exited before becoming healthy")
        try:
            response = client.get("/health")
        except httpx.HTTPError:
            time.sleep(0.1)
            continue
        if response.status_code == 200:
            return
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for runtime daemon health check")


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _parse_sse_events(payload: str) -> list[RuntimeEvent]:
    events: list[RuntimeEvent] = []
    data_lines: list[str] = []
    for line in payload.splitlines():
        if not line:
            if data_lines:
                events.append(
                    RuntimeEvent.model_validate(json.loads("\n".join(data_lines)))
                )
            data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines:
        events.append(RuntimeEvent.model_validate(json.loads("\n".join(data_lines))))
    return events


def _stage_metrics_from_events(events: list[RuntimeEvent]) -> list[StageMetricResult]:
    stage_metrics: list[StageMetricResult] = []
    for event in events:
        if event.kind != RuntimeEventKind.JOB_METRICS:
            continue
        stage_id = event.data.get("stage_id")
        duration_ms = event.data.get("duration_ms")
        memory = event.data.get("memory")
        metrics = event.data.get("metrics", {})
        if (
            not isinstance(stage_id, str)
            or not isinstance(duration_ms, (float, int))
            or not isinstance(memory, dict)
            or not isinstance(metrics, dict)
        ):
            continue
        stage_metrics.append(
            StageMetricResult(
                timestamp=event.timestamp,
                phase=event.phase,
                stage_id=stage_id,
                duration_ms=float(duration_ms),
                memory=memory,
                metrics=metrics,
            )
        )
    return stage_metrics


def _generation_backend_from_stage_metrics(
    stage_metrics: list[StageMetricResult],
) -> str | None:
    for stage_metric in stage_metrics:
        if stage_metric.stage_id != "generate":
            continue
        backend = stage_metric.metrics.get("backend")
        if isinstance(backend, str):
            return backend
    return None


def _first_artifact(record: JobRecord) -> OutputArtifactRecord | None:
    if not record.artifacts:
        return None
    return record.artifacts[0]


def _response_error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        error = payload.get("error")
        if isinstance(error, str):
            return error
    text = response.text.strip()
    return text or f"HTTP {response.status_code}"


def _classify_route_failure(*, step: str, status_code: int, error: str | None) -> str:
    message = (error or "").lower()
    if step == "submit" and status_code == 409:
        return "admission_rejection"
    if "remote code" in message:
        return "remote_code_policy_failure"
    if status_code == 403 or any(
        phrase in message for phrase in ("gated", "authenticated access", "hf_token")
    ):
        return "auth_or_access_failure"
    if step == "convert":
        return "conversion_failure"
    if step == "inspect":
        return "provider_resolution_failure"
    if step == "download":
        return "output_encode_or_export_failure"
    return "unknown_failure"


def _classify_job_failure(*, events: list[RuntimeEvent], error: str | None) -> str:
    message = (error or "").lower()
    if "oom" in message or "memory" in message:
        return "runtime_oom_or_memory_pressure"
    last_stage_id = _last_stage_id(events)
    if last_stage_id == "load_model":
        return "load_failure"
    if last_stage_id == "encode_output" or "ffmpeg" in message or "encode" in message:
        return "output_encode_or_export_failure"
    return "unknown_failure"


def _last_stage_id(events: list[RuntimeEvent]) -> str | None:
    for event in reversed(events):
        stage_id = event.data.get("stage_id")
        if isinstance(stage_id, str):
            return stage_id
    return None


def _machine_details() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "macos_version": platform.mac_ver()[0] or None,
        "cpu_count": os.cpu_count(),
        "memory_bytes": _system_memory_bytes(),
    }


def _system_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    raw_value = result.stdout.strip()
    return int(raw_value) if raw_value.isdigit() else None


def _package_versions() -> dict[str, str | None]:
    package_names = (
        "mlx",
        "mlx-runtime-cli",
        "mlx-runtime-core",
        "mlx-runtime-family-ltx",
        "mlx-runtime-schemas",
        "mlx-runtime-server",
    )
    return {
        package_name: _package_version(package_name) for package_name in package_names
    }


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _benchmark_socket_path(*, session_label: str, scenario_id: str) -> Path:
    digest = hashlib.sha1(f"{session_label}:{scenario_id}".encode("utf-8")).hexdigest()[
        :12
    ]
    return DEFAULT_UDS_SOCKET_ROOT / f"{scenario_id}-{digest}.sock"


def _huggingface_runtime_env(*, scenario_output_dir: Path) -> dict[str, str]:
    cache_root = scenario_output_dir.resolve()
    return {
        "HF_HUB_CACHE": str(cache_root / "hf-hub-cache"),
        "HF_XET_CACHE": str(cache_root / "hf-xet-cache"),
    }


def _unavailable_metrics() -> list[UnavailableMetric]:
    return [
        UnavailableMetric(
            name="compile_count",
            reason="The runtime does not emit compile-count telemetry yet.",
        ),
        UnavailableMetric(
            name="compile_time_ms",
            reason="The runtime does not emit compile-time telemetry yet.",
        ),
        UnavailableMetric(
            name="build_cache_hit_rate",
            reason="Machine-local build-cache hit telemetry is not implemented yet.",
        ),
        UnavailableMetric(
            name="source_cache_hit_rate",
            reason="Provider cache hit telemetry is not implemented yet.",
        ),
        UnavailableMetric(
            name="artifact_cache_hit_rate",
            reason="Portable-artifact cache hit telemetry is not implemented yet.",
        ),
        UnavailableMetric(
            name="metal.recommended_max_working_set_size_bytes",
            reason="Metal working-set telemetry is not wired into the benchmark harness yet.",
        ),
        UnavailableMetric(
            name="metal.current_allocated_size_bytes",
            reason="Metal current-allocation telemetry is not wired into the benchmark harness yet.",
        ),
        UnavailableMetric(
            name="cpu_utilization",
            reason="CPU utilization sampling is out of scope for this first benchmark slice.",
        ),
        UnavailableMetric(
            name="gpu_utilization",
            reason="GPU utilization sampling is out of scope for this first benchmark slice.",
        ),
        UnavailableMetric(
            name="media_engine_utilization",
            reason="Media-engine utilization sampling is out of scope for this first benchmark slice.",
        ),
        UnavailableMetric(
            name="cancellation_latency_ms",
            reason="Cancellation is not benchmarked in this first slice.",
        ),
        UnavailableMetric(
            name="model_switch",
            reason="Model-switch benchmarks are not included in this first slice.",
        ),
    ]


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)

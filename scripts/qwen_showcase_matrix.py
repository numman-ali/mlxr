from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_HOME = Path("/tmp/mlxr-qwen-lightning-runtime")
DEFAULT_SOCKET_PATH = DEFAULT_RUNTIME_HOME / "control-plane.sock"
DEFAULT_MODEL_ID = "qwen-image-2512-lightning-test"
DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 384
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low detail, deformed, broken anatomy, text artifacts, "
    "watermark, oversaturated, flat lighting"
)
DEFAULT_RESULTS_ROOT = REPO_ROOT / "tmp" / "showcase-runs"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_JOB_TIMEOUT_SECONDS = 1800.0


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class Profile:
    profile_id: str
    steps: int
    guidance_scale: float
    scheduler_preset: str
    lora_filename: str | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="floating_city",
        prompt=(
            "A vast bioluminescent floating city above the ocean at blue hour, "
            "layered sky bridges, hanging gardens, luminous trains curving "
            "through the skyline, thousands of tiny windows, reflective water "
            "below, cinematic wide establishing shot, ultra-detailed, "
            "awe-inspiring, intricate architecture, atmospheric depth."
        ),
    ),
    Scenario(
        scenario_id="seaside_portrait",
        prompt=(
            "A young East Asian woman with long flowing dark hair in a white "
            "summer dress standing on a seaside promenade at dusk, soft wind, "
            "gentle smile, seagulls in the distance, realistic smartphone-like "
            "photo but beautifully composed, fresh natural skin detail, "
            "cinematic wide frame, calm ocean atmosphere."
        ),
    ),
    Scenario(
        scenario_id="rain_violinist",
        prompt=(
            "A noir violinist performing beneath glowing signs in a rain-soaked "
            "alley, tailored black coat, wet cobblestones reflecting amber and "
            "cyan light, drifting steam, expressive posture, cinematic wide "
            "frame, richly textured realism, moody atmosphere."
        ),
    ),
    Scenario(
        scenario_id="greenhouse_rooftop",
        prompt=(
            "A visionary architect walking through a colossal rooftop greenhouse "
            "at sunrise, layered terraces of tropical plants, glass structures "
            "catching warm dawn light, suspended pathways, subtle mist, "
            "detailed environmental storytelling, cinematic wide shot."
        ),
    ),
    Scenario(
        scenario_id="night_market",
        prompt=(
            "A bustling futuristic night market beside a canal, families, cooks, "
            "musicians, and lantern sellers under dense layers of glowing signs, "
            "boats drifting through reflections, intricate stalls, rich depth, "
            "wide cinematic scene packed with life and detail."
        ),
    ),
)


PROFILES: tuple[Profile, ...] = (
    Profile(
        profile_id="base_4step",
        steps=4,
        guidance_scale=4.0,
        scheduler_preset="default",
    ),
    Profile(
        profile_id="base_8step",
        steps=8,
        guidance_scale=4.0,
        scheduler_preset="default",
    ),
    Profile(
        profile_id="lightx_4step",
        steps=4,
        guidance_scale=1.0,
        scheduler_preset="lightning",
        lora_filename="Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
    ),
    Profile(
        profile_id="lightx_8step",
        steps=8,
        guidance_scale=1.0,
        scheduler_preset="lightning",
        lora_filename="Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors",
    ),
    Profile(
        profile_id="wuli_v3_4step",
        steps=4,
        guidance_scale=1.0,
        scheduler_preset="turbo_wuli",
        lora_filename="Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V3.0-bf16.safetensors",
    ),
    Profile(
        profile_id="wuli_v3_8step",
        steps=8,
        guidance_scale=1.0,
        scheduler_preset="turbo_wuli",
        lora_filename="Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V3.0-bf16.safetensors",
    ),
)


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _start_runtime(
    *,
    runtime_home: Path,
    socket_path: Path,
) -> tuple[subprocess.Popen[bytes], Path]:
    runtime_home.mkdir(parents=True, exist_ok=True)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    stdio_log_path = runtime_home / "logs" / "qwen-showcase-stdio.log"
    stdio_log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MLX_RUNTIME_HOME"] = str(runtime_home)
    env["MLX_RUNTIME_UDS_PATH"] = str(socket_path)
    env.pop("MLX_RUNTIME_HTTP_HOST", None)
    env.pop("MLX_RUNTIME_HTTP_PORT", None)
    env.pop("MLX_RUNTIME_HTTP_TOKEN", None)
    with stdio_log_path.open("ab") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "mlxr.core.server"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return process, stdio_log_path


def _wait_for_health(
    *,
    socket_path: Path,
    process: subprocess.Popen[bytes],
    startup_timeout_seconds: float,
) -> httpx.Client:
    transport = httpx.HTTPTransport(uds=str(socket_path), retries=0, trust_env=False)
    client = httpx.Client(
        transport=transport,
        base_url="http://mlxr",
        timeout=httpx.Timeout(30.0),
    )
    deadline = time.time() + startup_timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            client.close()
            raise RuntimeError(
                f"Qwen showcase runtime exited early with code {process.returncode}"
            )
        try:
            response = client.get("/v1/capabilities")
            if response.status_code == 200:
                return client
        except Exception as error:  # pragma: no cover - exercised by real runs
            last_error = error
        time.sleep(0.25)
    client.close()
    raise RuntimeError(
        f"Timed out waiting for runtime health on {socket_path}: {last_error!r}"
    )


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:  # pragma: no cover - safety guard
        process.kill()
        process.wait(timeout=5.0)


def _load_handle_map(runtime_home: Path) -> dict[str, str]:
    manifests = runtime_home.glob("temp/inputs/*/manifest.json")
    mapping: dict[str, str] = {}
    for manifest_path in manifests:
        raw = json.loads(manifest_path.read_text("utf-8"))
        filename = raw.get("filename")
        handle_id = raw.get("handle_id")
        if isinstance(filename, str) and isinstance(handle_id, str):
            mapping[filename] = handle_id
    return mapping


def _sanitize_filename(raw: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in raw
    )


def _submit_workflow(
    client: httpx.Client,
    *,
    model_id: str,
    scenario: Scenario,
    profile: Profile,
    width: int,
    height: int,
    negative_prompt: str,
    lora_handle_map: dict[str, str],
    output_path: Path,
    job_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    references: list[dict[str, Any]] = []
    if profile.lora_filename is not None:
        handle_id = lora_handle_map.get(profile.lora_filename)
        if handle_id is None:
            raise RuntimeError(
                "Missing runtime handle for LoRA file "
                f"{profile.lora_filename!r} in {DEFAULT_RUNTIME_HOME / 'temp/inputs'}"
            )
        references.append(
            {
                "input_handle": handle_id,
                "kind": "lora",
                "role": "reference",
                "metadata": {"strength": 1.0},
            }
        )

    payload = {
        "intent": {
            "model_id": model_id,
            "prompt": scenario.prompt,
            "negative_prompt": negative_prompt,
            "references": references,
            "params": {
                "width": width,
                "height": height,
                "seed": 42,
                "num_inference_steps": profile.steps,
                "guidance_scale": profile.guidance_scale,
            },
            "output": {"artifact_format": "png"},
            "extensions": {
                "qwen_image": {"scheduler_preset": profile.scheduler_preset}
            },
        }
    }
    response = client.post("/v1/workflows/run", json=payload)
    response.raise_for_status()
    run_payload = response.json()
    job_id = str(run_payload["submit"]["job_id"])
    deadline = time.time() + job_timeout_seconds
    record: dict[str, Any] | None = None
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}")
        status.raise_for_status()
        record = status.json()
        state = record.get("state")
        if state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(poll_interval_seconds)
    if record is None:
        raise RuntimeError(f"No job record returned for {job_id}")
    if record.get("state") != "completed":
        raise RuntimeError(
            f"Job {job_id} ended in state {record.get('state')}: {record.get('error')}"
        )
    artifacts = record.get("artifacts", [])
    if not artifacts:
        raise RuntimeError(f"Job {job_id} completed without output artifacts")
    artifact_id = str(artifacts[0]["artifact_id"])
    export = client.post(
        f"/v1/outputs/{artifact_id}/export",
        json={"destination_path": str(output_path), "overwrite": True},
    )
    export.raise_for_status()
    return {"job_id": job_id, "record": record, "export": export.json()}


def _load_job_metrics(runtime_home: Path, job_id: str) -> dict[str, Any]:
    events_path = runtime_home / "jobs" / job_id / "events.jsonl"
    metrics: dict[str, Any] = {"events_path": str(events_path)}
    if not events_path.exists():
        return metrics
    metrics["events"] = []
    for line in events_path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        metrics["events"].append(payload)
        if payload.get("kind") == "job.metrics":
            stage_id = payload.get("data", {}).get("stage_id")
            if isinstance(stage_id, str):
                metrics[f"{stage_id}_metrics"] = payload
    return metrics


def _write_summary_csv(results: list[dict[str, Any]], destination: Path) -> None:
    fieldnames = [
        "scenario_id",
        "profile_id",
        "status",
        "width",
        "height",
        "steps",
        "guidance_scale",
        "scheduler_preset",
        "lora_filename",
        "job_id",
        "prompt_encode_ms",
        "generate_ms",
        "encode_output_ms",
        "output_path",
        "error",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main() -> int:
    session_dir = DEFAULT_RESULTS_ROOT / f"qwen-step-matrix-{_timestamp_label()}"
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_home": str(DEFAULT_RUNTIME_HOME),
        "socket_path": str(DEFAULT_SOCKET_PATH),
        "model_id": DEFAULT_MODEL_ID,
        "resolution": {"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT},
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "notes": [
            "Sequential runner only; no parallel job submission.",
            "Base uses guidance 4.0 as the current truthful high-control recipe.",
            "LightX and Wuli use guidance 1.0 plus family-local scheduler presets.",
            "This matrix reuses already-converted artifacts and already-imported LoRA handles "
            "from the runtime home to avoid duplicate disk pressure.",
        ],
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "profiles": [asdict(profile) for profile in PROFILES],
        "results": [],
    }

    handle_map = _load_handle_map(DEFAULT_RUNTIME_HOME)
    stdio_log_path = DEFAULT_RUNTIME_HOME / "logs" / "qwen-showcase-stdio.log"
    manifest["runtime_stdio_log_path"] = str(stdio_log_path)
    results: list[dict[str, Any]] = []
    try:
        for scenario in SCENARIOS:
            for profile in PROFILES:
                image_name = (
                    f"{scenario.scenario_id}__{profile.profile_id}"
                    f"__{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}.png"
                )
                output_path = session_dir / _sanitize_filename(image_name)
                result_row: dict[str, Any] = {
                    "scenario_id": scenario.scenario_id,
                    "profile_id": profile.profile_id,
                    "status": "pending",
                    "width": DEFAULT_WIDTH,
                    "height": DEFAULT_HEIGHT,
                    "steps": profile.steps,
                    "guidance_scale": profile.guidance_scale,
                    "scheduler_preset": profile.scheduler_preset,
                    "lora_filename": profile.lora_filename,
                    "job_id": "",
                    "prompt_encode_ms": "",
                    "generate_ms": "",
                    "encode_output_ms": "",
                    "output_path": str(output_path),
                    "error": "",
                }
                process, _ = _start_runtime(
                    runtime_home=DEFAULT_RUNTIME_HOME,
                    socket_path=DEFAULT_SOCKET_PATH,
                )
                try:
                    client = _wait_for_health(
                        socket_path=DEFAULT_SOCKET_PATH,
                        process=process,
                        startup_timeout_seconds=DEFAULT_STARTUP_TIMEOUT_SECONDS,
                    )
                    try:
                        run_result = _submit_workflow(
                            client,
                            model_id=DEFAULT_MODEL_ID,
                            scenario=scenario,
                            profile=profile,
                            width=DEFAULT_WIDTH,
                            height=DEFAULT_HEIGHT,
                            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                            lora_handle_map=handle_map,
                            output_path=output_path,
                            job_timeout_seconds=DEFAULT_JOB_TIMEOUT_SECONDS,
                            poll_interval_seconds=DEFAULT_POLL_INTERVAL_SECONDS,
                        )
                    finally:
                        client.close()
                    metrics = _load_job_metrics(
                        DEFAULT_RUNTIME_HOME, run_result["job_id"]
                    )
                    prompt_encode = metrics.get("prompt_encode_metrics", {})
                    generate = metrics.get("generate_metrics", {})
                    encode_output = metrics.get("encode_output_metrics", {})
                    result_row.update(
                        {
                            "status": "completed",
                            "job_id": run_result["job_id"],
                            "prompt_encode_ms": prompt_encode.get("data", {}).get(
                                "duration_ms", ""
                            ),
                            "generate_ms": generate.get("data", {}).get(
                                "duration_ms", ""
                            ),
                            "encode_output_ms": encode_output.get("data", {}).get(
                                "duration_ms", ""
                            ),
                            "metrics": metrics,
                        }
                    )
                except Exception as error:  # pragma: no cover - real run recovery
                    result_row.update({"status": "failed", "error": str(error)})
                finally:
                    _terminate_process(process)
                results.append(result_row)
                manifest["results"] = results
                (session_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
                _write_summary_csv(results, session_dir / "summary.csv")
    finally:
        if DEFAULT_SOCKET_PATH.exists():
            DEFAULT_SOCKET_PATH.unlink()

    completed = sum(1 for result in results if result["status"] == "completed")
    failed = sum(1 for result in results if result["status"] == "failed")
    print(
        json.dumps(
            {
                "session_dir": str(session_dir),
                "completed": completed,
                "failed": failed,
                "summary_csv": str(session_dir / "summary.csv"),
                "manifest_path": str(session_dir / "manifest.json"),
            },
            indent=2,
        )
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

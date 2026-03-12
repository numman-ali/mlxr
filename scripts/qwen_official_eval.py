from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "tmp" / "showcase-runs"
RUNTIME_HOME = Path("/tmp/mlxr-qwen-official-eval")
SOCKET_PATH = RUNTIME_HOME / "control-plane.sock"
STARTUP_TIMEOUT_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 5.0
DEFAULT_WAIT_TIMEOUT_SECONDS = 7200.0

GENERATE_MODEL_ID = "qwen-image-2512-official-eval"
EDIT_MODEL_ID = "qwen-image-edit-2511-official-eval"

GENERATE_BUNDLE = REPO_ROOT / "tmp" / "hf-qwen-image-2512-live"
EDIT_BUNDLE = REPO_ROOT / "tmp" / "hf-qwen-image-edit-2511-live"

LIGHTX_8_LORA = Path(
    "/tmp/mlxr-qwen-lightning-runtime/temp/inputs/"
    "inp_4de1cab751384cdca7b58adc9556978f/"
    "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors"
)
WULI_4_LORA = Path(
    "/tmp/mlxr-qwen-lightning-runtime/temp/inputs/"
    "inp_cced9dead06948eb9ddf0b0c37e16e4d/"
    "Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V3.0-bf16.safetensors"
)
EDIT_LIGHTNING_4_LORA = Path(
    "/tmp/mlxr-qwen-edit-runtime/temp/inputs/"
    "inp_13cceef13dd749aeb14210e5a51d0601/"
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)
EDIT_SOURCE_IMAGE = REPO_ROOT / "tmp" / "manual-runs" / "kenji_on_fire.png"

OFFICIAL_QWEN_PROMPT = (
    "A 20-year-old East Asian girl with delicate, charming features and large, "
    "bright brown eyes-expressive and lively, with a cheerful or subtly smiling "
    "expression. Her naturally wavy long hair is either loose or tied in twin "
    "ponytails. She has fair skin and light makeup accentuating her youthful "
    "freshness. She wears a modern, cute dress or relaxed outfit in bright, soft "
    "colors-lightweight fabric, minimalist cut. She stands indoors at an anime "
    "convention, surrounded by banners, posters, or stalls. Lighting is typical "
    "indoor illumination-no staged lighting-and the image resembles a casual "
    "iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, "
    "youthful charm."
)
OFFICIAL_QWEN_NEGATIVE = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)
SHOWCASE_NEGATIVE = (
    "blurry, low detail, deformed anatomy, broken hands, ugly text artifacts, "
    "watermark, oversaturated, flat lighting, malformed perspective"
)
EDIT_NEGATIVE = " "


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    mode: str
    model_id: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    scheduler_preset: str = "default"
    task: str | None = None
    image_paths: tuple[Path, ...] = ()
    lora_path: Path | None = None
    review: bool = True


GENERATE_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        run_id="official_anime_base50_1664x928",
        mode="generate",
        model_id=GENERATE_MODEL_ID,
        prompt=OFFICIAL_QWEN_PROMPT,
        negative_prompt=OFFICIAL_QWEN_NEGATIVE,
        width=1664,
        height=928,
        num_inference_steps=50,
        guidance_scale=4.0,
    ),
    RunSpec(
        run_id="floating_city_base50_1664x928",
        mode="generate",
        model_id=GENERATE_MODEL_ID,
        prompt=(
            "A vast bioluminescent floating city above the ocean at blue hour, "
            "layered sky bridges, hanging gardens, luminous trains curving "
            "through the skyline, thousands of tiny windows, reflective water "
            "below, cinematic wide establishing shot, ultra-detailed, "
            "awe-inspiring, intricate architecture, atmospheric depth."
        ),
        negative_prompt=SHOWCASE_NEGATIVE,
        width=1664,
        height=928,
        num_inference_steps=50,
        guidance_scale=4.0,
    ),
    RunSpec(
        run_id="floating_city_lightx8_1664x928",
        mode="generate",
        model_id=GENERATE_MODEL_ID,
        prompt=(
            "A vast bioluminescent floating city above the ocean at blue hour, "
            "layered sky bridges, hanging gardens, luminous trains curving "
            "through the skyline, thousands of tiny windows, reflective water "
            "below, cinematic wide establishing shot, ultra-detailed, "
            "awe-inspiring, intricate architecture, atmospheric depth."
        ),
        negative_prompt=SHOWCASE_NEGATIVE,
        width=1664,
        height=928,
        num_inference_steps=8,
        guidance_scale=1.0,
        scheduler_preset="lightning",
        lora_path=LIGHTX_8_LORA,
    ),
    RunSpec(
        run_id="floating_city_wuli4_1664x928",
        mode="generate",
        model_id=GENERATE_MODEL_ID,
        prompt=(
            "A vast bioluminescent floating city above the ocean at blue hour, "
            "layered sky bridges, hanging gardens, luminous trains curving "
            "through the skyline, thousands of tiny windows, reflective water "
            "below, cinematic wide establishing shot, ultra-detailed, "
            "awe-inspiring, intricate architecture, atmospheric depth."
        ),
        negative_prompt=SHOWCASE_NEGATIVE,
        width=1664,
        height=928,
        num_inference_steps=4,
        guidance_scale=1.0,
        scheduler_preset="turbo_wuli",
        lora_path=WULI_4_LORA,
    ),
)

EDIT_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        run_id="kenji_edit_base40_1344x768",
        mode="edit",
        model_id=EDIT_MODEL_ID,
        task="image.edit",
        prompt=(
            "Keep the same young swordsman, flaming sword, and dramatic stance, "
            "but transform the scene into a rain-soaked neon alley at night with "
            "glowing signage, drifting steam, wet reflections, and vivid blue and "
            "orange cinematic lighting."
        ),
        negative_prompt=EDIT_NEGATIVE,
        width=1344,
        height=768,
        num_inference_steps=40,
        guidance_scale=4.0,
        image_paths=(EDIT_SOURCE_IMAGE,),
    ),
    RunSpec(
        run_id="kenji_edit_lightning4_1344x768",
        mode="edit",
        model_id=EDIT_MODEL_ID,
        task="image.edit",
        prompt=(
            "Keep the same young swordsman, flaming sword, and dramatic stance, "
            "but transform the scene into a rain-soaked neon alley at night with "
            "glowing signage, drifting steam, wet reflections, and vivid blue and "
            "orange cinematic lighting."
        ),
        negative_prompt=EDIT_NEGATIVE,
        width=1344,
        height=768,
        num_inference_steps=4,
        guidance_scale=1.0,
        scheduler_preset="lightning",
        image_paths=(EDIT_SOURCE_IMAGE,),
        lora_path=EDIT_LIGHTNING_4_LORA,
    ),
)

RUNS_BY_ID: dict[str, RunSpec] = {
    spec.run_id: spec for spec in (*GENERATE_RUNS, *EDIT_RUNS)
}


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_subprocess(
    args: list[str],
    *,
    env: dict[str, str],
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _start_runtime() -> tuple[subprocess.Popen[bytes], Path]:
    RUNTIME_HOME.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    stdio_log_path = RUNTIME_HOME / "logs" / "qwen-official-eval-stdio.log"
    stdio_log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MLX_RUNTIME_HOME"] = str(RUNTIME_HOME)
    env["MLX_RUNTIME_UDS_PATH"] = str(SOCKET_PATH)
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


def _wait_for_health(process: subprocess.Popen[bytes]) -> None:
    transport = httpx.HTTPTransport(uds=str(SOCKET_PATH), retries=0, trust_env=False)
    with httpx.Client(
        transport=transport,
        base_url="http://mlxr",
        timeout=httpx.Timeout(30.0),
    ) as client:
        deadline = time.time() + STARTUP_TIMEOUT_SECONDS
        last_error: Exception | None = None
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Qwen official eval runtime exited early with code {process.returncode}"
                )
            try:
                response = client.get("/v1/capabilities")
                if response.status_code == 200:
                    return
            except Exception as error:  # pragma: no cover - real runtime path
                last_error = error
            time.sleep(0.25)
    raise RuntimeError(
        f"Timed out waiting for runtime health on {SOCKET_PATH}: {last_error!r}"
    )


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _register_bundle(
    *,
    env: dict[str, str],
    bundle_path: Path,
    family: str,
    family_variant: str,
    model_id: str,
) -> dict[str, Any]:
    args = [
        "uv",
        "run",
        "python",
        "scripts/runtime_register_bundle.py",
        "--bundle-path",
        str(bundle_path),
        "--family",
        family,
        "--family-variant",
        family_variant,
        "--model-id",
        model_id,
        "--uds-path",
        str(SOCKET_PATH),
    ]
    completed = _run_subprocess(args, env=env)
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("Expected bundle registration to return a JSON object")
    return payload


def _run_generate_cli(
    *,
    env: dict[str, str],
    spec: RunSpec,
    export_path: Path,
) -> dict[str, Any]:
    args = [
        "uv",
        "run",
        "mlxr",
        "--uds-path",
        str(SOCKET_PATH),
        "generate",
        "--model-id",
        spec.model_id,
        "--prompt",
        spec.prompt,
        "--negative-prompt",
        spec.negative_prompt,
        "--width",
        str(spec.width),
        "--height",
        str(spec.height),
        "--seed",
        "42",
        "--num-inference-steps",
        str(spec.num_inference_steps),
        "--guidance-scale",
        str(spec.guidance_scale),
        "--artifact-format",
        "png",
        "--wait",
        "--timeout-seconds",
        str(DEFAULT_WAIT_TIMEOUT_SECONDS),
        "--poll-interval-seconds",
        str(POLL_INTERVAL_SECONDS),
        "--export-path",
        str(export_path),
        "--overwrite-export",
        "--extensions-json",
        json.dumps({"qwen_image": {"scheduler_preset": spec.scheduler_preset}}),
    ]
    if spec.task is not None:
        args.extend(["--task", spec.task])
    for image_path in spec.image_paths:
        args.extend(["--image", str(image_path)])
    if spec.lora_path is not None:
        args.extend(["--lora", str(spec.lora_path)])
    completed = _run_subprocess(args, env=env)
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("Expected mlxr generate to return a JSON object")
    return payload


def _load_job_metrics(job_id: str) -> dict[str, Any]:
    events_path = RUNTIME_HOME / "jobs" / job_id / "events.jsonl"
    metrics: dict[str, Any] = {"events_path": str(events_path)}
    if not events_path.exists():
        return metrics
    for line in events_path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("kind") == "job.metrics":
            stage_id = payload.get("data", {}).get("stage_id")
            if isinstance(stage_id, str):
                metrics[f"{stage_id}_metrics"] = payload
    return metrics


def _review_image(
    *,
    env: dict[str, str],
    image_path: Path,
    prompt_text: str,
    review_dir: Path,
    reference_images: tuple[Path, ...] = (),
) -> dict[str, Any] | None:
    args = [
        "uv",
        "run",
        "python",
        "scripts/gemini_review_image.py",
        str(image_path),
        "--save-dir",
        str(review_dir),
        "--prompt-text",
        prompt_text,
        "--output-format",
        "json",
    ]
    for reference_image in reference_images:
        args.extend(["--reference-image", str(reference_image), "source"])
    try:
        completed = _run_subprocess(args, env=env)
    except subprocess.CalledProcessError as error:  # pragma: no cover - external CLI
        return {
            "status": "failed",
            "stdout": error.stdout,
            "stderr": error.stderr,
        }
    return {"status": "completed", "payload": json.loads(completed.stdout)}


def _session_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MLX_RUNTIME_HOME"] = str(RUNTIME_HOME)
    env["MLX_RUNTIME_UDS_PATH"] = str(SOCKET_PATH)
    return env


def _write_summary_csv(results: list[dict[str, Any]], destination: Path) -> None:
    fieldnames = [
        "run_id",
        "mode",
        "status",
        "width",
        "height",
        "steps",
        "guidance_scale",
        "scheduler_preset",
        "job_id",
        "prompt_encode_ms",
        "generate_ms",
        "encode_output_ms",
        "output_path",
        "review_status",
        "review_verdict",
        "error",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def _output_name(spec: RunSpec) -> str:
    return f"{spec.run_id}.png"


def _jsonable_spec(spec: RunSpec) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "mode": spec.mode,
        "model_id": spec.model_id,
        "prompt": spec.prompt,
        "negative_prompt": spec.negative_prompt,
        "width": spec.width,
        "height": spec.height,
        "num_inference_steps": spec.num_inference_steps,
        "guidance_scale": spec.guidance_scale,
        "scheduler_preset": spec.scheduler_preset,
        "task": spec.task,
        "image_paths": [str(path) for path in spec.image_paths],
        "lora_path": str(spec.lora_path) if spec.lora_path is not None else None,
        "review": spec.review,
    }


def _run_group(
    *,
    session_dir: Path,
    specs: tuple[RunSpec, ...],
    bundle_path: Path,
    model_id: str,
    family: str,
    family_variant: str,
) -> list[dict[str, Any]]:
    env = _session_env()
    process, stdio_log_path = _start_runtime()
    results: list[dict[str, Any]] = []
    try:
        _wait_for_health(process)
        registration = _register_bundle(
            env=env,
            bundle_path=bundle_path,
            family=family,
            family_variant=family_variant,
            model_id=model_id,
        )
        for spec in specs:
            output_path = session_dir / _output_name(spec)
            row: dict[str, Any] = {
                "run_id": spec.run_id,
                "mode": spec.mode,
                "status": "pending",
                "width": spec.width,
                "height": spec.height,
                "steps": spec.num_inference_steps,
                "guidance_scale": spec.guidance_scale,
                "scheduler_preset": spec.scheduler_preset,
                "output_path": str(output_path),
                "review_status": "",
                "review_verdict": "",
                "error": "",
            }
            try:
                payload = _run_generate_cli(env=env, spec=spec, export_path=output_path)
                job_id = str(payload["job_id"])
                metrics = _load_job_metrics(job_id)
                row.update(
                    {
                        "status": "completed",
                        "job_id": job_id,
                        "prompt_encode_ms": metrics.get("prompt_encode_metrics", {})
                        .get("data", {})
                        .get("duration_ms", ""),
                        "generate_ms": metrics.get("generate_metrics", {})
                        .get("data", {})
                        .get("duration_ms", ""),
                        "encode_output_ms": metrics.get("encode_output_metrics", {})
                        .get("data", {})
                        .get("duration_ms", ""),
                        "metrics": metrics,
                        "cli_payload": payload,
                    }
                )
                if spec.review:
                    review_dir = session_dir / f"{spec.run_id}-review"
                    review = _review_image(
                        env=env,
                        image_path=output_path,
                        prompt_text=spec.prompt,
                        review_dir=review_dir,
                        reference_images=spec.image_paths,
                    )
                    row["review"] = review
                    if review is not None:
                        row["review_status"] = str(review.get("status", ""))
                        if review.get("status") == "completed":
                            payload = review.get("payload", {})
                            parsed = (
                                payload.get("parsed_review", {})
                                if isinstance(payload, dict)
                                else {}
                            )
                            if not parsed and isinstance(payload, dict):
                                parsed = payload
                            if isinstance(parsed, dict):
                                row["review_verdict"] = str(parsed.get("verdict", ""))
            except Exception as error:  # pragma: no cover - real-run recovery
                row.update({"status": "failed", "error": str(error)})
            results.append(row)
            (session_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "runtime_home": str(RUNTIME_HOME),
                        "socket_path": str(SOCKET_PATH),
                        "runtime_stdio_log_path": str(stdio_log_path),
                        "registration": registration,
                        "results": results,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _write_summary_csv(results, session_dir / "summary.csv")
    finally:
        _terminate_process(process)
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run heavyweight Qwen generation/edit evaluations through the canonical "
            "mlxr CLI path."
        )
    )
    parser.add_argument(
        "--run-id",
        dest="run_ids",
        action="append",
        default=[],
        help="Repeat to run only specific named specs.",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip Gemini image review for faster targeted reruns.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for required_path in (
        GENERATE_BUNDLE,
        EDIT_BUNDLE,
        LIGHTX_8_LORA,
        WULI_4_LORA,
        EDIT_LIGHTNING_4_LORA,
        EDIT_SOURCE_IMAGE,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required evaluation asset is missing: {required_path}"
            )
    if args.run_ids:
        missing = [run_id for run_id in args.run_ids if run_id not in RUNS_BY_ID]
        if missing:
            raise KeyError(f"Unknown qwen_official_eval run ids: {', '.join(missing)}")
        selected_specs = tuple(RUNS_BY_ID[run_id] for run_id in args.run_ids)
    else:
        selected_specs = ()

    def _maybe_disable_review(spec: RunSpec) -> RunSpec:
        if not args.skip_review:
            return spec
        return RunSpec(
            run_id=spec.run_id,
            mode=spec.mode,
            model_id=spec.model_id,
            prompt=spec.prompt,
            negative_prompt=spec.negative_prompt,
            width=spec.width,
            height=spec.height,
            num_inference_steps=spec.num_inference_steps,
            guidance_scale=spec.guidance_scale,
            scheduler_preset=spec.scheduler_preset,
            task=spec.task,
            image_paths=spec.image_paths,
            lora_path=spec.lora_path,
            review=False,
        )

    base_generate_specs = (
        tuple(spec for spec in selected_specs if spec.mode == "generate")
        if selected_specs
        else GENERATE_RUNS
    )
    base_edit_specs = (
        tuple(spec for spec in selected_specs if spec.mode == "edit")
        if selected_specs
        else EDIT_RUNS
    )
    generation_specs = tuple(
        _maybe_disable_review(spec) for spec in base_generate_specs
    )
    edit_specs = tuple(_maybe_disable_review(spec) for spec in base_edit_specs)

    session_dir = RESULTS_ROOT / f"qwen-official-eval-{_timestamp_label()}"
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session_dir),
        "notes": [
            "Runs are sequential only.",
            "One runtime group is used for generation and one for edit so the heavy model stacks do not overlap.",
            "Generation runs use the canonical mlxr CLI and official or near-official Qwen recipes.",
            "Edit runs use the canonical mlxr CLI with the base edit recipe first, then the Lightning fast lane.",
        ],
        "generation_specs": [_jsonable_spec(spec) for spec in generation_specs],
        "edit_specs": [_jsonable_spec(spec) for spec in edit_specs],
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")

    generation_results = _run_group(
        session_dir=session_dir,
        specs=generation_specs,
        bundle_path=GENERATE_BUNDLE,
        model_id=GENERATE_MODEL_ID,
        family="qwen_image",
        family_variant="qwen-image-2512",
    )
    edit_results = _run_group(
        session_dir=session_dir,
        specs=edit_specs,
        bundle_path=EDIT_BUNDLE,
        model_id=EDIT_MODEL_ID,
        family="qwen_image",
        family_variant="qwen-image-edit-2511",
    )

    final_manifest = {
        **manifest,
        "generation_results": generation_results,
        "edit_results": edit_results,
    }
    (session_dir / "manifest.json").write_text(
        json.dumps(final_manifest, indent=2),
        encoding="utf-8",
    )

    all_results = [*generation_results, *edit_results]
    _write_summary_csv(all_results, session_dir / "summary.csv")
    failed = sum(1 for result in all_results if result["status"] != "completed")
    print(
        json.dumps(
            {
                "session_dir": str(session_dir),
                "summary_csv": str(session_dir / "summary.csv"),
                "manifest_path": str(session_dir / "manifest.json"),
                "completed": len(all_results) - failed,
                "failed": failed,
            },
            indent=2,
        )
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the heavyweight LTX debug and fidelity smoke workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "manual-runs"
DEFAULT_RUN_NAME = "ltx-fidelity-debug"
DEFAULT_TMUX_SESSION = "mlxr-ltx-debug"
DEFAULT_HEARTBEAT_SECONDS = 2.0
DEFAULT_PROFILE_NAME = "safe-smoke"
DEFAULT_MAX_ACTIVE_GB = 24.0
DEFAULT_MAX_PEAK_GB = 32.0
DEFAULT_MAX_SECONDS = 180.0
DEFAULT_MEMORY_POLL_SECONDS = 0.5
KNOWN_REAL_BACKENDS = frozenset(
    {"mlxr_ltx_distilled_two_stage", "mlxr_ltx_distilled_two_stage_video_only"}
)
KNOWN_REAL_PIPELINE_KINDS = frozenset(
    {"distilled_two_stage", "distilled_two_stage_video_only"}
)
PROFILE_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "safe-smoke": (256, 160, 17, 24),
    "visual-gate": (384, 224, 17, 24),
    "coherence": (768, 512, 33, 24),
    "recommended": (1536, 1024, 121, 24),
    "hq-first-pass": (1920, 1088, 65, 24),
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class PromptEncoderLike(Protocol):
    def encode(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        return_audio_context: bool = True,
    ) -> object: ...

    def close(self) -> None: ...


class GeneratedVideoLike(Protocol):
    frames: np.ndarray
    fps: int
    seed: int
    backend: str
    conditioning_count: int
    prompt_signature: str
    metadata: dict[str, object]


class VideoGeneratorLike(Protocol):
    def generate(
        self,
        *,
        prompt_context: object,
        conditioning_inputs: tuple[object, ...],
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int | None = None,
    ) -> GeneratedVideoLike: ...

    def close(self) -> None: ...


def _is_preview_backend(
    backend: str, metadata: dict[str, object] | None = None
) -> bool:
    if "preview" in backend.lower():
        return True
    if metadata is None:
        return False
    pipeline_kind = metadata.get("pipeline_kind")
    return isinstance(pipeline_kind, str) and "preview" in pipeline_kind.lower()


def _assert_real_backend(
    backend: str, metadata: dict[str, object] | None = None
) -> None:
    if _is_preview_backend(backend, metadata):
        raise RuntimeError(
            "LTX debug smoke fell back to the preview backend; this run is not valid "
            "evidence for real generation"
        )
    if backend not in KNOWN_REAL_BACKENDS:
        raise RuntimeError(
            f"LTX debug smoke reported unexpected backend '{backend}'; this run is not valid evidence for the known real bridge"
        )
    if metadata is None:
        raise RuntimeError(
            "LTX debug smoke is missing backend metadata; this run is not valid evidence for the known real bridge"
        )
    pipeline_kind = metadata.get("pipeline_kind")
    if pipeline_kind not in KNOWN_REAL_PIPELINE_KINDS:
        raise RuntimeError(
            "LTX debug smoke reported unexpected pipeline metadata; this run is not valid evidence for the known real bridge"
        )


def _default_prompt_encoder_factory(
    *, checkpoint_path: Path, text_encoder_path: Path
) -> PromptEncoderLike:
    module = importlib.import_module("mlxr.families.ltx.prompt_encoding")
    factory: Callable[..., PromptEncoderLike] = getattr(module, "create_prompt_encoder")
    return factory(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
    )


def _default_video_generator_factory(
    *, checkpoint_path: Path, spatial_upsampler_path: Path, audio_enabled: bool
) -> VideoGeneratorLike:
    module = importlib.import_module("mlxr.families.ltx.generation")
    factory: Callable[..., VideoGeneratorLike] = getattr(
        module, "create_video_generator"
    )
    return factory(
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        audio_enabled=audio_enabled,
    )


def _default_mp4_encoder(video: GeneratedVideoLike, output_path: Path) -> None:
    module = importlib.import_module("mlxr.families.ltx.generation")
    encoder: Callable[[GeneratedVideoLike, Path], None] = getattr(
        module, "encode_mp4_video"
    )
    encoder(video, output_path)


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    checkpoint_path: Path
    spatial_upsampler_path: Path
    text_encoder_path: Path
    artifact_root: Path | None = None


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    artifact_paths: ArtifactPaths
    prompt: str
    profile_name: str
    width: int
    height: int
    num_frames: int
    fps: int
    seed: int
    output_root: Path
    run_name: str
    stage_debug: bool
    trace: bool
    trace_sync: bool
    clean_lifecycle: bool
    video_only: bool
    backend_progress: bool
    heartbeat_seconds: float
    max_active_gb: float
    max_peak_gb: float
    max_seconds: float
    memory_poll_seconds: float
    negative_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class RunBundlePaths:
    run_dir: Path
    debug_dir: Path
    video_path: Path
    frame_path: Path
    frame_mid_path: Path
    frame_last_path: Path
    manifest_path: Path
    trace_path: Path
    detached_console_log_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real-weight LTX debug smoke with clean lifecycle and optional "
            "stage dumps."
        )
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--spatial-upsampler-path", type=Path)
    parser.add_argument("--text-encoder-path", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_PRESETS.keys()),
        default=DEFAULT_PROFILE_NAME,
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--seed", type=int, default=165783600)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--stage-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit stage1/post_x2/final debug frames via backend env flags.",
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record structured generation trace data during the debug smoke.",
    )
    parser.add_argument(
        "--trace-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force MLX synchronization around traced forward spans for more precise "
            "timings at extra runtime cost."
        ),
    )
    parser.add_argument(
        "--clean-lifecycle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode the prompt, close the encoder, clear cache, then load generate.",
    )
    parser.add_argument(
        "--video-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the owned LTX distilled bridge without the audio branch.",
    )
    parser.add_argument(
        "--backend-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable backend progress prints from the LTX generation bridge.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
    )
    parser.add_argument(
        "--max-active-gb",
        type=float,
        default=DEFAULT_MAX_ACTIVE_GB,
        help="Abort the smoke if MLX active memory exceeds this many GiB.",
    )
    parser.add_argument(
        "--max-peak-gb",
        type=float,
        default=DEFAULT_MAX_PEAK_GB,
        help="Abort the smoke if MLX peak memory exceeds this many GiB.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Abort the smoke if total runtime exceeds this many seconds.",
    )
    parser.add_argument(
        "--memory-poll-seconds",
        type=float,
        default=DEFAULT_MEMORY_POLL_SECONDS,
        help="How often the watchdog checks memory and wall-clock limits.",
    )
    parser.add_argument(
        "--print-tmux-command",
        action="store_true",
        help="Print a detached tmux command for this exact smoke run and exit.",
    )
    parser.add_argument(
        "--tmux-session-name",
        default=DEFAULT_TMUX_SESSION,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    _validate_args(args, parser)
    return args


def build_config(args: argparse.Namespace) -> SmokeConfig:
    artifact_paths = resolve_artifact_paths(args)
    profile_name = str(args.profile)
    width, height, num_frames, fps = _resolved_profile_shape(args)
    return SmokeConfig(
        artifact_paths=artifact_paths,
        prompt=args.prompt,
        negative_prompt=(
            str(getattr(args, "negative_prompt", "")).strip()
            if getattr(args, "negative_prompt", None)
            else None
        ),
        profile_name=profile_name,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        seed=int(args.seed),
        output_root=Path(args.output_root),
        run_name=str(args.run_name),
        stage_debug=bool(args.stage_debug),
        trace=bool(args.trace),
        trace_sync=bool(args.trace_sync),
        clean_lifecycle=bool(args.clean_lifecycle),
        video_only=bool(args.video_only),
        backend_progress=bool(args.backend_progress),
        heartbeat_seconds=float(args.heartbeat_seconds),
        max_active_gb=float(args.max_active_gb),
        max_peak_gb=float(args.max_peak_gb),
        max_seconds=float(args.max_seconds),
        memory_poll_seconds=float(args.memory_poll_seconds),
    )


def resolve_artifact_paths(args: argparse.Namespace) -> ArtifactPaths:
    if args.artifact_root is not None:
        artifact_root = Path(args.artifact_root)
        checkpoint_path = (
            artifact_root / "checkpoint" / "ltx-2.3-22b-distilled.safetensors"
        )
        spatial_upsampler_path = (
            artifact_root
            / "spatial_upsampler"
            / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        )
        text_encoder_path = artifact_root / "text_encoder"
        return _validate_artifact_paths(
            ArtifactPaths(
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=spatial_upsampler_path,
                text_encoder_path=text_encoder_path,
                artifact_root=artifact_root,
            )
        )

    return _validate_artifact_paths(
        ArtifactPaths(
            checkpoint_path=Path(args.checkpoint_path),
            spatial_upsampler_path=Path(args.spatial_upsampler_path),
            text_encoder_path=Path(args.text_encoder_path),
            artifact_root=None,
        )
    )


def build_run_bundle_paths(
    config: SmokeConfig,
    *,
    started_at: datetime,
) -> RunBundlePaths:
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(config.run_name)
    run_dir = config.output_root / f"{timestamp}-{slug}"
    return RunBundlePaths(
        run_dir=run_dir,
        debug_dir=run_dir / "debug",
        video_path=run_dir
        / f"{slug}_{config.width}x{config.height}_{config.num_frames}f.mp4",
        frame_path=run_dir / "frame_0001.png",
        frame_mid_path=run_dir / "frame_mid.png",
        frame_last_path=run_dir / "frame_last.png",
        manifest_path=run_dir / "run_manifest.json",
        trace_path=run_dir / "trace.json",
        detached_console_log_path=run_dir / "run.log",
    )


def build_tmux_command(
    config: SmokeConfig,
    *,
    session_name: str = DEFAULT_TMUX_SESSION,
    log_path: Path | None = None,
) -> str:
    resolved_log_path = log_path or (
        config.output_root / f"{_slugify(config.run_name)}-tmux.log"
    )
    cli_args = _cli_args_from_config(config)
    inner = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"{' '.join(shlex.quote(arg) for arg in cli_args)} "
        f"> {shlex.quote(str(resolved_log_path))} 2>&1"
    )
    return f"tmux new-session -d -s {shlex.quote(session_name)} {shlex.quote(inner)}"


def run_smoke(
    config: SmokeConfig,
    *,
    now: Callable[[], datetime] = _now_utc,
    prompt_encoder_factory: Callable[
        ..., PromptEncoderLike
    ] = _default_prompt_encoder_factory,
    video_generator_factory: Callable[
        ..., VideoGeneratorLike
    ] = _default_video_generator_factory,
    mp4_encoder: Callable[[GeneratedVideoLike, Path], None] = _default_mp4_encoder,
    cache_clearer: Callable[[], None] = mx.clear_cache,
    active_memory_getter: Callable[[], int] = mx.get_active_memory,
    peak_memory_getter: Callable[[], int] = mx.get_peak_memory,
    peak_memory_resetter: Callable[[], object] = mx.reset_peak_memory,
    memory_limit_setter: Callable[[int], object] = mx.set_memory_limit,
    monotonic: Callable[[], float] = time.perf_counter,
    abort_process: Callable[[int], None] | None = None,
) -> dict[str, object]:
    started_at = now()
    bundle = build_run_bundle_paths(config, started_at=started_at)
    bundle.run_dir.mkdir(parents=True, exist_ok=True)
    if config.stage_debug:
        bundle.debug_dir.mkdir(parents=True, exist_ok=True)

    timings_ms: dict[str, float] = {}
    manifest: dict[str, object] = {
        "schema_version": "0.1.0",
        "status": "running",
        "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "profile_name": config.profile_name,
        "seed": config.seed,
        "width": config.width,
        "height": config.height,
        "num_frames": config.num_frames,
        "fps": config.fps,
        "clean_lifecycle": config.clean_lifecycle,
        "stage_debug": config.stage_debug,
        "trace": config.trace,
        "trace_sync": config.trace_sync,
        "video_only": config.video_only,
        "backend_progress": config.backend_progress,
        "artifact_source": {
            "artifact_root": (
                str(config.artifact_paths.artifact_root)
                if config.artifact_paths.artifact_root is not None
                else None
            ),
            "checkpoint_path": str(config.artifact_paths.checkpoint_path),
            "spatial_upsampler_path": str(config.artifact_paths.spatial_upsampler_path),
            "text_encoder_path": str(config.artifact_paths.text_encoder_path),
        },
        "outputs": {
            "run_dir": str(bundle.run_dir),
            "debug_dir": str(bundle.debug_dir) if config.stage_debug else None,
            "video_path": str(bundle.video_path),
            "frame_path": str(bundle.frame_path),
            "frame_mid_path": str(bundle.frame_mid_path),
            "frame_last_path": str(bundle.frame_last_path),
            "trace_path": str(bundle.trace_path),
            "detached_console_log_path": str(bundle.detached_console_log_path),
        },
        "timings_ms": timings_ms,
        "guardrails": {
            "max_active_gb": config.max_active_gb,
            "max_peak_gb": config.max_peak_gb,
            "max_seconds": config.max_seconds,
            "memory_poll_seconds": config.memory_poll_seconds,
        },
    }
    _write_json(bundle.manifest_path, manifest)

    encoder: PromptEncoderLike | None = None
    generator: VideoGeneratorLike | None = None
    prompt_context: object | None = None
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    watchdog_stop = threading.Event()
    watchdog_thread: threading.Thread | None = None
    abort_state: dict[str, object] = {}
    started_at_perf = monotonic()
    if abort_process is None:
        abort_process = _default_abort_process

    def heartbeat() -> None:
        while not heartbeat_stop.wait(config.heartbeat_seconds):
            print("[heartbeat] still generating", flush=True)

    def watchdog() -> None:
        while not watchdog_stop.wait(config.memory_poll_seconds):
            active_bytes = int(active_memory_getter())
            peak_bytes = int(peak_memory_getter())
            elapsed_seconds = monotonic() - started_at_perf
            active_gb = _bytes_to_gb(active_bytes)
            peak_gb = _bytes_to_gb(peak_bytes)
            if active_gb > config.max_active_gb:
                _record_abort(
                    manifest=manifest,
                    bundle=bundle,
                    timings_ms=timings_ms,
                    reason="active_memory_limit_exceeded",
                    details={
                        "active_gb": round(active_gb, 3),
                        "peak_gb": round(peak_gb, 3),
                        "elapsed_seconds": round(elapsed_seconds, 3),
                    },
                    abort_state=abort_state,
                )
                abort_process(137)
            if peak_gb > config.max_peak_gb:
                _record_abort(
                    manifest=manifest,
                    bundle=bundle,
                    timings_ms=timings_ms,
                    reason="peak_memory_limit_exceeded",
                    details={
                        "active_gb": round(active_gb, 3),
                        "peak_gb": round(peak_gb, 3),
                        "elapsed_seconds": round(elapsed_seconds, 3),
                    },
                    abort_state=abort_state,
                )
                abort_process(137)
            if elapsed_seconds > config.max_seconds:
                _record_abort(
                    manifest=manifest,
                    bundle=bundle,
                    timings_ms=timings_ms,
                    reason="wall_clock_limit_exceeded",
                    details={
                        "active_gb": round(active_gb, 3),
                        "peak_gb": round(peak_gb, 3),
                        "elapsed_seconds": round(elapsed_seconds, 3),
                    },
                    abort_state=abort_state,
                )
                abort_process(124)

    env_updates: dict[str, str | None] = {}
    if config.stage_debug:
        env_updates["MLXR_LTX_DEBUG_STAGE_DUMPS_DIR"] = str(bundle.debug_dir)
    if config.trace:
        env_updates["MLXR_LTX_DEBUG_TRACE"] = "1"
    if config.trace_sync:
        env_updates["MLXR_LTX_DEBUG_TRACE_SYNC"] = "1"
    if config.backend_progress:
        env_updates["MLXR_LTX_DEBUG_PROGRESS"] = "1"
    if config.video_only:
        env_updates["MLXR_LTX_DEBUG_VIDEO_ONLY"] = "1"

    try:
        with temporary_env(env_updates):
            peak_memory_resetter()
            memory_limit_setter(_bytes_from_gb(config.max_active_gb))
            print("creating encoder", flush=True)
            encoder = prompt_encoder_factory(
                checkpoint_path=config.artifact_paths.checkpoint_path,
                text_encoder_path=config.artifact_paths.text_encoder_path,
            )

            prompt_started = time.perf_counter()
            print("encoding prompt", flush=True)
            prompt_context = encoder.encode(
                config.prompt,
                return_audio_context=not config.video_only,
                negative_prompt=config.negative_prompt,
            )
            timings_ms["prompt_encode"] = _elapsed_ms(prompt_started)

            if config.clean_lifecycle:
                print("closing prompt encoder", flush=True)
                _safe_close(encoder)
                encoder = None
                cache_clearer()

            print("creating generator", flush=True)
            generator = video_generator_factory(
                checkpoint_path=config.artifact_paths.checkpoint_path,
                spatial_upsampler_path=config.artifact_paths.spatial_upsampler_path,
                audio_enabled=not config.video_only,
            )

            print("generating video", flush=True)
            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            watchdog_thread = threading.Thread(target=watchdog, daemon=True)
            watchdog_thread.start()
            generate_started = time.perf_counter()
            generated_video = generator.generate(
                prompt_context=prompt_context,
                conditioning_inputs=(),
                width=config.width,
                height=config.height,
                num_frames=config.num_frames,
                fps=config.fps,
                seed=config.seed,
            )
            timings_ms["generate"] = _elapsed_ms(generate_started)
            _assert_real_backend(generated_video.backend, generated_video.metadata)

            heartbeat_stop.set()
            encode_started = time.perf_counter()
            print("encoding mp4", flush=True)
            mp4_encoder(generated_video, bundle.video_path)
            timings_ms["mp4_encode"] = _elapsed_ms(encode_started)

            if len(generated_video.frames) == 0:
                raise ValueError("Generated video has no frames")
            _write_review_frames(generated_video.frames, bundle)

            manifest["status"] = "success"
            manifest["backend"] = generated_video.backend
            manifest["backend_verified"] = True
            manifest["prompt_signature"] = generated_video.prompt_signature
            manifest["conditioning_count"] = generated_video.conditioning_count
            manifest["video_metadata"] = generated_video.metadata
            trace_metadata = generated_video.metadata.get("trace")
            if isinstance(trace_metadata, dict):
                _write_json(bundle.trace_path, trace_metadata)
                manifest["trace_summary"] = trace_metadata.get("summary")
            manifest["video_fps"] = generated_video.fps
            manifest["video_seed"] = generated_video.seed
            manifest["review_frame_paths"] = {
                "frame_0001": str(bundle.frame_path),
                "frame_mid": str(bundle.frame_mid_path),
                "frame_last": str(bundle.frame_last_path),
            }
            print(
                json.dumps(
                    {
                        "video_path": str(bundle.video_path),
                        "frame_path": str(bundle.frame_path),
                        "frame_mid_path": str(bundle.frame_mid_path),
                        "frame_last_path": str(bundle.frame_last_path),
                        "manifest_path": str(bundle.manifest_path),
                        "trace_path": (
                            str(bundle.trace_path)
                            if bundle.trace_path.exists()
                            else None
                        ),
                    },
                    indent=2,
                ),
                flush=True,
            )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["backend_verified"] = False
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        heartbeat_stop.set()
        watchdog_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.1)
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=0.1)
        cleanup_errors: list[str] = []
        if generator is not None:
            _safe_close(generator, cleanup_errors)
        if encoder is not None:
            _safe_close(encoder, cleanup_errors)
        cache_clearer()
        manifest["observed_memory_gb"] = {
            "active": round(_bytes_to_gb(int(active_memory_getter())), 3),
            "peak": round(_bytes_to_gb(int(peak_memory_getter())), 3),
        }
        if cleanup_errors:
            manifest["cleanup_errors"] = cleanup_errors
        manifest["completed_at_utc"] = now().isoformat().replace("+00:00", "Z")
        _write_json(bundle.manifest_path, manifest)

    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    if args.print_tmux_command:
        print(
            build_tmux_command(
                config,
                session_name=args.tmux_session_name,
            )
        )
        return 0
    run_smoke(config)
    return 0


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.artifact_root is None:
        explicit = (
            args.checkpoint_path,
            args.spatial_upsampler_path,
            args.text_encoder_path,
        )
        if any(value is None for value in explicit):
            parser.error(
                "Provide either --artifact-root or all of --checkpoint-path, "
                "--spatial-upsampler-path, and --text-encoder-path."
            )
    width, height, num_frames, fps = _resolved_profile_shape(args)
    if width < 32 or height < 32:
        parser.error("LTX debug smokes require width and height >= 32.")
    if num_frames < 1 or num_frames % 8 != 1:
        parser.error("LTX debug smokes require num-frames to satisfy 8n+1.")
    if fps < 1:
        parser.error("LTX debug smokes require fps >= 1.")
    if args.heartbeat_seconds <= 0:
        parser.error("Heartbeat seconds must be > 0.")
    if args.max_active_gb <= 0:
        parser.error("Max active GiB must be > 0.")
    if args.max_peak_gb <= 0:
        parser.error("Max peak GiB must be > 0.")
    if args.max_peak_gb < args.max_active_gb:
        parser.error("Max peak GiB must be >= max active GiB.")
    if args.max_seconds <= 0:
        parser.error("Max seconds must be > 0.")
    if args.memory_poll_seconds <= 0:
        parser.error("Memory poll seconds must be > 0.")
    if args.trace_sync and not args.trace:
        parser.error("--trace-sync requires --trace.")


def _resolved_profile_shape(args: argparse.Namespace) -> tuple[int, int, int, int]:
    profile_name = str(getattr(args, "profile", DEFAULT_PROFILE_NAME))
    profile_width, profile_height, profile_frames, profile_fps = PROFILE_PRESETS[
        profile_name
    ]
    return (
        int(args.width if args.width is not None else profile_width),
        int(args.height if args.height is not None else profile_height),
        int(args.num_frames if args.num_frames is not None else profile_frames),
        int(args.fps if args.fps is not None else profile_fps),
    )


def _validate_artifact_paths(paths: ArtifactPaths) -> ArtifactPaths:
    missing: list[str] = []
    if not paths.checkpoint_path.is_file():
        missing.append(str(paths.checkpoint_path))
    if not paths.spatial_upsampler_path.is_file():
        missing.append(str(paths.spatial_upsampler_path))
    if not paths.text_encoder_path.is_dir():
        missing.append(str(paths.text_encoder_path))
    if missing:
        raise FileNotFoundError(
            "LTX debug smoke artifact paths are missing: " + ", ".join(missing)
        )
    return paths


def _cli_args_from_config(config: SmokeConfig) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/ltx_debug_smoke.py",
        "--checkpoint-path",
        str(config.artifact_paths.checkpoint_path),
        "--spatial-upsampler-path",
        str(config.artifact_paths.spatial_upsampler_path),
        "--text-encoder-path",
        str(config.artifact_paths.text_encoder_path),
        "--prompt",
        config.prompt,
        "--profile",
        config.profile_name,
        "--width",
        str(config.width),
        "--height",
        str(config.height),
        "--num-frames",
        str(config.num_frames),
        "--fps",
        str(config.fps),
        "--seed",
        str(config.seed),
        "--output-root",
        str(config.output_root),
        "--run-name",
        config.run_name,
        "--stage-debug" if config.stage_debug else "--no-stage-debug",
        "--trace" if config.trace else "--no-trace",
        "--trace-sync" if config.trace_sync else "--no-trace-sync",
        "--clean-lifecycle" if config.clean_lifecycle else "--no-clean-lifecycle",
        "--video-only" if config.video_only else "--no-video-only",
        "--backend-progress" if config.backend_progress else "--no-backend-progress",
        "--heartbeat-seconds",
        str(config.heartbeat_seconds),
        "--max-active-gb",
        str(config.max_active_gb),
        "--max-peak-gb",
        str(config.max_peak_gb),
        "--max-seconds",
        str(config.max_seconds),
        "--memory-poll-seconds",
        str(config.memory_poll_seconds),
    ]
    if config.negative_prompt is not None:
        command.extend(["--negative-prompt", config.negative_prompt])
    return command


def _safe_close(resource: object, errors: list[str] | None = None) -> None:
    try:
        close = getattr(resource, "close")
        close()
    except Exception as exc:
        if errors is not None:
            errors.append(f"{type(exc).__name__}: {exc}")


def _first_frame(frames: np.ndarray) -> np.ndarray:
    frame = np.asarray(frames[0])
    if frame.dtype != np.uint8:
        return frame.astype(np.uint8)
    return frame


def _review_frame(frames: np.ndarray, index: int) -> np.ndarray:
    frame = np.asarray(frames[index])
    if frame.dtype != np.uint8:
        return frame.astype(np.uint8)
    return frame


def _write_review_frames(frames: np.ndarray, bundle: RunBundlePaths) -> None:
    total_frames = int(len(frames))
    mid_index = total_frames // 2
    last_index = total_frames - 1
    Image.fromarray(_review_frame(frames, 0)).save(bundle.frame_path)
    Image.fromarray(_review_frame(frames, mid_index)).save(bundle.frame_mid_path)
    Image.fromarray(_review_frame(frames, last_index)).save(bundle.frame_last_path)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def _bytes_from_gb(value: float) -> int:
    return int(value * (1024**3))


def _bytes_to_gb(value: int) -> float:
    return value / float(1024**3)


def _slugify(raw: str) -> str:
    pieces = [
        part
        for part in "".join(
            char.lower() if char.isalnum() else "-" for char in raw.strip()
        ).split("-")
        if part
    ]
    if not pieces:
        return DEFAULT_RUN_NAME
    return "-".join(pieces[:8])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _record_abort(
    *,
    manifest: dict[str, object],
    bundle: RunBundlePaths,
    timings_ms: dict[str, float],
    reason: str,
    details: dict[str, object],
    abort_state: dict[str, object],
) -> None:
    if abort_state:
        return
    abort_state["reason"] = reason
    manifest["status"] = "aborted"
    manifest["backend_verified"] = False
    manifest["abort"] = {
        "reason": reason,
        "details": details,
    }
    manifest["timings_ms"] = timings_ms
    _write_json(bundle.manifest_path, manifest)


def _default_abort_process(exit_code: int) -> None:
    os.kill(os.getpid(), signal.SIGTERM)
    os._exit(exit_code)


@contextmanager
def temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())

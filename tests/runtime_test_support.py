from __future__ import annotations

import base64
import hashlib
import io
import os
import queue
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from httpx import Response
from mlxr.core.runtime import RuntimeHome
from mlxr.core.schemas import (
    ArtifactConversionResult,
    InputHandleRecord,
    JobRecord,
    SourceRegistrationRecord,
)
from mlxr.core.server.registry import default_runtime_registry
from mlxr.core.server.settings import ServerSettings
from mlxr.core.server.state import RuntimeState
from mlxr.families.ltx.generation import (
    AudioConditioningInput,
    GeneratedVideo,
    LoraInput,
    RetakeOptions,
    VideoReferenceInput,
)
from mlxr.families.ltx.prompt_encoding import PromptEncodingResult
from PIL import Image
from pydantic import BaseModel

os.environ.setdefault(
    "MLX_RUNTIME_HOME", str(Path(tempfile.gettempdir()) / "mlxr-test-import-home")
)

ModelT = TypeVar("ModelT", bound=BaseModel)
LTX_CHECKPOINT_FILENAME = "ltx-2.3-22b-distilled.safetensors"
LTX_DEV_CHECKPOINT_FILENAME = "ltx-2.3-22b-dev.safetensors"
LTX_SPATIAL_UPSAMPLER_FILENAME = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
LTX_DISTILLED_LORA_FILENAME = "ltx-2.3-22b-distilled-lora-384.safetensors"
LTX_TEXT_ENCODER_DIRNAME = "gemma-3-12b-it-qat-q4_0-unquantized"


def make_state(tmp_path: Path, settings: ServerSettings | None = None) -> RuntimeState:
    return RuntimeState(
        registry=default_runtime_registry(),
        runtime_home=RuntimeHome(root=tmp_path / "runtime-home"),
        settings=settings or ServerSettings(),
    )


def make_local_bundle(
    root: Path,
    *,
    directory_name: str = "ltx-bundle",
    checkpoint_filename: str = LTX_CHECKPOINT_FILENAME,
    include_spatial_upsampler: bool = True,
    include_distilled_lora: bool = False,
) -> Path:
    source_dir = root / directory_name
    source_dir.mkdir()
    (source_dir / checkpoint_filename).write_text("bundle", encoding="utf-8")
    if include_spatial_upsampler:
        (source_dir / LTX_SPATIAL_UPSAMPLER_FILENAME).write_text(
            "upsampler", encoding="utf-8"
        )
    if include_distilled_lora:
        (source_dir / LTX_DISTILLED_LORA_FILENAME).write_text(
            "distilled-lora", encoding="utf-8"
        )
    text_encoder_dir = source_dir / LTX_TEXT_ENCODER_DIRNAME
    _write_fake_text_encoder(text_encoder_dir)
    return source_dir


def make_split_local_ltx_sources(
    root: Path,
    *,
    checkpoint_filename: str = LTX_CHECKPOINT_FILENAME,
    include_spatial_upsampler: bool = True,
    include_distilled_lora: bool = False,
) -> dict[str, Path]:
    checkpoint_dir = root / "ltx-checkpoint-source"
    checkpoint_dir.mkdir()
    (checkpoint_dir / checkpoint_filename).write_text("bundle", encoding="utf-8")

    text_encoder_dir = root / "ltx-text-encoder-source"
    text_encoder_dir.mkdir()
    _write_fake_text_encoder(text_encoder_dir)

    sources = {
        "checkpoint": checkpoint_dir,
        "text_encoder": text_encoder_dir,
    }
    if include_spatial_upsampler:
        upsampler_dir = root / "ltx-upsampler-source"
        upsampler_dir.mkdir()
        (upsampler_dir / LTX_SPATIAL_UPSAMPLER_FILENAME).write_text(
            "upsampler", encoding="utf-8"
        )
        sources["spatial_upsampler"] = upsampler_dir
    if include_distilled_lora:
        distilled_lora_dir = root / "ltx-distilled-lora-source"
        distilled_lora_dir.mkdir()
        (distilled_lora_dir / LTX_DISTILLED_LORA_FILENAME).write_text(
            "distilled-lora", encoding="utf-8"
        )
        sources["distilled_lora"] = distilled_lora_dir
    return sources


def _write_fake_text_encoder(text_encoder_dir: Path) -> None:
    text_encoder_dir.mkdir(exist_ok=True)
    (text_encoder_dir / "config.json").write_text("{}", encoding="utf-8")
    (text_encoder_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (text_encoder_dir / "model-00001-of-00001.safetensors").write_text(
        "weights", encoding="utf-8"
    )


def response_model(response: Response, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(response.json())


def register_local_ltx_model(
    client: TestClient,
    source_dir: Path,
    *,
    headers: dict[str, str] | None = None,
    model_id: str = "ltx-2.3-fast-local",
) -> str:
    register_response = client.post(
        "/v1/sources/register",
        json={
            "provider": "local",
            "locator": {"path": str(source_dir)},
            "family_hint": "ltx",
        },
        headers=headers,
    )
    assert register_response.status_code == 200, register_response.text
    source_record = response_model(register_response, SourceRegistrationRecord)

    convert_response = client.post(
        "/v1/artifacts/convert",
        json={"source_id": source_record.source_id, "model_id": model_id},
        headers=headers,
    )
    assert convert_response.status_code == 200, convert_response.text
    response_model(convert_response, ArtifactConversionResult)
    return source_record.source_id


def import_input_handle(
    client: TestClient,
    payload: bytes,
    *,
    headers: dict[str, str] | None = None,
    media_type: str = "image/png",
    filename: str = "conditioning.png",
) -> str:
    response = client.post(
        "/v1/inputs/import",
        json={
            "content_base64": base64.b64encode(payload).decode("ascii"),
            "media_type": media_type,
            "filename": filename,
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response_model(response, InputHandleRecord).handle_id


def make_png_bytes(
    *,
    width: int = 64,
    height: int = 64,
    color: tuple[int, int, int] = (36, 108, 196),
) -> bytes:
    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def wait_for_job_terminal_state(
    client: TestClient,
    job_id: str,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        record = response_model(response, JobRecord)
        if record.state.value in {"completed", "failed", "cancelled"}:
            return record.model_dump(mode="json")
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for job '{job_id}' to finish")


def http_headers(
    *,
    token: str = "secret-token",
    origin: str | None = None,
    referer: str | None = None,
    fetch_site: str | None = None,
) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if origin is not None:
        headers["Origin"] = origin
    if referer is not None:
        headers["Referer"] = referer
    if fetch_site is not None:
        headers["Sec-Fetch-Site"] = fetch_site
    return headers


class FakePromptEncoder:
    def __init__(
        self,
        *,
        token_count: int = 8,
        sequence_length: int = 1024,
        include_audio_context: bool = True,
    ) -> None:
        self.token_count = token_count
        self.sequence_length = sequence_length
        self.include_audio_context = include_audio_context
        self.calls: list[tuple[str, int, bool]] = []
        self.closed = False

    def encode(
        self,
        prompt: str,
        *,
        max_length: int = 1024,
        return_audio_context: bool = True,
        negative_prompt: str | None = None,
    ) -> PromptEncodingResult:
        self.calls.append((prompt, max_length, return_audio_context))
        audio_context = (
            "audio-context"
            if self.include_audio_context and return_audio_context
            else None
        )
        normalized_negative_prompt = negative_prompt.strip() if negative_prompt else ""
        return PromptEncodingResult(
            video_context="video-context",
            audio_context=audio_context,
            attention_mask="attention-mask",
            prompt_text=prompt,
            token_count=self.token_count,
            sequence_length=self.sequence_length,
            video_context_shape=(1, self.sequence_length, 3840),
            attention_mask_shape=(1, self.sequence_length),
            audio_context_shape=(
                (1, self.sequence_length, 2048) if audio_context is not None else None
            ),
            context_representation="post_connector",
            caption_proj_before_connector=True,
            rope_type="split",
            double_precision_rope=True,
            connector_apply_gated_attention=True,
            transformer_context_dim=3840,
            transformer_apply_gated_attention=True,
            transformer_cross_attention_adaln=True,
            config_source="fake://prompt-config",
            negative_prompt_text=normalized_negative_prompt or None,
            negative_video_context=(
                "negative-video-context" if normalized_negative_prompt else None
            ),
            negative_audio_context=(
                "negative-audio-context"
                if normalized_negative_prompt and audio_context is not None
                else None
            ),
            negative_video_context_shape=(
                (1, self.sequence_length, 3840) if normalized_negative_prompt else None
            ),
            negative_audio_context_shape=(
                (1, self.sequence_length, 2048)
                if normalized_negative_prompt and audio_context is not None
                else None
            ),
        )

    def close(self) -> None:
        self.closed = True


class FakeVideoGenerator:
    def __init__(
        self,
        *,
        backend: str = "ltx_test_distilled_generator",
        include_audio: bool = False,
    ) -> None:
        self.backend = backend
        self.include_audio = include_audio
        self.guidance_mode = "positive_only"
        self.control_variant: str | None = None
        self.conditioning_attention_strength: float | None = None
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        task: str = "video.generate",
        conditioning_inputs: tuple[object, ...],
        video_inputs: tuple[VideoReferenceInput, ...] = (),
        lora_inputs: tuple[LoraInput, ...] = (),
        audio_conditioning: AudioConditioningInput | None = None,
        retake_options: RetakeOptions | None = None,
        control_variant: str | None = None,
        conditioning_attention_strength: float | None = None,
        pipeline_variant: str = "distilled_two_stage",
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int | None = None,
    ) -> GeneratedVideo:
        self.calls.append(
            {
                "prompt": prompt_context.prompt_text,
                "task": task,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "seed": seed,
                "conditioning_count": len(conditioning_inputs),
                "video_input_count": len(video_inputs),
                "lora_input_count": len(lora_inputs),
                "conditioning_frame_indices": [
                    getattr(item, "frame_index", None) for item in conditioning_inputs
                ],
                "conditioning_strengths": [
                    getattr(item, "strength", None) for item in conditioning_inputs
                ],
                "lora_strengths": [item.strength for item in lora_inputs],
                "retake_options": (
                    None
                    if retake_options is None
                    else {
                        "start_time_seconds": retake_options.start_time_seconds,
                        "end_time_seconds": retake_options.end_time_seconds,
                        "regenerate_video": retake_options.regenerate_video,
                        "regenerate_audio": retake_options.regenerate_audio,
                    }
                ),
                "audio_conditioned": audio_conditioning is not None,
                "pipeline_variant": pipeline_variant,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "negative_prompt_present": prompt_context.negative_prompt_text
                is not None,
                "guidance_mode": self.guidance_mode,
                "control_variant": (
                    control_variant
                    if control_variant is not None
                    else self.control_variant
                ),
                "conditioning_attention_strength": (
                    conditioning_attention_strength
                    if conditioning_attention_strength is not None
                    else self.conditioning_attention_strength
                ),
            }
        )
        effective_seed = 0 if seed is None else seed
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        width_ramp = np.linspace(0, 255, width, dtype=np.uint8)
        height_ramp = np.linspace(0, 255, height, dtype=np.uint8)
        for index in range(num_frames):
            frames[index, :, :, 0] = (width_ramp + index * 17) % 255
            frames[index, :, :, 1] = height_ramp[:, None]
            frames[index, :, :, 2] = (effective_seed + index * 13) % 255
        prompt_signature = hashlib.sha256(
            prompt_context.prompt_text.encode("utf-8")
        ).hexdigest()[:12]
        audio_waveform: np.ndarray | None = None
        audio_sample_rate: int | None = None
        if self.include_audio:
            sample_rate = 24000
            sample_count = max(
                sample_rate // 2, int(sample_rate * num_frames / max(fps, 1))
            )
            time_axis = np.linspace(
                0.0, 1.0, sample_count, endpoint=False, dtype=np.float32
            )
            left = np.sin(2.0 * np.pi * 220.0 * time_axis).astype(np.float32)
            right = np.sin(2.0 * np.pi * 330.0 * time_axis).astype(np.float32)
            audio_waveform = np.stack((left, right), axis=1).astype(np.float32)
            audio_sample_rate = sample_rate
        return GeneratedVideo(
            frames=frames,
            fps=fps,
            seed=effective_seed,
            backend=self.backend,
            conditioning_count=len(conditioning_inputs),
            prompt_signature=prompt_signature,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            metadata={
                "pipeline_kind": pipeline_variant,
                "stage1_duration_ms": 12.5,
                "upsample_duration_ms": 3.25,
                "stage2_duration_ms": 9.75,
                "decode_duration_ms": 4.0,
                "audio_decode_duration_ms": 2.0 if self.include_audio else 0.0,
                "tiling_mode": "none",
                "output_width": width,
                "output_height": height,
                "output_frames": num_frames,
                "internal_width": width,
                "internal_height": height,
                "audio_present": self.include_audio,
                "audio_sample_rate": audio_sample_rate,
                "audio_channels": 2 if self.include_audio else 0,
            },
        )

    def close(self) -> None:
        self.closed = True


class ThreadMessageQueue(queue.Queue[dict[str, object]]):
    def close(self) -> None:
        return None


class ThreadManagedProcess:
    def __init__(
        self,
        *,
        target: object,
        kwargs: dict[str, object],
    ) -> None:
        if not callable(target):
            raise TypeError("ThreadManagedProcess target must be callable")
        self._target = target
        self._kwargs = kwargs
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self._target(**self._kwargs)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def terminate(self) -> None:
        return None

    @property
    def exitcode(self) -> int | None:
        if self._thread.is_alive():
            return None
        return 0


class ThreadProcessContext:
    def Queue(self) -> ThreadMessageQueue:
        return ThreadMessageQueue()

    def Process(
        self,
        target: object,
        kwargs: dict[str, object],
    ) -> ThreadManagedProcess:
        return ThreadManagedProcess(target=target, kwargs=kwargs)


class SilentExitProcessContext:
    def Queue(self) -> ThreadMessageQueue:
        return ThreadMessageQueue()

    def Process(
        self,
        target: object,
        kwargs: dict[str, object],
    ) -> ThreadManagedProcess:
        del target, kwargs
        return ThreadManagedProcess(target=lambda: None, kwargs={})


@contextmanager
def patched_ltx_prompt_encoder(
    *,
    token_count: int = 8,
    sequence_length: int = 1024,
    include_audio_context: bool = True,
) -> Iterator[list[FakePromptEncoder]]:
    from mlxr.families.ltx import adapter as adapter_module

    instances: list[FakePromptEncoder] = []

    def factory(
        checkpoint_path: Path,
        text_encoder_path: Path,
    ) -> FakePromptEncoder:
        del checkpoint_path, text_encoder_path
        encoder = FakePromptEncoder(
            token_count=token_count,
            sequence_length=sequence_length,
            include_audio_context=include_audio_context,
        )
        instances.append(encoder)
        return encoder

    with patch.object(adapter_module, "create_prompt_encoder", side_effect=factory):
        yield instances


@contextmanager
def patched_ltx_video_generator(
    *,
    backend: str = "ltx_test_distilled_generator",
    include_audio: bool = False,
) -> Iterator[list[FakeVideoGenerator]]:
    from mlxr.families.ltx import adapter as adapter_module

    instances: list[FakeVideoGenerator] = []

    def factory(
        checkpoint_path: Path,
        spatial_upsampler_path: Path | None,
        distilled_lora_path: Path | None,
    ) -> FakeVideoGenerator:
        del checkpoint_path, spatial_upsampler_path, distilled_lora_path
        generator = FakeVideoGenerator(
            backend=backend,
            include_audio=include_audio,
        )
        instances.append(generator)
        return generator

    with patch.object(adapter_module, "create_video_generator", side_effect=factory):
        yield instances


@contextmanager
def patched_inline_job_process_context() -> Iterator[None]:
    from mlxr.core.server import jobs as jobs_module

    with patch.object(
        jobs_module,
        "_job_process_context",
        return_value=ThreadProcessContext(),
    ):
        yield


@contextmanager
def patched_silent_exit_job_process_context() -> Iterator[None]:
    from mlxr.core.server import jobs as jobs_module

    with patch.object(
        jobs_module,
        "_job_process_context",
        return_value=SilentExitProcessContext(),
    ):
        yield

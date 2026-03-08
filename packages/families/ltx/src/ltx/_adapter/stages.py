from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_runtime_core import ExecutionStage, LoadedModelHandle, StageResult
from mlx_runtime_schemas import ArtifactHandle

from ..generation import AudioConditioningInput, ConditioningInput, VideoGenerator
from ..prompt_encoding import PromptEncoder
from .state import LoadedLTXRuntimeState

if TYPE_CHECKING:
    from ..adapter import LTXFamilyAdapter


def run_stage(
    self: LTXFamilyAdapter, loaded: LoadedModelHandle, stage: ExecutionStage
) -> StageResult:
    delay = stage.params.get("simulate_delay_seconds", 0.05)
    if isinstance(delay, (float, int)) and delay > 0:
        time.sleep(float(delay))

    runtime_state = self._runtime_state(loaded)
    if stage.stage_id == "prompt_encode":
        if runtime_state is None:
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "placeholder",
                    "tokens_estimate": len(str(stage.inputs.get("prompt", "")).split()),
                }
            )
        prompt = self._prompt_text(stage.inputs)
        self._reject_unsupported_negative_prompt(stage.inputs)
        prompt_encoder = self._prompt_encoder(runtime_state)
        prompt_context = prompt_encoder.encode(
            prompt,
            max_length=1024,
            return_audio_context=True,
        )
        runtime_state.prompt_context = prompt_context
        return StageResult(
            metrics={
                "stage": stage.stage_id,
                "status": "encoded",
                "prompt_chars": len(prompt),
                "token_count": prompt_context.token_count,
                "sequence_length": prompt_context.sequence_length,
                "video_context_shape": list(prompt_context.video_context_shape),
                "audio_context_shape": (
                    list(prompt_context.audio_context_shape)
                    if prompt_context.audio_context_shape is not None
                    else None
                ),
                "attention_mask_shape": list(prompt_context.attention_mask_shape),
                "audio_context_available": prompt_context.audio_context is not None,
                "context_representation": prompt_context.context_representation,
                "caption_proj_before_connector": (
                    prompt_context.caption_proj_before_connector
                ),
                "rope_type": prompt_context.rope_type,
                "double_precision_rope": prompt_context.double_precision_rope,
                "connector_apply_gated_attention": (
                    prompt_context.connector_apply_gated_attention
                ),
                "transformer_context_dim": prompt_context.transformer_context_dim,
                "config_source": prompt_context.config_source,
            }
        )
    if stage.stage_id == "condition_inputs":
        prepared_inputs = self._prepared_conditioning_inputs(stage)
        audio_conditioning = self._prepared_audio_conditioning_input(stage)
        if runtime_state is None:
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "placeholder",
                    "conditioning_count": len(prepared_inputs),
                    "audio_conditioned": audio_conditioning is not None,
                }
            )
        runtime_state.conditioning_inputs = prepared_inputs
        runtime_state.audio_conditioning = audio_conditioning
        task = self._stage_task(stage)
        if task == "video.condition.image" and not prepared_inputs:
            raise ValueError(
                "video.condition.image requires at least one resolved conditioning image"
            )
        if task == "video.condition.audio" and audio_conditioning is None:
            raise ValueError(
                "video.condition.audio requires one resolved conditioning audio input"
            )
        return StageResult(
            metrics={
                "stage": stage.stage_id,
                "status": (
                    "prepared"
                    if prepared_inputs or audio_conditioning is not None
                    else "skipped"
                ),
                "conditioning_count": len(prepared_inputs),
                "frame_indices": [item.frame_index for item in prepared_inputs],
                "audio_conditioned": audio_conditioning is not None,
                "audio_handle_id": (
                    audio_conditioning.handle_id
                    if audio_conditioning is not None
                    else None
                ),
            }
        )
    if stage.stage_id == "generate":
        if runtime_state is None:
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "placeholder",
                    "frames_requested": stage.params.get(
                        "num_frames", stage.inputs.get("num_frames")
                    ),
                }
            )
        if runtime_state.prompt_context is None:
            raise ValueError(
                "LTX generate stage requires prompt_encode to run successfully first"
            )
        if runtime_state.prompt_encoder is not None:
            runtime_state.prompt_encoder.close()
            runtime_state.prompt_encoder = None
            mx.clear_cache()
        generator = self._video_generator(runtime_state)
        generated_video = generator.generate(
            prompt_context=runtime_state.prompt_context,
            conditioning_inputs=runtime_state.conditioning_inputs,
            audio_conditioning=runtime_state.audio_conditioning,
            width=self._stage_dimension(stage.params.get("width"), name="width"),
            height=self._stage_dimension(stage.params.get("height"), name="height"),
            num_frames=self._num_frames(stage),
            fps=self._fps(stage),
            seed=self._seed(stage),
        )
        runtime_state.generated_video = generated_video
        metrics = {
            "stage": stage.stage_id,
            "status": "generated",
            "frames_generated": int(generated_video.frames.shape[0]),
            "width": int(generated_video.frames.shape[2]),
            "height": int(generated_video.frames.shape[1]),
            "fps": generated_video.fps,
            "seed": generated_video.seed,
            "backend": generated_video.backend,
            "conditioning_count": generated_video.conditioning_count,
            "prompt_signature": generated_video.prompt_signature,
        }
        for key, value in generated_video.metadata.items():
            metrics[key] = value
        return StageResult(metrics=metrics)
    if stage.stage_id == "encode_output":
        artifact_id = self._require_str(stage.params.get("artifact_id"), "artifact_id")
        artifact_format = self._require_str(
            stage.params.get("artifact_format"), "artifact_format"
        )
        output_dir = self._require_str(stage.params.get("output_dir"), "output_dir")
        storage_key = self._require_str(stage.params.get("storage_key"), "storage_key")
        filename = f"{artifact_id}.{artifact_format}"
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if runtime_state is None or runtime_state.generated_video is None:
            raise ValueError(
                "LTX encode_output requires generate to run successfully first"
            )
        adapter_module = import_module("ltx.adapter")
        if artifact_format == "mp4":
            adapter_module.encode_mp4_video(runtime_state.generated_video, output_path)
        elif artifact_format == "wav":
            adapter_module.encode_wav_audio(runtime_state.generated_video, output_path)
        else:
            raise ValueError(
                "LTX encode_output only supports runtime-managed mp4 or wav "
                f"artifacts, got '{artifact_format}'"
            )
        generated_video = runtime_state.generated_video
        runtime_state.generated_video = None
        audio_channels = 0
        if generated_video.audio_waveform is not None:
            audio_channels = (
                int(generated_video.audio_waveform.shape[1])
                if generated_video.audio_waveform.ndim == 2
                else 1
            )
        return StageResult(
            artifacts=[
                ArtifactHandle(
                    artifact_id=artifact_id,
                    artifact_format=artifact_format,
                    metadata={
                        "filename": filename,
                        "media_type": self._media_type_for_format(artifact_format),
                        "size_bytes": output_path.stat().st_size,
                        "storage_key": storage_key,
                        "audio_present": generated_video.audio_waveform is not None,
                        "audio_sample_rate": generated_video.audio_sample_rate,
                        "audio_channels": audio_channels,
                    },
                )
            ],
            metrics={
                "stage": stage.stage_id,
                "status": "encoded",
                "output_bytes": output_path.stat().st_size,
                "frames_encoded": int(generated_video.frames.shape[0]),
                "fps": generated_video.fps,
                "backend": generated_video.backend,
                "pipeline_kind": generated_video.metadata.get("pipeline_kind"),
                "artifact_format": artifact_format,
                "audio_present": generated_video.audio_waveform is not None,
                "audio_sample_rate": generated_video.audio_sample_rate,
                "audio_channels": audio_channels,
            },
        )

    raise ValueError(
        f"LTX stage execution is not implemented for unsupported stage '{stage.stage_id}'"
    )


def unload(self: LTXFamilyAdapter, loaded: LoadedModelHandle) -> None:
    runtime_state = self._runtime_state(loaded)
    if runtime_state is not None:
        if runtime_state.prompt_encoder is not None:
            runtime_state.prompt_encoder.close()
            runtime_state.prompt_encoder = None
        if runtime_state.video_generator is not None:
            runtime_state.video_generator.close()
            runtime_state.video_generator = None
        runtime_state.prompt_context = None
        runtime_state.conditioning_inputs = ()
        runtime_state.audio_conditioning = None
        runtime_state.generated_video = None
        loaded.metadata.pop(self._runtime_state_key, None)
    return None


def _runtime_state(
    self: LTXFamilyAdapter, loaded: LoadedModelHandle
) -> LoadedLTXRuntimeState | None:
    state = loaded.metadata.get(self._runtime_state_key)
    if isinstance(state, LoadedLTXRuntimeState):
        return state
    return None


def _prompt_encoder(
    self: LTXFamilyAdapter, runtime_state: LoadedLTXRuntimeState
) -> PromptEncoder:
    if runtime_state.prompt_encoder is None:
        adapter_module = import_module("ltx.adapter")
        runtime_state.prompt_encoder = adapter_module.create_prompt_encoder(
            checkpoint_path=runtime_state.component_paths["checkpoint"],
            text_encoder_path=runtime_state.component_paths["text_encoder"],
        )
    return runtime_state.prompt_encoder


def _video_generator(
    self: LTXFamilyAdapter, runtime_state: LoadedLTXRuntimeState
) -> VideoGenerator:
    if runtime_state.video_generator is None:
        adapter_module = import_module("ltx.adapter")
        runtime_state.video_generator = adapter_module.create_video_generator(
            checkpoint_path=runtime_state.component_paths["checkpoint"],
            spatial_upsampler_path=runtime_state.component_paths["spatial_upsampler"],
        )
    return runtime_state.video_generator


def _prompt_text(self: LTXFamilyAdapter, inputs: dict[str, object]) -> str:
    prompt = inputs.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("LTX prompt_encode expects inputs.prompt to be a string")
    return prompt


def _prepared_conditioning_inputs(
    self: LTXFamilyAdapter, stage: ExecutionStage
) -> tuple[ConditioningInput, ...]:
    resolved_inputs = stage.params.get("resolved_inputs")
    if resolved_inputs is None:
        return ()
    if not isinstance(resolved_inputs, dict):
        raise ValueError("LTX stage params.resolved_inputs must be an object")
    raw_images = resolved_inputs.get("images", [])
    if not isinstance(raw_images, list):
        raise ValueError("LTX resolved conditioning inputs must be a list")

    num_frames = self._num_frames(stage)
    prepared: list[ConditioningInput] = []
    for entry in raw_images:
        if not isinstance(entry, dict):
            raise ValueError("LTX resolved conditioning image entries must be objects")
        handle_id = self._require_str(entry.get("input_handle"), "input_handle")
        payload_path = Path(
            self._require_str(entry.get("payload_path"), "payload_path")
        )
        if not payload_path.is_file():
            raise ValueError(
                f"LTX resolved conditioning payload '{payload_path}' does not exist"
            )
        frame_index = entry.get("frame_index", 0)
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError(
                "LTX conditioning frame_index must be a non-negative integer"
            )
        if frame_index >= num_frames:
            raise ValueError(
                f"LTX conditioning frame_index {frame_index} is outside num_frames={num_frames}"
            )
        strength = entry.get("strength", 1.0)
        if not isinstance(strength, (int, float)):
            raise ValueError("LTX conditioning strength must be numeric")
        strength_value = float(strength)
        if not 0.0 <= strength_value <= 1.0:
            raise ValueError("LTX conditioning strength must be between 0.0 and 1.0")
        media_type = entry.get("media_type")
        filename = entry.get("filename")
        prepared.append(
            ConditioningInput(
                handle_id=handle_id,
                payload_path=payload_path,
                frame_index=frame_index,
                strength=strength_value,
                media_type=media_type if isinstance(media_type, str) else None,
                filename=filename if isinstance(filename, str) else None,
            )
        )
    return tuple(prepared)


def _prepared_audio_conditioning_input(
    self: LTXFamilyAdapter, stage: ExecutionStage
) -> AudioConditioningInput | None:
    resolved_inputs = stage.params.get("resolved_inputs")
    if resolved_inputs is None:
        return None
    if not isinstance(resolved_inputs, dict):
        raise ValueError("LTX stage params.resolved_inputs must be an object")
    raw_audio = resolved_inputs.get("audio")
    if raw_audio is None:
        return None
    if not isinstance(raw_audio, dict):
        raise ValueError("LTX resolved conditioning audio must be an object")

    handle_id = self._require_str(raw_audio.get("input_handle"), "input_handle")
    payload_path = Path(
        self._require_str(raw_audio.get("payload_path"), "payload_path")
    )
    if not payload_path.is_file():
        raise ValueError(
            f"LTX resolved conditioning audio payload '{payload_path}' does not exist"
        )
    start_time_seconds = raw_audio.get("start_time_seconds", 0.0)
    if (
        not isinstance(start_time_seconds, (int, float))
        or float(start_time_seconds) < 0.0
    ):
        raise ValueError(
            "LTX conditioning audio start_time_seconds must be numeric and >= 0.0"
        )
    raw_max_duration = raw_audio.get("max_duration_seconds")
    max_duration_seconds: float | None = None
    if raw_max_duration is not None:
        if (
            not isinstance(raw_max_duration, (int, float))
            or float(raw_max_duration) <= 0.0
        ):
            raise ValueError(
                "LTX conditioning audio max_duration_seconds must be numeric and > 0.0"
            )
        max_duration_seconds = float(raw_max_duration)
    media_type = raw_audio.get("media_type")
    filename = raw_audio.get("filename")
    return AudioConditioningInput(
        handle_id=handle_id,
        payload_path=payload_path,
        start_time_seconds=float(start_time_seconds),
        max_duration_seconds=max_duration_seconds,
        media_type=media_type if isinstance(media_type, str) else None,
        filename=filename if isinstance(filename, str) else None,
    )


def _stage_dimension(self: LTXFamilyAdapter, value: object, *, name: str) -> int:
    if value is None:
        return 768 if name == "width" else 512
    if not isinstance(value, int) or value < 32:
        raise ValueError(f"LTX {name} must be an integer >= 32")
    return value


def _num_frames(self: LTXFamilyAdapter, stage: ExecutionStage) -> int:
    value = stage.params.get("num_frames", stage.inputs.get("num_frames"))
    if value is None:
        return 121
    if not isinstance(value, int) or value < 1:
        raise ValueError("LTX num_frames must be a positive integer")
    return value


def _fps(self: LTXFamilyAdapter, stage: ExecutionStage) -> int:
    value = stage.params.get("fps", stage.inputs.get("fps"))
    if value is None:
        return 24
    if not isinstance(value, int) or value < 1:
        raise ValueError("LTX fps must be a positive integer")
    return value


def _seed(self: LTXFamilyAdapter, stage: ExecutionStage) -> int | None:
    value = stage.params.get("seed", stage.inputs.get("seed"))
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("LTX seed must be an integer when provided")
    return value


def _stage_task(self: LTXFamilyAdapter, stage: ExecutionStage) -> str:
    value = stage.params.get("task")
    if isinstance(value, str) and value:
        return value
    return "video.generate"


def _reject_unsupported_negative_prompt(
    self: LTXFamilyAdapter, inputs: dict[str, object]
) -> None:
    negative_prompt = inputs.get("negative_prompt")
    if negative_prompt in {None, ""}:
        return
    if not isinstance(negative_prompt, str):
        raise ValueError(
            "LTX prompt_encode expects inputs.negative_prompt to be a string when provided"
        )
    if negative_prompt.strip():
        raise ValueError(
            "LTX fast-path prompt encoding does not support negative_prompt yet"
        )

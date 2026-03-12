from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlxr.core.runtime import ExecutionStage, LoadedModelHandle, StageResult
from mlxr.core.schemas import ArtifactHandle

from ..family_options import (
    DistilledGuidanceMode,
    conditioning_attention_strength_from_extensions,
    control_variant_from_extensions,
    distilled_guidance_mode_from_extensions,
    hq_stage_1_distilled_lora_strength_from_extensions,
    hq_stage_2_distilled_lora_strength_from_extensions,
)
from ..generation import (
    AudioConditioningInput,
    ConditioningInput,
    LoraInput,
    RetakeOptions,
    VideoGenerator,
    VideoReferenceInput,
)
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
        negative_prompt = self._negative_prompt_text(stage.inputs)
        prompt_encoder = self._prompt_encoder(runtime_state)
        prompt_context = prompt_encoder.encode(
            prompt,
            max_length=1024,
            return_audio_context=True,
            negative_prompt=negative_prompt,
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
                "negative_prompt_present": prompt_context.negative_prompt_text
                is not None,
                "negative_video_context_shape": (
                    list(prompt_context.negative_video_context_shape)
                    if prompt_context.negative_video_context_shape is not None
                    else None
                ),
                "negative_audio_context_shape": (
                    list(prompt_context.negative_audio_context_shape)
                    if prompt_context.negative_audio_context_shape is not None
                    else None
                ),
            }
        )
    if stage.stage_id == "condition_inputs":
        prepared_inputs = self._prepared_conditioning_inputs(stage)
        prepared_videos = self._prepared_video_inputs(stage)
        prepared_loras = self._prepared_lora_inputs(stage)
        audio_conditioning = self._prepared_audio_conditioning_input(stage)
        retake_options = self._retake_options(stage)
        if runtime_state is None:
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "placeholder",
                    "conditioning_count": len(prepared_inputs),
                    "video_input_count": len(prepared_videos),
                    "lora_input_count": len(prepared_loras),
                    "audio_conditioned": audio_conditioning is not None,
                    "retake_enabled": retake_options is not None,
                }
            )
        runtime_state.conditioning_inputs = prepared_inputs
        runtime_state.video_inputs = prepared_videos
        runtime_state.lora_inputs = prepared_loras
        runtime_state.audio_conditioning = audio_conditioning
        runtime_state.retake_options = retake_options
        task = self._stage_task(stage)
        if task == "video.condition.image" and not prepared_inputs:
            raise ValueError(
                "video.condition.image requires at least one resolved conditioning image"
            )
        if task == "video.condition.video":
            if len(prepared_videos) != 1:
                raise ValueError(
                    "video.condition.video requires exactly one resolved reference video"
                )
            if len(prepared_loras) != 1:
                raise ValueError(
                    "video.condition.video requires exactly one resolved LoRA input"
                )
        if task == "video.retake":
            if len(prepared_videos) != 1:
                raise ValueError(
                    "video.retake requires exactly one resolved source video"
                )
            if retake_options is None:
                raise ValueError(
                    "video.retake requires retake timing options in params"
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
                    if prepared_inputs
                    or audio_conditioning is not None
                    or prepared_videos
                    or prepared_loras
                    else "skipped"
                ),
                "conditioning_count": len(prepared_inputs),
                "frame_indices": [item.frame_index for item in prepared_inputs],
                "video_input_count": len(prepared_videos),
                "lora_input_count": len(prepared_loras),
                "audio_conditioned": audio_conditioning is not None,
                "audio_handle_id": (
                    audio_conditioning.handle_id
                    if audio_conditioning is not None
                    else None
                ),
                "retake_enabled": retake_options is not None,
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
        _apply_family_stage_options(
            generator,
            distilled_guidance_mode=self._distilled_guidance_mode(stage),
            hq_stage_1_distilled_lora_strength=self._hq_stage_1_distilled_lora_strength(
                stage
            ),
            hq_stage_2_distilled_lora_strength=self._hq_stage_2_distilled_lora_strength(
                stage
            ),
            control_variant=self._control_variant(stage),
            conditioning_attention_strength=self._conditioning_attention_strength(
                stage
            ),
        )
        generated_video = generator.generate(
            prompt_context=runtime_state.prompt_context,
            task=self._stage_task(stage),
            conditioning_inputs=runtime_state.conditioning_inputs,
            video_inputs=runtime_state.video_inputs,
            lora_inputs=runtime_state.lora_inputs,
            audio_conditioning=runtime_state.audio_conditioning,
            retake_options=runtime_state.retake_options,
            control_variant=self._control_variant(stage),
            conditioning_attention_strength=self._conditioning_attention_strength(
                stage
            ),
            pipeline_variant=self._pipeline_variant(stage),
            num_inference_steps=self._num_inference_steps(stage),
            guidance_scale=self._guidance_scale(stage),
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
        adapter_module = import_module("mlxr.families.ltx.adapter")
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
        runtime_state.video_inputs = ()
        runtime_state.lora_inputs = ()
        runtime_state.audio_conditioning = None
        runtime_state.retake_options = None
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
        adapter_module = import_module("mlxr.families.ltx.adapter")
        runtime_state.prompt_encoder = adapter_module.create_prompt_encoder(
            checkpoint_path=runtime_state.component_paths["checkpoint"],
            text_encoder_path=runtime_state.component_paths["text_encoder"],
        )
    return runtime_state.prompt_encoder


def _video_generator(
    self: LTXFamilyAdapter, runtime_state: LoadedLTXRuntimeState
) -> VideoGenerator:
    if runtime_state.video_generator is None:
        adapter_module = import_module("mlxr.families.ltx.adapter")
        spatial_upsampler_path = runtime_state.component_paths.get("spatial_upsampler")
        distilled_lora_path = runtime_state.component_paths.get("distilled_lora")
        runtime_state.video_generator = adapter_module.create_video_generator(
            checkpoint_path=runtime_state.component_paths["checkpoint"],
            spatial_upsampler_path=spatial_upsampler_path,
            distilled_lora_path=distilled_lora_path,
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


def _prepared_video_inputs(
    self: LTXFamilyAdapter, stage: ExecutionStage
) -> tuple[VideoReferenceInput, ...]:
    resolved_inputs = stage.params.get("resolved_inputs")
    if resolved_inputs is None:
        return ()
    if not isinstance(resolved_inputs, dict):
        raise ValueError("LTX stage params.resolved_inputs must be an object")
    raw_videos = resolved_inputs.get("videos", [])
    if not isinstance(raw_videos, list):
        raise ValueError("LTX resolved video inputs must be a list")

    prepared: list[VideoReferenceInput] = []
    for entry in raw_videos:
        if not isinstance(entry, dict):
            raise ValueError("LTX resolved video entries must be objects")
        handle_id = self._require_str(entry.get("input_handle"), "input_handle")
        payload_path = Path(
            self._require_str(entry.get("payload_path"), "payload_path")
        )
        if not payload_path.is_file():
            raise ValueError(
                f"LTX resolved video payload '{payload_path}' does not exist"
            )
        strength = entry.get("strength", 1.0)
        if not isinstance(strength, (int, float)):
            raise ValueError("LTX video strength must be numeric")
        strength_value = float(strength)
        if not 0.0 <= strength_value <= 1.0:
            raise ValueError("LTX video strength must be between 0.0 and 1.0")
        media_type = entry.get("media_type")
        filename = entry.get("filename")
        prepared.append(
            VideoReferenceInput(
                handle_id=handle_id,
                payload_path=payload_path,
                strength=strength_value,
                media_type=media_type if isinstance(media_type, str) else None,
                filename=filename if isinstance(filename, str) else None,
            )
        )
    return tuple(prepared)


def _prepared_lora_inputs(
    self: LTXFamilyAdapter, stage: ExecutionStage
) -> tuple[LoraInput, ...]:
    resolved_inputs = stage.params.get("resolved_inputs")
    if resolved_inputs is None:
        return ()
    if not isinstance(resolved_inputs, dict):
        raise ValueError("LTX stage params.resolved_inputs must be an object")
    raw_loras = resolved_inputs.get("loras", [])
    if not isinstance(raw_loras, list):
        raise ValueError("LTX resolved LoRA inputs must be a list")

    prepared: list[LoraInput] = []
    for entry in raw_loras:
        if not isinstance(entry, dict):
            raise ValueError("LTX resolved LoRA entries must be objects")
        handle_id = self._require_str(entry.get("input_handle"), "input_handle")
        payload_path = Path(
            self._require_str(entry.get("payload_path"), "payload_path")
        )
        if not payload_path.is_file():
            raise ValueError(
                f"LTX resolved LoRA payload '{payload_path}' does not exist"
            )
        strength = entry.get("strength", 1.0)
        if not isinstance(strength, (int, float)):
            raise ValueError("LTX LoRA strength must be numeric")
        strength_value = float(strength)
        if strength_value <= 0.0:
            raise ValueError("LTX LoRA strength must be greater than 0.0")
        media_type = entry.get("media_type")
        filename = entry.get("filename")
        prepared.append(
            LoraInput(
                handle_id=handle_id,
                payload_path=payload_path,
                strength=strength_value,
                media_type=media_type if isinstance(media_type, str) else None,
                filename=filename if isinstance(filename, str) else None,
            )
        )
    return tuple(prepared)


def _retake_options(
    self: LTXFamilyAdapter, stage: ExecutionStage
) -> RetakeOptions | None:
    if self._stage_task(stage) != "video.retake":
        return None
    start_time_seconds = stage.params.get("window_start_seconds")
    end_time_seconds = stage.params.get("window_end_seconds")
    if not isinstance(start_time_seconds, (int, float)):
        raise ValueError("LTX retake requires numeric params.window_start_seconds")
    if not isinstance(end_time_seconds, (int, float)):
        raise ValueError("LTX retake requires numeric params.window_end_seconds")
    start_value = float(start_time_seconds)
    end_value = float(end_time_seconds)
    if start_value < 0.0:
        raise ValueError("LTX retake window_start_seconds must be >= 0.0")
    if end_value <= start_value:
        raise ValueError(
            "LTX retake window_end_seconds must be greater than window_start_seconds"
        )
    regenerate_video = stage.params.get("regenerate_video", True)
    regenerate_audio = stage.params.get("regenerate_audio", True)
    if not isinstance(regenerate_video, bool):
        raise ValueError("LTX retake regenerate_video must be boolean when provided")
    if not isinstance(regenerate_audio, bool):
        raise ValueError("LTX retake regenerate_audio must be boolean when provided")
    return RetakeOptions(
        start_time_seconds=start_value,
        end_time_seconds=end_value,
        regenerate_video=regenerate_video,
        regenerate_audio=regenerate_audio,
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


def _num_inference_steps(self: LTXFamilyAdapter, stage: ExecutionStage) -> int | None:
    value = stage.params.get("num_inference_steps")
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError("LTX num_inference_steps must be a positive integer")
    return value


def _guidance_scale(self: LTXFamilyAdapter, stage: ExecutionStage) -> float | None:
    value = stage.params.get("guidance_scale")
    if value is None:
        return None
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ValueError("LTX guidance_scale must be numeric and >= 0.0")
    return float(value)


def _distilled_guidance_mode(
    self: LTXFamilyAdapter, stage: ExecutionStage
) -> DistilledGuidanceMode:
    return distilled_guidance_mode_from_extensions(
        stage.params.get("family_extensions")
    )


def _hq_stage_1_distilled_lora_strength(
    self: LTXFamilyAdapter, stage: ExecutionStage
) -> float:
    return hq_stage_1_distilled_lora_strength_from_extensions(
        stage.params.get("family_extensions")
    )


def _hq_stage_2_distilled_lora_strength(
    self: LTXFamilyAdapter, stage: ExecutionStage
) -> float:
    return hq_stage_2_distilled_lora_strength_from_extensions(
        stage.params.get("family_extensions")
    )


def _control_variant(self: LTXFamilyAdapter, stage: ExecutionStage) -> str | None:
    return control_variant_from_extensions(stage.params.get("family_extensions"))


def _conditioning_attention_strength(
    self: LTXFamilyAdapter, stage: ExecutionStage
) -> float:
    value = conditioning_attention_strength_from_extensions(
        stage.params.get("family_extensions")
    )
    return 1.0 if value is None else value


def _stage_task(self: LTXFamilyAdapter, stage: ExecutionStage) -> str:
    value = stage.params.get("task")
    if isinstance(value, str) and value:
        return value
    return "video.generate"


def _pipeline_variant(self: LTXFamilyAdapter, stage: ExecutionStage) -> str:
    family_extensions = stage.params.get("family_extensions")
    if isinstance(family_extensions, dict):
        raw_variant = family_extensions.get("workflow_variant")
        if isinstance(raw_variant, str) and raw_variant:
            return raw_variant
    return "distilled_two_stage"


def _negative_prompt_text(
    self: LTXFamilyAdapter, inputs: dict[str, object]
) -> str | None:
    negative_prompt = inputs.get("negative_prompt")
    if negative_prompt in {None, ""}:
        return None
    if not isinstance(negative_prompt, str):
        raise ValueError(
            "LTX prompt_encode expects inputs.negative_prompt to be a string when provided"
        )
    normalized = negative_prompt.strip()
    return normalized or None


def _apply_family_stage_options(
    generator: VideoGenerator,
    *,
    distilled_guidance_mode: DistilledGuidanceMode,
    hq_stage_1_distilled_lora_strength: float,
    hq_stage_2_distilled_lora_strength: float,
    control_variant: str | None,
    conditioning_attention_strength: float | None,
) -> None:
    if hasattr(generator, "guidance_mode"):
        setattr(generator, "guidance_mode", distilled_guidance_mode)
    if hasattr(generator, "hq_stage_1_distilled_lora_strength"):
        setattr(
            generator,
            "hq_stage_1_distilled_lora_strength",
            hq_stage_1_distilled_lora_strength,
        )
    if hasattr(generator, "hq_stage_2_distilled_lora_strength"):
        setattr(
            generator,
            "hq_stage_2_distilled_lora_strength",
            hq_stage_2_distilled_lora_strength,
        )
    if hasattr(generator, "control_variant"):
        setattr(generator, "control_variant", control_variant)
    if hasattr(generator, "conditioning_attention_strength"):
        setattr(
            generator,
            "conditioning_attention_strength",
            conditioning_attention_strength,
        )

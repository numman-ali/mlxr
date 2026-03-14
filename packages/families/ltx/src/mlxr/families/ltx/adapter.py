from __future__ import annotations

from ._adapter.artifacts import (
    _artifact_components,
    _artifact_digest,
    _combined_policy,
    _component_paths,
    _directory_digest,
    _directory_files,
    _directory_size,
    _file_digest,
    _looks_like_text_encoder_snapshot,
    _media_type_for_format,
    _prepare_bundle_components,
    _prepare_components,
    _require_str,
    _resolve_checkpoint_file,
    _resolve_required_file,
    _resolve_text_encoder_dir,
    _role_candidates,
    _validate_text_encoder_dir,
)
from ._adapter.capability import (
    capabilities,
    convert,
    fetch_policy_for_conversion,
    inspect_source,
    load,
    normalize_capability,
)
from ._adapter.stages import (
    _conditioning_attention_strength,
    _control_variant,
    _distilled_guidance_mode,
    _fps,
    _guidance_scale,
    _hq_stage_1_distilled_lora_strength,
    _hq_stage_2_distilled_lora_strength,
    _negative_prompt_text,
    _num_frames,
    _num_inference_steps,
    _pipeline_variant,
    _prepared_audio_conditioning_input,
    _prepared_conditioning_inputs,
    _prepared_lora_inputs,
    _prepared_video_inputs,
    _prompt_encoder,
    _prompt_text,
    _retake_options,
    _runtime_state,
    _seed,
    _stage_dimension,
    _stage_task,
    _video_generator,
    run_stage,
    unload,
)
from .generation import create_video_generator, encode_mp4_video, encode_wav_audio
from .prompt_encoding import create_prompt_encoder


class LTXFamilyAdapter:
    family_id = "ltx"
    _checkpoint_filename = "ltx-2.3-22b-distilled.safetensors"
    _dev_checkpoint_filename = "ltx-2.3-22b-dev.safetensors"
    _distilled_lora_filename = "ltx-2.3-22b-distilled-lora-384.safetensors"
    _checkpoint_filenames = (
        _checkpoint_filename,
        _dev_checkpoint_filename,
    )
    _spatial_upsampler_filename = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    _text_encoder_dirname = "gemma-3-12b-it-qat-q4_0-unquantized"
    _required_roles = ("checkpoint", "spatial_upsampler", "text_encoder")
    _optional_roles = ("distilled_lora",)
    _runtime_state_key = "_ltx_runtime_state"

    inspect_source = inspect_source
    fetch_policy_for_conversion = fetch_policy_for_conversion
    convert = convert
    load = load
    capabilities = capabilities
    normalize_capability = normalize_capability
    run_stage = run_stage
    unload = unload

    _runtime_state = _runtime_state
    _prompt_encoder = _prompt_encoder
    _video_generator = _video_generator
    _prompt_text = _prompt_text
    _negative_prompt_text = _negative_prompt_text
    _prepared_conditioning_inputs = _prepared_conditioning_inputs
    _prepared_audio_conditioning_input = _prepared_audio_conditioning_input
    _prepared_video_inputs = _prepared_video_inputs
    _prepared_lora_inputs = _prepared_lora_inputs
    _retake_options = _retake_options
    _stage_dimension = _stage_dimension
    _num_inference_steps = _num_inference_steps
    _guidance_scale = _guidance_scale
    _num_frames = _num_frames
    _fps = _fps
    _distilled_guidance_mode = _distilled_guidance_mode
    _hq_stage_1_distilled_lora_strength = _hq_stage_1_distilled_lora_strength
    _hq_stage_2_distilled_lora_strength = _hq_stage_2_distilled_lora_strength
    _control_variant = _control_variant
    _conditioning_attention_strength = _conditioning_attention_strength
    _seed = _seed
    _stage_task = _stage_task
    _pipeline_variant = _pipeline_variant

    _role_candidates = _role_candidates
    _prepare_components = _prepare_components
    _prepare_bundle_components = _prepare_bundle_components
    _artifact_components = _artifact_components
    _artifact_digest = _artifact_digest
    _combined_policy = _combined_policy
    _component_paths = _component_paths
    _resolve_checkpoint_file = _resolve_checkpoint_file
    _resolve_required_file = _resolve_required_file
    _resolve_text_encoder_dir = _resolve_text_encoder_dir
    _validate_text_encoder_dir = _validate_text_encoder_dir
    _looks_like_text_encoder_snapshot = _looks_like_text_encoder_snapshot
    _directory_files = _directory_files
    _directory_size = _directory_size
    _file_digest = _file_digest
    _directory_digest = _directory_digest
    _media_type_for_format = _media_type_for_format
    _require_str = _require_str


__all__ = [
    "LTXFamilyAdapter",
    "create_prompt_encoder",
    "create_video_generator",
    "encode_mp4_video",
    "encode_wav_audio",
]

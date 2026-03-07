# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


class _PaddedShape:
    output_width: int
    output_height: int
    internal_width: int
    internal_height: int
    crop_top: int
    crop_left: int


@dataclass(frozen=True, slots=True)
class _ReferenceImports:
    model_class: object
    model_config_class: object
    model_type_enum: object
    rope_type_enum: object
    BasicAVTransformerBlock: object
    attention_class: object
    preprocessor_class: object
    multi_preprocessor_class: object
    feed_forward_class: object
    adaln_class: object
    apply_rotary_emb: object
    precompute_freqs_cis: object
    rms_norm: object
    to_denoised: object
    scaled_dot_product_attention: object
    latent_state_class: object
    tiling_config_class: object
    condition_class: object
    stage_1_sigmas: tuple[float, ...]
    stage_2_sigmas: tuple[float, ...]
    apply_conditioning: object
    apply_denoise_mask: object
    create_position_grid: object
    create_audio_position_grid: object
    compute_audio_frames: object
    load_image: object
    load_upsampler: object
    load_vae_decoder: object
    load_vae_encoder: object
    load_audio_decoder: object
    audio_encoder_class: object
    audio_processor_class: object
    audio_norm_type_enum: object
    audio_causality_axis_enum: object
    load_audio_vae_weights: object
    load_vocoder: object
    decode_audio: object
    sanitize_audio_vae_weights: object
    sanitize_vocoder_weights: object
    audio_vocoder_class: object
    prepare_image_for_encoding: object
    upsample_latents: object
    distilled_pipeline_type: object
    audio_latent_channels: int
    audio_mel_bins: int
    audio_sample_rate: int


@dataclass(frozen=True, slots=True)
class _ConditioningPlan:
    stage1: tuple[object, ...]
    stage2: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeModelConfig:
    num_attention_heads: int
    attention_head_dim: int
    in_channels: int
    out_channels: int
    num_layers: int
    cross_attention_dim: int
    audio_enabled: bool
    audio_num_attention_heads: int
    audio_attention_head_dim: int
    audio_in_channels: int
    audio_out_channels: int
    audio_cross_attention_dim: int
    positional_embedding_theta: float
    positional_embedding_max_pos: list[int]
    audio_positional_embedding_max_pos: list[int]
    use_middle_indices_grid: bool
    rope_type: str
    double_precision_rope: bool
    timestep_scale_multiplier: int
    av_ca_timestep_scale_multiplier: int
    norm_eps: float
    apply_gated_attention: bool
    cross_attention_adaln: bool
    caption_proj_before_connector: bool


@dataclass(frozen=True, slots=True)
class _RuntimeVocoderConfig:
    resblock_kernel_sizes: tuple[int, ...]
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    upsample_initial_channel: int
    stereo: bool
    resblock: str
    activation: str
    use_tanh_at_final: bool
    apply_final_activation: bool
    use_bias_at_final: bool
    output_sample_rate: int
    uses_bwe: bool
    bwe_output_sample_rate: int | None


@dataclass(frozen=True, slots=True)
class _RuntimeVAEConfig:
    latent_channels: int
    out_channels: int
    patch_size: int
    decoder_blocks: tuple[tuple[str, object], ...]
    base_channels: int
    spatial_padding_mode: str
    timestep_conditioning: bool
    norm_layer: str
    causal_decoder: bool


@dataclass(frozen=True, slots=True)
class _PatchedTransformerArgs:
    x: mx.array
    context: mx.array
    context_mask: mx.array | None
    timesteps: mx.array
    embedded_timestep: mx.array
    positional_embeddings: tuple[mx.array, mx.array]
    cross_positional_embeddings: tuple[mx.array, mx.array] | None
    cross_scale_shift_timestep: mx.array | None
    cross_gate_timestep: mx.array | None
    enabled: bool
    prompt_timestep: mx.array | None = None


@dataclass(frozen=True, slots=True)
class _PatchedModality:
    latent: mx.array
    sigma: mx.array
    timesteps: mx.array
    positions: mx.array
    context: mx.array
    enabled: bool = True
    context_mask: mx.array | None = None
    positional_embeddings: tuple[mx.array, mx.array] | None = None

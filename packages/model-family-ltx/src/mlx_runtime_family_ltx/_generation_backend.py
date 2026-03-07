from __future__ import annotations

# mypy: ignore-errors
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import numpy.typing as npt
from PIL import Image
from safetensors import safe_open

from ._audio_vocoder import AudioVocoder
from .generation import ConditioningInput, GeneratedVideo, VideoGenerator
from .prompt_encoding import PromptEncodingResult

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REFERENCE_MLX_VIDEO_ROOT = _REPO_ROOT / "references" / "ecosystem" / "mlx-video"


@dataclass(frozen=True, slots=True)
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


class _WrappedCausalConv3d(nn.Module):
    def __init__(
        self,
        *,
        decoder_module: types.ModuleType,
        in_channels: int,
        out_channels: int,
        spatial_padding_mode: object,
    ) -> None:
        super().__init__()
        self.conv = decoder_module.CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array, *, causal: bool = False) -> mx.array:
        return self.conv(x, causal=causal)


class _ConfiguredVideoDecoder(nn.Module):
    def __init__(
        self,
        *,
        decoder_module: types.ModuleType,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        decoder_blocks: tuple[tuple[str, object], ...],
        base_channels: int,
        spatial_padding_mode: object,
        timestep_conditioning: bool,
        causal_decoder: bool,
    ) -> None:
        super().__init__()
        self._decoder_module = decoder_module
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.timestep_conditioning = timestep_conditioning
        self.causal_decoder = causal_decoder
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05
        self.latents_mean = mx.zeros((in_channels,))
        self.latents_std = mx.ones((in_channels,))

        feature_channels = _decoder_initial_feature_channels(
            base_channels=base_channels,
            decoder_blocks=decoder_blocks,
        )
        self.conv_in = _WrappedCausalConv3d(
            decoder_module=decoder_module,
            in_channels=in_channels,
            out_channels=feature_channels,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.up_blocks: dict[int, object] = {}
        for index, (block_name, raw_params) in enumerate(reversed(decoder_blocks)):
            params = (
                raw_params
                if isinstance(raw_params, dict)
                else {"num_layers": raw_params}
            )
            block, feature_channels = self._make_block(
                block_name=block_name,
                block_config=params,
                in_channels=feature_channels,
                spatial_padding_mode=spatial_padding_mode,
            )
            self.up_blocks[index] = block

        final_out_channels = out_channels * patch_size * patch_size
        self.conv_out = _WrappedCausalConv3d(
            decoder_module=decoder_module,
            in_channels=feature_channels,
            out_channels=final_out_channels,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.act = nn.SiLU()
        self._final_feature_channels = feature_channels

        if timestep_conditioning:
            self.timestep_scale_multiplier = mx.array(1000.0)
            self.last_time_embedder = decoder_module.PixArtAlphaTimestepEmbedder(
                embedding_dim=feature_channels * 2
            )
            self.last_scale_shift_table = mx.zeros((2, feature_channels))

    def _make_block(
        self,
        *,
        block_name: str,
        block_config: dict[str, object],
        in_channels: int,
        spatial_padding_mode: object,
    ) -> tuple[object, int]:
        decoder_module = self._decoder_module
        if block_name == "res_x":
            num_layers = int(block_config.get("num_layers", 1))
            return (
                decoder_module.ResBlockGroup(
                    in_channels,
                    num_layers,
                    spatial_padding_mode,
                    self.timestep_conditioning,
                ),
                in_channels,
            )

        reduction = int(block_config.get("multiplier", 1))
        if reduction < 1:
            raise ValueError(
                f"LTX decoder block '{block_name}' has invalid multiplier {reduction}"
            )
        residual = bool(block_config.get("residual", False))
        if block_name == "compress_all":
            stride = (2, 2, 2)
        elif block_name == "compress_time":
            stride = (2, 1, 1)
        elif block_name == "compress_space":
            stride = (1, 2, 2)
        else:
            raise ValueError(f"Unsupported LTX decoder block '{block_name}'")
        return (
            decoder_module.DepthToSpaceUpsample(
                dims=3,
                in_channels=in_channels,
                stride=stride,
                residual=residual,
                out_channels_reduction_factor=reduction,
                spatial_padding_mode=spatial_padding_mode,
            ),
            in_channels // reduction,
        )

    def denormalize(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        mean = self.latents_mean.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        std = self.latents_std.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        return (x * std + mean).astype(dtype)

    def pixel_norm(self, x: mx.array, eps: float = 1e-8) -> mx.array:
        return x / mx.sqrt(mx.mean(x**2, axis=1, keepdims=True) + eps)

    def __call__(
        self,
        sample: mx.array,
        *,
        causal: bool = False,
        timestep: mx.array | None = None,
        debug: bool = False,
        chunked_conv: bool = False,
    ) -> mx.array:
        del debug
        effective_causal = causal or self.causal_decoder
        batch_size = int(sample.shape[0])
        if self.timestep_conditioning:
            noise = mx.random.normal(sample.shape) * self.decode_noise_scale
            sample = noise + (1.0 - self.decode_noise_scale) * sample
        sample = self.denormalize(sample)

        if timestep is None and self.timestep_conditioning:
            timestep = mx.full((batch_size,), self.decode_timestep)

        scaled_timestep = None
        if self.timestep_conditioning and timestep is not None:
            scaled_timestep = timestep * self.timestep_scale_multiplier

        x = self.conv_in(sample, causal=effective_causal)
        for block in self.up_blocks.values():
            if isinstance(block, self._decoder_module.ResBlockGroup):
                x = block(x, causal=effective_causal, timestep=scaled_timestep)
            elif isinstance(block, self._decoder_module.DepthToSpaceUpsample):
                x = block(x, causal=effective_causal, chunked_conv=chunked_conv)
            else:
                x = block(x, causal=effective_causal)

        x = self.pixel_norm(x)
        if self.timestep_conditioning and scaled_timestep is not None:
            embedded_timestep = self.last_time_embedder(
                scaled_timestep.flatten(),
                hidden_dtype=x.dtype,
            )
            embedded_timestep = embedded_timestep.reshape(
                batch_size,
                2,
                self._final_feature_channels,
                1,
                1,
                1,
            )
            ada_values = (
                self.last_scale_shift_table[None, :, :, None, None, None]
                + embedded_timestep
            )
            shift = ada_values[:, 0]
            scale = ada_values[:, 1]
            x = x * (1 + scale) + shift

        x = self.act(x)
        x = self.conv_out(x, causal=effective_causal)
        return self._decoder_module.unpatchify(
            x,
            patch_size_hw=self.patch_size,
            patch_size_t=1,
        )

    def decode_tiled(
        self,
        sample: mx.array,
        *,
        tiling_config: object | None = None,
        tiling_mode: str = "auto",
        causal: bool = False,
        timestep: mx.array | None = None,
        debug: bool = False,
        on_frames_ready: object | None = None,
    ) -> mx.array:
        effective_causal = causal or self.causal_decoder
        if tiling_config is None:
            tiling_config = self._decoder_module.TilingConfig.default()

        _, _, frames, latent_h, latent_w = sample.shape
        needs_spatial_tiling = False
        needs_temporal_tiling = False
        spatial_scale = 32
        temporal_scale = 8

        if getattr(tiling_config, "spatial_config", None) is not None:
            spatial_config = tiling_config.spatial_config
            tile_size_latent = spatial_config.tile_size_in_pixels // spatial_scale
            if latent_h > tile_size_latent or latent_w > tile_size_latent:
                needs_spatial_tiling = True

        if getattr(tiling_config, "temporal_config", None) is not None:
            temporal_config = tiling_config.temporal_config
            tile_size_latent = temporal_config.tile_size_in_frames // temporal_scale
            if frames > tile_size_latent:
                needs_temporal_tiling = True

        use_chunked_conv = tiling_mode in (
            "conservative",
            "none",
            "auto",
            "default",
            "spatial",
        )
        if not needs_spatial_tiling and not needs_temporal_tiling:
            decoded = self(
                sample,
                causal=effective_causal,
                timestep=timestep,
                debug=debug,
                chunked_conv=use_chunked_conv,
            )
            if on_frames_ready is not None:
                try:
                    on_frames_ready(decoded, 0)
                except Exception:
                    return decoded
            return decoded

        return self._decoder_module.decode_with_tiling(
            decoder_fn=self,
            latents=sample,
            tiling_config=tiling_config,
            spatial_scale=32,
            temporal_scale=8,
            causal=effective_causal,
            timestep=timestep,
            chunked_conv=use_chunked_conv,
            on_frames_ready=on_frames_ready,
        )


@dataclass(slots=True)
class LTXDistilledVideoGenerator(VideoGenerator):
    checkpoint_path: Path
    spatial_upsampler_path: Path
    _reference_imports: _ReferenceImports | None = None
    _transformer: object | None = None
    _vae_decoder: object | None = None
    _vae_encoder: object | None = None
    _upsampler: object | None = None
    _audio_decoder: object | None = None
    _vocoder: object | None = None
    _audio_output_sample_rate: int | None = None
    _audio_backend: str | None = None

    def generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        conditioning_inputs: tuple[ConditioningInput, ...],
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int | None = None,
    ) -> GeneratedVideo:
        if width < 32 or height < 32:
            raise ValueError("LTX generation requires width and height >= 32")
        if num_frames < 1:
            raise ValueError("LTX generation requires at least one frame")
        if num_frames % 8 != 1:
            raise ValueError("LTX generation requires num_frames to satisfy 8n+1")
        if fps < 1:
            raise ValueError("LTX generation requires fps >= 1")

        imports = self._imports()
        runtime_config = _runtime_model_config(self.checkpoint_path)
        _assert_prompt_runtime_contract(prompt_context, runtime_config)
        padded_shape = _resolve_padded_shape(width=width, height=height)
        effective_seed = _effective_seed(
            prompt_context=prompt_context,
            checkpoint_path=self.checkpoint_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            seed=seed,
        )
        model_dtype = _prompt_context_dtype(prompt_context.video_context)
        latent_frames = 1 + (num_frames - 1) // 8
        stage1_height = padded_shape.internal_height // 2 // 32
        stage1_width = padded_shape.internal_width // 2 // 32
        stage2_height = padded_shape.internal_height // 32
        stage2_width = padded_shape.internal_width // 32
        audio_frames = int(imports.compute_audio_frames(num_frames, float(fps)))

        _debug_progress("ensure_transformer start")
        transformer = self._ensure_transformer(imports, runtime_config, prompt_context)
        _debug_progress("ensure_transformer done")
        _debug_progress("ensure_vae_decoder start")
        vae_decoder = self._ensure_vae_decoder(imports)
        _debug_progress("ensure_vae_decoder done")
        _debug_progress("ensure_upsampler start")
        upsampler = self._ensure_upsampler(imports)
        _debug_progress("ensure_upsampler done")
        conditioning_plan = self._prepare_conditionings(
            imports=imports,
            conditioning_inputs=conditioning_inputs,
            num_frames=num_frames,
            latent_frames=latent_frames,
            padded_shape=padded_shape,
            model_dtype=model_dtype,
        )

        mx.random.seed(effective_seed)
        timings_ms: dict[str, float] = {}

        stage1_started = time.perf_counter()
        _debug_progress("stage1 setup")
        stage1_positions = imports.create_position_grid(
            1,
            latent_frames,
            stage1_height,
            stage1_width,
            fps=float(fps),
        )
        stage1_audio_positions = imports.create_audio_position_grid(1, audio_frames)
        latents = mx.random.normal(
            (1, 128, latent_frames, stage1_height, stage1_width)
        ).astype(model_dtype)
        audio_latents = mx.random.normal(
            (
                1,
                imports.audio_latent_channels,
                audio_frames,
                imports.audio_mel_bins,
            )
        ).astype(model_dtype)
        if conditioning_plan.stage1:
            stage1_state = self._apply_conditionings_to_stage(
                imports=imports,
                latents=latents,
                conditionings=conditioning_plan.stage1,
                sigmas=imports.stage_1_sigmas,
            )
            latents = stage1_state.latent
        else:
            stage1_state = None
        latents, audio_latents = _denoise_distilled_audio_video(
            imports=imports,
            transformer=transformer,
            latents=latents,
            positions=stage1_positions,
            text_embeddings=_require_video_context(prompt_context),
            audio_latents=audio_latents,
            audio_positions=stage1_audio_positions,
            audio_embeddings=_require_audio_context(prompt_context),
            sigmas=imports.stage_1_sigmas,
            state=stage1_state,
            runtime_config=runtime_config,
        )
        mx.eval(latents, audio_latents)
        debug_dir = _debug_stage_dump_dir()
        if debug_dir is not None:
            _debug_progress("debug decode stage1")
            stage1_padded_shape = _half_resolution_padded_shape(
                padded_shape=padded_shape,
                width=width,
                height=height,
            )
            stage1_decoded, stage1_tiling = self._decode_video(
                imports=imports,
                vae_decoder=vae_decoder,
                latents=latents,
                padded_shape=stage1_padded_shape,
                num_frames=num_frames,
            )
            stage1_frames = _decode_to_uint8_frames(
                stage1_decoded,
                padded_shape=stage1_padded_shape,
            )
            _emit_debug_frame_snapshot(
                debug_dir=debug_dir,
                stage_name="stage1",
                frames_uint8=stage1_frames,
                metadata={
                    "tiling_mode": stage1_tiling,
                    "latents": _latent_stats(latents),
                },
            )
            mx.clear_cache()
        timings_ms["stage1_duration_ms"] = _elapsed_ms(stage1_started)

        upsample_started = time.perf_counter()
        _debug_progress("upsample start")
        latents = imports.upsample_latents(
            latents,
            upsampler,
            vae_decoder.latents_mean,
            vae_decoder.latents_std,
        )
        mx.eval(latents)
        if debug_dir is not None:
            _debug_progress("debug decode post_x2")
            post_x2_decoded, post_x2_tiling = self._decode_video(
                imports=imports,
                vae_decoder=vae_decoder,
                latents=latents,
                padded_shape=padded_shape,
                num_frames=num_frames,
            )
            post_x2_frames = _decode_to_uint8_frames(
                post_x2_decoded,
                padded_shape=padded_shape,
            )
            _emit_debug_frame_snapshot(
                debug_dir=debug_dir,
                stage_name="post_x2",
                frames_uint8=post_x2_frames,
                metadata={
                    "tiling_mode": post_x2_tiling,
                    "latents": _latent_stats(latents),
                },
            )
            mx.clear_cache()
        timings_ms["upsample_duration_ms"] = _elapsed_ms(upsample_started)

        stage2_started = time.perf_counter()
        _debug_progress("stage2 setup")
        stage2_positions = imports.create_position_grid(
            1,
            latent_frames,
            stage2_height,
            stage2_width,
            fps=float(fps),
        )
        stage2_audio_positions = imports.create_audio_position_grid(1, audio_frames)
        if conditioning_plan.stage2:
            stage2_state = self._apply_conditionings_to_stage(
                imports=imports,
                latents=latents,
                conditionings=conditioning_plan.stage2,
                sigmas=imports.stage_2_sigmas,
            )
            latents = stage2_state.latent
            noise_scale = mx.array(float(imports.stage_2_sigmas[0]), dtype=model_dtype)
            one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
            audio_latents = (
                mx.random.normal(audio_latents.shape).astype(model_dtype) * noise_scale
                + audio_latents * one_minus_scale
            ).astype(model_dtype)
            mx.eval(latents, audio_latents)
        else:
            stage2_state = None
            noise_scale = mx.array(float(imports.stage_2_sigmas[0]), dtype=model_dtype)
            one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
            latents = (
                mx.random.normal(latents.shape).astype(model_dtype) * noise_scale
                + latents * one_minus_scale
            ).astype(model_dtype)
            audio_latents = (
                mx.random.normal(audio_latents.shape).astype(model_dtype) * noise_scale
                + audio_latents * one_minus_scale
            ).astype(model_dtype)
            mx.eval(latents, audio_latents)
        latents, audio_latents = _denoise_distilled_audio_video(
            imports=imports,
            transformer=transformer,
            latents=latents,
            positions=stage2_positions,
            text_embeddings=_require_video_context(prompt_context),
            audio_latents=audio_latents,
            audio_positions=stage2_audio_positions,
            audio_embeddings=_require_audio_context(prompt_context),
            sigmas=imports.stage_2_sigmas,
            state=stage2_state,
            runtime_config=runtime_config,
        )
        mx.eval(latents, audio_latents)
        timings_ms["stage2_duration_ms"] = _elapsed_ms(stage2_started)

        decode_started = time.perf_counter()
        decoded_video, tiling_mode = self._decode_video(
            imports=imports,
            vae_decoder=vae_decoder,
            latents=latents,
            padded_shape=padded_shape,
            num_frames=num_frames,
        )
        timings_ms["decode_duration_ms"] = _elapsed_ms(decode_started)
        frames_uint8 = _decode_to_uint8_frames(decoded_video, padded_shape=padded_shape)
        audio_decode_started = time.perf_counter()
        audio_waveform, audio_sample_rate, audio_backend = self._decode_audio_waveform(
            imports=imports,
            audio_latents=audio_latents,
        )
        timings_ms["audio_decode_duration_ms"] = _elapsed_ms(audio_decode_started)
        if debug_dir is not None:
            _emit_debug_frame_snapshot(
                debug_dir=debug_dir,
                stage_name="final",
                frames_uint8=frames_uint8,
                metadata={
                    "tiling_mode": tiling_mode,
                    "latents": _latent_stats(latents),
                },
            )
        mx.clear_cache()

        return GeneratedVideo(
            frames=frames_uint8,
            fps=fps,
            seed=effective_seed,
            backend="mlx_video_distilled_two_stage_bridge",
            conditioning_count=len(conditioning_inputs),
            prompt_signature=_prompt_signature(prompt_context.prompt_text),
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            metadata={
                "pipeline_kind": "distilled_two_stage",
                "stage1_duration_ms": timings_ms["stage1_duration_ms"],
                "upsample_duration_ms": timings_ms["upsample_duration_ms"],
                "stage2_duration_ms": timings_ms["stage2_duration_ms"],
                "decode_duration_ms": timings_ms["decode_duration_ms"],
                "audio_decode_duration_ms": timings_ms["audio_decode_duration_ms"],
                "tiling_mode": tiling_mode,
                "output_width": width,
                "output_height": height,
                "output_frames": num_frames,
                "internal_width": padded_shape.internal_width,
                "internal_height": padded_shape.internal_height,
                "audio_present": audio_waveform is not None,
                "audio_sample_rate": audio_sample_rate,
                "audio_channels": (
                    int(audio_waveform.shape[1])
                    if audio_waveform is not None and audio_waveform.ndim == 2
                    else 1
                    if audio_waveform is not None
                    else 0
                ),
                "audio_backend": audio_backend,
                "audio_bwe_applied": audio_backend == "mlx_vocoder_with_bwe",
            },
        )

    def close(self) -> None:
        self._transformer = None
        self._vae_decoder = None
        self._vae_encoder = None
        self._upsampler = None
        self._audio_decoder = None
        self._vocoder = None
        self._audio_output_sample_rate = None
        self._audio_backend = None
        mx.clear_cache()

    def _imports(self) -> _ReferenceImports:
        if self._reference_imports is not None:
            return self._reference_imports

        with _reference_path_on_sys_path():
            conditioning_module = importlib.import_module("mlx_video.conditioning")
            conditioning_latent_module = importlib.import_module(
                "mlx_video.conditioning.latent"
            )
            generate_module = importlib.import_module("mlx_video.generate")
            convert_module = importlib.import_module("mlx_video.convert")
            config_module = importlib.import_module("mlx_video.models.ltx.config")
            ltx_module = importlib.import_module("mlx_video.models.ltx.ltx")
            attention_module = importlib.import_module("mlx_video.models.ltx.attention")
            adaln_module = importlib.import_module("mlx_video.models.ltx.adaln")
            rope_module = importlib.import_module("mlx_video.models.ltx.rope")
            feed_forward_module = importlib.import_module(
                "mlx_video.models.ltx.feed_forward"
            )
            transformer_module = importlib.import_module(
                "mlx_video.models.ltx.transformer"
            )
            upsampler_module = importlib.import_module("mlx_video.models.ltx.upsampler")
            decoder_module = importlib.import_module(
                "mlx_video.models.ltx.video_vae.decoder"
            )
            encoder_module = importlib.import_module(
                "mlx_video.models.ltx.video_vae.encoder"
            )
            audio_vae_module = importlib.import_module(
                "mlx_video.models.ltx.audio_vae.audio_vae"
            )
            tiling_module = importlib.import_module(
                "mlx_video.models.ltx.video_vae.tiling"
            )
            utils_module = importlib.import_module("mlx_video.utils")

        self._reference_imports = _ReferenceImports(
            model_class=ltx_module.LTXModel,
            model_config_class=config_module.LTXModelConfig,
            model_type_enum=config_module.LTXModelType,
            rope_type_enum=config_module.LTXRopeType,
            BasicAVTransformerBlock=transformer_module.BasicAVTransformerBlock,
            attention_class=attention_module.Attention,
            preprocessor_class=ltx_module.TransformerArgsPreprocessor,
            multi_preprocessor_class=ltx_module.MultiModalTransformerArgsPreprocessor,
            feed_forward_class=feed_forward_module.FeedForward,
            adaln_class=adaln_module.AdaLayerNormSingle,
            apply_rotary_emb=rope_module.apply_rotary_emb,
            precompute_freqs_cis=rope_module.precompute_freqs_cis,
            rms_norm=utils_module.rms_norm,
            to_denoised=utils_module.to_denoised,
            scaled_dot_product_attention=attention_module.scaled_dot_product_attention,
            latent_state_class=conditioning_latent_module.LatentState,
            tiling_config_class=tiling_module.TilingConfig,
            condition_class=conditioning_module.VideoConditionByLatentIndex,
            stage_1_sigmas=tuple(
                float(value) for value in generate_module.STAGE_1_SIGMAS
            ),
            stage_2_sigmas=tuple(
                float(value) for value in generate_module.STAGE_2_SIGMAS
            ),
            apply_conditioning=conditioning_module.apply_conditioning,
            apply_denoise_mask=conditioning_latent_module.apply_denoise_mask,
            create_position_grid=generate_module.create_position_grid,
            create_audio_position_grid=generate_module.create_audio_position_grid,
            compute_audio_frames=generate_module.compute_audio_frames,
            load_image=utils_module.load_image,
            load_upsampler=upsampler_module.load_upsampler,
            load_vae_decoder=decoder_module.load_vae_decoder,
            load_vae_encoder=encoder_module.load_vae_encoder,
            load_audio_decoder=generate_module.load_audio_decoder,
            load_vocoder=generate_module.load_vocoder,
            decode_audio=audio_vae_module.decode_audio,
            sanitize_audio_vae_weights=convert_module.sanitize_audio_vae_weights,
            sanitize_vocoder_weights=convert_module.sanitize_vocoder_weights,
            audio_vocoder_class=importlib.import_module(
                "mlx_video.models.ltx.audio_vae.vocoder"
            ).Vocoder,
            prepare_image_for_encoding=utils_module.prepare_image_for_encoding,
            upsample_latents=upsampler_module.upsample_latents,
            distilled_pipeline_type=generate_module.PipelineType.DISTILLED,
            audio_latent_channels=int(generate_module.AUDIO_LATENT_CHANNELS),
            audio_mel_bins=int(generate_module.AUDIO_MEL_BINS),
            audio_sample_rate=int(generate_module.AUDIO_SAMPLE_RATE),
        )
        _patch_reference_modules(self._reference_imports)
        return self._reference_imports

    def _ensure_transformer(
        self,
        imports: _ReferenceImports,
        runtime_config: _RuntimeModelConfig,
        prompt_context: PromptEncodingResult,
    ) -> object:
        if self._transformer is not None:
            return self._transformer

        config = imports.model_config_class.from_dict(
            {
                "model_type": imports.model_type_enum.AudioVideo,
                "num_attention_heads": runtime_config.num_attention_heads,
                "attention_head_dim": runtime_config.attention_head_dim,
                "in_channels": runtime_config.in_channels,
                "out_channels": runtime_config.out_channels,
                "num_layers": runtime_config.num_layers,
                "cross_attention_dim": runtime_config.cross_attention_dim,
                "caption_channels": _context_width(prompt_context.video_context),
                "audio_num_attention_heads": runtime_config.audio_num_attention_heads,
                "audio_attention_head_dim": runtime_config.audio_attention_head_dim,
                "audio_in_channels": runtime_config.audio_in_channels,
                "audio_out_channels": runtime_config.audio_out_channels,
                "audio_cross_attention_dim": runtime_config.audio_cross_attention_dim,
                "audio_caption_channels": _context_width(
                    _require_audio_context(prompt_context)
                ),
                "positional_embedding_theta": runtime_config.positional_embedding_theta,
                "positional_embedding_max_pos": runtime_config.positional_embedding_max_pos,
                "audio_positional_embedding_max_pos": runtime_config.audio_positional_embedding_max_pos,
                "use_middle_indices_grid": runtime_config.use_middle_indices_grid,
                "rope_type": prompt_context.rope_type,
                "double_precision_rope": prompt_context.double_precision_rope,
                "timestep_scale_multiplier": runtime_config.timestep_scale_multiplier,
                "av_ca_timestep_scale_multiplier": runtime_config.av_ca_timestep_scale_multiplier,
                "norm_eps": runtime_config.norm_eps,
            }
        )
        config.apply_gated_attention = runtime_config.apply_gated_attention
        config.cross_attention_adaln = runtime_config.cross_attention_adaln
        config.caption_proj_before_connector = (
            prompt_context.caption_proj_before_connector
        )
        transformer = imports.model_class.from_pretrained(
            self.checkpoint_path,
            config=config,
            strict=True,
        )
        if runtime_config.apply_gated_attention:
            first_block = next(iter(transformer.transformer_blocks.values()))
            required_gate_attrs = (
                "attn1",
                "attn2",
                "audio_attn1",
                "audio_attn2",
                "audio_to_video_attn",
                "video_to_audio_attn",
            )
            missing_gate_attrs = [
                attr
                for attr in required_gate_attrs
                if not hasattr(getattr(first_block, attr, None), "to_gate_logits")
            ]
            if missing_gate_attrs:
                raise RuntimeError(
                    "LTX checkpoint requires gated attention, but the MLX transformer "
                    "bridge did not instantiate gate projections for "
                    + ", ".join(missing_gate_attrs)
                )
        if runtime_config.cross_attention_adaln:
            if not hasattr(transformer, "prompt_adaln_single"):
                raise RuntimeError(
                    "LTX checkpoint requires cross-attention AdaLN, but the MLX "
                    "transformer bridge did not instantiate prompt AdaLN"
                )
            if not hasattr(transformer, "audio_prompt_adaln_single"):
                raise RuntimeError(
                    "LTX checkpoint requires audio prompt AdaLN, but the MLX "
                    "transformer bridge did not instantiate the audio prompt AdaLN"
                )
            linear = getattr(transformer.adaln_single, "linear", None)
            if (
                linear is None
                or int(linear.weight.shape[0]) != 9 * transformer.inner_dim
            ):
                raise RuntimeError(
                    "LTX checkpoint requires 9-way AdaLN modulation, but the MLX "
                    "transformer bridge is still using the wrong AdaLN shape"
                )
            audio_linear = getattr(transformer.audio_adaln_single, "linear", None)
            if (
                audio_linear is None
                or int(audio_linear.weight.shape[0]) != 9 * transformer.audio_inner_dim
            ):
                raise RuntimeError(
                    "LTX checkpoint requires 9-way audio AdaLN modulation, but the "
                    "MLX transformer bridge is still using the wrong audio AdaLN shape"
                )
            first_block = next(iter(transformer.transformer_blocks.values()))
            if not hasattr(first_block, "audio_prompt_scale_shift_table"):
                raise RuntimeError(
                    "LTX checkpoint requires audio prompt AdaLN tables, but the MLX "
                    "transformer bridge did not instantiate them"
                )
        mx.eval(transformer.parameters())
        self._transformer = transformer
        return transformer

    def _ensure_vae_decoder(self, imports: _ReferenceImports) -> object:
        del imports
        if self._vae_decoder is None:
            self._vae_decoder = _load_configured_vae_decoder(self.checkpoint_path)
            mx.eval(self._vae_decoder.parameters())
        return self._vae_decoder

    def _ensure_vae_encoder(self, imports: _ReferenceImports) -> object:
        if self._vae_encoder is None:
            self._vae_encoder = imports.load_vae_encoder(str(self.checkpoint_path))
            mx.eval(self._vae_encoder.parameters())
        return self._vae_encoder

    def _ensure_upsampler(self, imports: _ReferenceImports) -> object:
        del imports
        if self._upsampler is None:
            self._upsampler = _load_configured_upsampler(self.spatial_upsampler_path)
            mx.eval(self._upsampler.parameters())
        return self._upsampler

    def _ensure_audio_stack(
        self, imports: _ReferenceImports
    ) -> tuple[object, object, int, str]:
        if (
            self._audio_decoder is not None
            and self._vocoder is not None
            and self._audio_output_sample_rate is not None
            and self._audio_backend is not None
        ):
            return (
                self._audio_decoder,
                self._vocoder,
                self._audio_output_sample_rate,
                self._audio_backend,
            )

        checkpoint_audio_weights = _load_checkpoint_prefixed_weights(
            self.checkpoint_path,
            prefixes=("audio_vae.", "vocoder."),
        )
        sanitized_audio_weights = {
            f"audio_vae.{key}": value
            for key, value in imports.sanitize_audio_vae_weights(
                checkpoint_audio_weights
            ).items()
        }
        checkpoint_root = self.checkpoint_path.parent
        if self._audio_decoder is None:
            self._audio_decoder = imports.load_audio_decoder(
                checkpoint_root,
                imports.distilled_pipeline_type,
                unified_weights=sanitized_audio_weights,
            )
            mx.eval(self._audio_decoder.parameters())
        if self._vocoder is None:
            (
                self._vocoder,
                self._audio_output_sample_rate,
                self._audio_backend,
            ) = _load_runtime_vocoder(
                checkpoint_path=self.checkpoint_path,
                checkpoint_weights=checkpoint_audio_weights,
                sanitize_vocoder_weights=imports.sanitize_vocoder_weights,
            )
            mx.eval(self._vocoder.parameters())
        mx.clear_cache()
        return (
            self._audio_decoder,
            self._vocoder,
            self._audio_output_sample_rate,
            self._audio_backend,
        )

    def _decode_audio_waveform(
        self,
        *,
        imports: _ReferenceImports,
        audio_latents: object,
    ) -> tuple[npt.NDArray[np.float32] | None, int, str]:
        audio_decoder, vocoder, output_sample_rate, backend_label = (
            self._ensure_audio_stack(imports)
        )
        decoded_audio = imports.decode_audio(
            audio_latents.astype(mx.float32),
            audio_decoder,
            vocoder,
        )
        mx.eval(decoded_audio)
        return (
            _audio_waveform_to_numpy(decoded_audio),
            output_sample_rate,
            backend_label,
        )

    def _prepare_conditionings(
        self,
        *,
        imports: _ReferenceImports,
        conditioning_inputs: tuple[ConditioningInput, ...],
        num_frames: int,
        latent_frames: int,
        padded_shape: _PaddedShape,
        model_dtype: mx.Dtype,
    ) -> _ConditioningPlan:
        if not conditioning_inputs:
            return _ConditioningPlan(stage1=(), stage2=())

        vae_encoder = self._ensure_vae_encoder(imports)
        stage1_width = padded_shape.internal_width // 2
        stage1_height = padded_shape.internal_height // 2
        stage1_conditionings: list[object] = []
        stage2_conditionings: list[object] = []

        for conditioning_input in conditioning_inputs:
            resolved_index = _resolve_latent_frame_index(
                frame_index=conditioning_input.frame_index,
                num_frames=num_frames,
                latent_frames=latent_frames,
            )
            stage1_latent = _encode_conditioning_latent(
                imports=imports,
                vae_encoder=vae_encoder,
                payload_path=conditioning_input.payload_path,
                width=stage1_width,
                height=stage1_height,
                dtype=model_dtype,
            )
            stage2_latent = _encode_conditioning_latent(
                imports=imports,
                vae_encoder=vae_encoder,
                payload_path=conditioning_input.payload_path,
                width=padded_shape.internal_width,
                height=padded_shape.internal_height,
                dtype=model_dtype,
            )
            stage1_conditionings.append(
                imports.condition_class(
                    latent=stage1_latent,
                    frame_idx=resolved_index,
                    strength=float(conditioning_input.strength),
                )
            )
            stage2_conditionings.append(
                imports.condition_class(
                    latent=stage2_latent,
                    frame_idx=resolved_index,
                    strength=float(conditioning_input.strength),
                )
            )

        return _ConditioningPlan(
            stage1=tuple(stage1_conditionings),
            stage2=tuple(stage2_conditionings),
        )

    def _apply_conditionings_to_stage(
        self,
        *,
        imports: _ReferenceImports,
        latents: object,
        conditionings: tuple[object, ...],
        sigmas: tuple[float, ...],
    ) -> object:
        model_dtype = latents.dtype
        state = imports.latent_state_class(
            latent=latents,
            clean_latent=mx.zeros_like(latents),
            denoise_mask=mx.ones(
                (int(latents.shape[0]), 1, int(latents.shape[2]), 1, 1),
                dtype=model_dtype,
            ),
        )
        state = imports.apply_conditioning(state, list(conditionings))
        noise = mx.random.normal(latents.shape).astype(model_dtype)
        noise_scale = mx.array(float(sigmas[0]), dtype=model_dtype)
        scaled_mask = state.denoise_mask * noise_scale
        conditioned = (
            noise * scaled_mask
            + state.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask)
        ).astype(model_dtype)
        mx.eval(conditioned)
        return replace(state, latent=conditioned)

    def _decode_video(
        self,
        *,
        imports: _ReferenceImports,
        vae_decoder: object,
        latents: object,
        padded_shape: _PaddedShape,
        num_frames: int,
    ) -> tuple[object, str]:
        tiling_config = imports.tiling_config_class.auto(
            padded_shape.internal_height,
            padded_shape.internal_width,
            num_frames,
        )
        if tiling_config is None:
            video = vae_decoder(latents, chunked_conv=False)
            mx.eval(video)
            return video, "none"

        try:
            video = vae_decoder(latents, chunked_conv=False)
            mx.eval(video)
            return video, "auto-fast"
        except Exception as exc:
            if not _looks_like_metal_oom(exc):
                raise

        try:
            video = vae_decoder(latents, chunked_conv=True)
            mx.eval(video)
            return video, "chunked"
        except Exception as exc:
            if not _looks_like_metal_oom(exc):
                raise

        video = vae_decoder.decode_tiled(
            latents,
            tiling_config=tiling_config,
            tiling_mode="auto",
            debug=False,
        )
        mx.eval(video)
        return video, "auto-tiled"


@dataclass(slots=True)
class LTXPreviewVideoGenerator(VideoGenerator):
    checkpoint_path: Path
    spatial_upsampler_path: Path

    def generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        conditioning_inputs: tuple[ConditioningInput, ...],
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int | None = None,
    ) -> GeneratedVideo:
        if width < 32 or height < 32:
            raise ValueError("LTX preview generation requires width and height >= 32")
        if num_frames < 1:
            raise ValueError("LTX preview generation requires at least one frame")
        if fps < 1:
            raise ValueError("LTX preview generation requires fps >= 1")
        effective_seed = _effective_seed(
            prompt_context=prompt_context,
            checkpoint_path=self.checkpoint_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            seed=seed,
        )
        mx.random.seed(effective_seed)
        frames = _base_frames(
            width=width,
            height=height,
            num_frames=num_frames,
            prompt_context=prompt_context,
            seed=effective_seed,
        )
        for conditioning_input in conditioning_inputs:
            frames = _apply_conditioning(
                frames=frames,
                conditioning_input=conditioning_input,
                width=width,
                height=height,
                num_frames=num_frames,
            )
        frames_uint8 = np.asarray((mx.clip(frames, 0.0, 1.0) * 255.0).astype(mx.uint8))
        return GeneratedVideo(
            frames=frames_uint8,
            fps=fps,
            seed=effective_seed,
            backend="mlx_prompt_conditioned_preview",
            conditioning_count=len(conditioning_inputs),
            prompt_signature=_prompt_signature(prompt_context.prompt_text),
            metadata={
                "pipeline_kind": "preview",
                "output_width": width,
                "output_height": height,
                "output_frames": num_frames,
                "tiling_mode": "none",
                "audio_present": False,
            },
        )

    def close(self) -> None:
        return None


def create_video_generator(
    checkpoint_path: Path,
    spatial_upsampler_path: Path,
) -> VideoGenerator:
    return (
        LTXDistilledVideoGenerator(
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
        )
        if can_use_reference_backend(checkpoint_path, spatial_upsampler_path)
        else LTXPreviewVideoGenerator(
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
        )
    )


def encode_mp4_video(*, video: GeneratedVideo, output_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("Current LTX mp4 output requires an 'ffmpeg' binary on PATH")
    if video.audio_waveform is not None and video.audio_sample_rate is None:
        raise ValueError("Generated audio requires audio_sample_rate metadata")
    if video.frames.ndim != 4 or video.frames.shape[-1] != 3:
        raise ValueError(
            "Generated video frames must have shape [frames, height, width, 3]"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if video.audio_waveform is None:
        _encode_video_only_mp4(
            video=video, output_path=output_path, ffmpeg_path=ffmpeg_path
        )
        return

    with tempfile.TemporaryDirectory(prefix="mlxr-ltx-audio-export-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        video_only_path = tmp_root / "video-only.mp4"
        audio_path = tmp_root / "audio.wav"
        _encode_video_only_mp4(
            video=video,
            output_path=video_only_path,
            ffmpeg_path=ffmpeg_path,
        )
        encode_wav_audio(video=video, output_path=audio_path)
        _mux_mp4_with_audio(
            ffmpeg_path=ffmpeg_path,
            video_path=video_only_path,
            audio_path=audio_path,
            output_path=output_path,
            audio_sample_rate=video.audio_sample_rate,
        )


def encode_wav_audio(*, video: GeneratedVideo, output_path: Path) -> None:
    if video.audio_waveform is None or video.audio_sample_rate is None:
        raise ValueError("Generated video does not contain decodable audio output")

    audio = _normalized_audio_waveform(video.audio_waveform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    audio_int16 = (audio * 32767.0).astype(np.int16)

    import wave

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(video.audio_sample_rate))
        wav_file.writeframes(audio_int16.tobytes(order="C"))


def _encode_video_only_mp4(
    *,
    video: GeneratedVideo,
    output_path: Path,
    ffmpeg_path: str,
) -> None:
    num_frames, height, width, _ = video.frames.shape
    if num_frames < 1:
        raise ValueError("Generated video must contain at least one frame")

    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(video.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "mpeg4",
            "-q:v",
            "4",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        input=video.frames.tobytes(order="C"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to encode mp4 output: {stderr}")


def _mux_mp4_with_audio(
    *,
    ffmpeg_path: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_sample_rate: int | None,
) -> None:
    if audio_sample_rate is None:
        raise ValueError("Generated audio requires audio_sample_rate metadata")

    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-ar",
            str(audio_sample_rate),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to mux mp4 audio output: {stderr}")


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, object]:
    with safe_open(str(checkpoint_path), framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
    raw_config = metadata.get("config")
    if not isinstance(raw_config, str):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing JSON config metadata"
        )
    parsed = json.loads(raw_config)
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid config metadata"
        )
    return parsed


def _checkpoint_keys(checkpoint_path: Path) -> set[str]:
    with safe_open(str(checkpoint_path), framework="numpy") as checkpoint:
        return set(checkpoint.keys())


def _load_checkpoint_prefixed_weights(
    checkpoint_path: Path,
    *,
    prefixes: tuple[str, ...],
) -> dict[str, mx.array]:
    # Use the same MLX-native loader path as the main bridge. The audio branch in
    # the current 22B checkpoint includes dtypes that do not round-trip cleanly
    # through the numpy view returned by safetensors here.
    weights = mx.load(str(checkpoint_path))
    selected: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.startswith(prefixes):
            selected[key] = value
    if not selected:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing required prefixed weights for {prefixes!r}"
        )
    return selected


def _first_present(mapping: dict[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _decoder_initial_feature_channels(
    *,
    base_channels: int,
    decoder_blocks: tuple[tuple[str, object], ...],
) -> int:
    feature_channels = base_channels
    for block_name, raw_params in decoder_blocks:
        if block_name == "res_x":
            continue
        params = (
            raw_params if isinstance(raw_params, dict) else {"num_layers": raw_params}
        )
        multiplier = int(params.get("multiplier", 1))
        if multiplier < 1:
            raise ValueError(
                f"LTX decoder block '{block_name}' has invalid multiplier {multiplier}"
            )
        feature_channels *= multiplier
    return feature_channels


def _runtime_vae_config(checkpoint_path: Path) -> _RuntimeVAEConfig:
    metadata = _checkpoint_metadata(checkpoint_path)
    raw_vae_config = metadata.get("vae")
    if not isinstance(raw_vae_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing VAE metadata"
        )

    raw_decoder_blocks = raw_vae_config.get("decoder_blocks")
    if not isinstance(raw_decoder_blocks, list) or not raw_decoder_blocks:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing decoder_blocks metadata"
        )

    decoder_blocks: list[tuple[str, object]] = []
    for raw_block in raw_decoder_blocks:
        if (
            isinstance(raw_block, list)
            and len(raw_block) == 2
            and isinstance(raw_block[0], str)
        ):
            decoder_blocks.append((raw_block[0], raw_block[1]))
            continue
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid decoder block metadata: {raw_block!r}"
        )

    norm_layer = str(raw_vae_config.get("norm_layer", "pixel_norm"))
    if norm_layer != "pixel_norm":
        raise NotImplementedError(
            f"LTX real generation only supports pixel_norm VAE decoders, got {norm_layer!r}"
        )

    return _RuntimeVAEConfig(
        latent_channels=int(raw_vae_config.get("latent_channels", 128)),
        out_channels=int(raw_vae_config.get("out_channels", 3)),
        patch_size=int(raw_vae_config.get("patch_size", 4)),
        decoder_blocks=tuple(decoder_blocks),
        base_channels=int(raw_vae_config.get("decoder_base_channels", 128)),
        spatial_padding_mode=str(
            raw_vae_config.get(
                "decoder_spatial_padding_mode",
                raw_vae_config.get("spatial_padding_mode", "reflect"),
            )
        ),
        timestep_conditioning=bool(raw_vae_config.get("timestep_conditioning", True)),
        norm_layer=norm_layer,
        causal_decoder=bool(raw_vae_config.get("causal_decoder", False)),
    )


def _runtime_vocoder_config(checkpoint_path: Path) -> _RuntimeVocoderConfig:
    metadata = _checkpoint_metadata(checkpoint_path)
    raw_vocoder_config = metadata.get("vocoder")
    if not isinstance(raw_vocoder_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing vocoder metadata"
        )

    uses_bwe = isinstance(raw_vocoder_config.get("bwe"), dict) and isinstance(
        raw_vocoder_config.get("vocoder"), dict
    )
    base_config = raw_vocoder_config["vocoder"] if uses_bwe else raw_vocoder_config
    if not isinstance(base_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid base vocoder metadata"
        )
    bwe_config = raw_vocoder_config.get("bwe") if uses_bwe else None
    bwe_output_sample_rate = (
        int(bwe_config.get("output_sampling_rate"))
        if isinstance(bwe_config, dict)
        and bwe_config.get("output_sampling_rate") is not None
        else None
    )
    output_sample_rate = (
        int(bwe_config.get("input_sampling_rate"))
        if isinstance(bwe_config, dict)
        and bwe_config.get("input_sampling_rate") is not None
        else int(base_config.get("output_sampling_rate", 24000))
    )

    return _RuntimeVocoderConfig(
        resblock_kernel_sizes=tuple(
            int(value) for value in base_config.get("resblock_kernel_sizes", [3, 7, 11])
        ),
        upsample_rates=tuple(
            int(value) for value in base_config.get("upsample_rates", [6, 5, 2, 2, 2])
        ),
        upsample_kernel_sizes=tuple(
            int(value)
            for value in base_config.get("upsample_kernel_sizes", [16, 15, 8, 4, 4])
        ),
        resblock_dilation_sizes=tuple(
            tuple(int(v) for v in block)
            for block in base_config.get(
                "resblock_dilation_sizes",
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            )
        ),
        upsample_initial_channel=int(base_config.get("upsample_initial_channel", 1024)),
        stereo=bool(base_config.get("stereo", True)),
        resblock=str(base_config.get("resblock", "1")),
        activation=str(base_config.get("activation", "snake")),
        use_tanh_at_final=bool(base_config.get("use_tanh_at_final", True)),
        apply_final_activation=bool(base_config.get("apply_final_activation", True)),
        use_bias_at_final=bool(base_config.get("use_bias_at_final", True)),
        output_sample_rate=output_sample_rate,
        uses_bwe=uses_bwe,
        bwe_output_sample_rate=bwe_output_sample_rate,
    )


def _load_configured_vae_decoder(checkpoint_path: Path) -> _ConfiguredVideoDecoder:
    vae_config = _runtime_vae_config(checkpoint_path)
    with _reference_path_on_sys_path():
        decoder_module = importlib.import_module(
            "mlx_video.models.ltx.video_vae.decoder"
        )

    spatial_padding_mode = decoder_module.PaddingModeType(
        vae_config.spatial_padding_mode
    )
    decoder = _ConfiguredVideoDecoder(
        decoder_module=decoder_module,
        in_channels=vae_config.latent_channels,
        out_channels=vae_config.out_channels,
        patch_size=vae_config.patch_size,
        decoder_blocks=vae_config.decoder_blocks,
        base_channels=vae_config.base_channels,
        spatial_padding_mode=spatial_padding_mode,
        timestep_conditioning=vae_config.timestep_conditioning,
        causal_decoder=vae_config.causal_decoder,
    )

    weights = mx.load(str(checkpoint_path))
    decoder_weights: dict[str, object] = {}
    for key, value in weights.items():
        if not key.startswith("vae.decoder."):
            continue
        new_key = key[len("vae.decoder.") :]
        if value.ndim == 5 and ".conv.weight" in new_key:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        if ".conv.weight" in new_key or ".conv.bias" in new_key:
            if ".conv.conv.weight" not in new_key and ".conv.conv.bias" not in new_key:
                new_key = new_key.replace(".conv.weight", ".conv.conv.weight")
                new_key = new_key.replace(".conv.bias", ".conv.conv.bias")
        decoder_weights[new_key] = value
    mean = _first_present(
        weights,
        (
            "vae.per_channel_statistics.mean-of-means",
            "vae.per_channel_statistics.mean",
            "per_channel_statistics.mean-of-means",
            "per_channel_statistics.mean",
            "latents_mean",
        ),
    )
    std = _first_present(
        weights,
        (
            "vae.per_channel_statistics.std-of-means",
            "vae.per_channel_statistics.std",
            "per_channel_statistics.std-of-means",
            "per_channel_statistics.std",
            "latents_std",
        ),
    )
    if mean is None or std is None:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing VAE per-channel statistics"
        )
    expected_shape = (vae_config.latent_channels,)
    if tuple(int(size) for size in mean.shape) != expected_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid latents_mean shape "
            f"{tuple(int(size) for size in mean.shape)!r}; expected {expected_shape!r}"
        )
    if tuple(int(size) for size in std.shape) != expected_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid latents_std shape "
            f"{tuple(int(size) for size in std.shape)!r}; expected {expected_shape!r}"
        )
    decoder_weights["latents_mean"] = mean
    decoder_weights["latents_std"] = std
    decoder.load_weights(list(decoder_weights.items()), strict=True)
    return decoder


def _load_configured_upsampler(weights_path: Path) -> object:
    with _reference_path_on_sys_path():
        upsampler_module = importlib.import_module("mlx_video.models.ltx.upsampler")

    _validate_upsampler_layout(weights_path)
    raw_weights = mx.load(str(weights_path))
    sample_key = "res_blocks.0.conv1.weight"
    mid_channels = (
        int(raw_weights[sample_key].shape[0]) if sample_key in raw_weights else 1024
    )
    upsampler = upsampler_module.LatentUpsampler(
        in_channels=128,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
    )

    sanitized: dict[str, object] = {}
    for key, value in raw_weights.items():
        new_key = key
        if value.ndim == 5 and "conv" in key and "weight" in key:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        if value.ndim == 4 and (
            ("conv" in key and "weight" in key) or key == "upsampler.0.weight"
        ):
            value = mx.transpose(value, (0, 2, 3, 1))
        if key.startswith("upsampler.0."):
            new_key = key.replace("upsampler.0.", "upsampler.conv.")
        sanitized[new_key] = value

    upsampler.load_weights(list(sanitized.items()), strict=False)
    return upsampler


def _load_runtime_vocoder(
    *,
    checkpoint_path: Path,
    checkpoint_weights: dict[str, mx.array],
    sanitize_vocoder_weights: object,
) -> tuple[object, int, str]:
    runtime_vocoder_config = _runtime_vocoder_config(checkpoint_path)

    raw_base_weights = {
        key[len("vocoder.vocoder.") :]: value
        for key, value in checkpoint_weights.items()
        if key.startswith("vocoder.vocoder.")
    }
    if not raw_base_weights:
        raw_base_weights = {
            key[len("vocoder.") :]: value
            for key, value in checkpoint_weights.items()
            if key.startswith("vocoder.")
            and not key.startswith("vocoder.bwe_generator.")
        }
    if not raw_base_weights:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing base vocoder weights"
        )

    sanitized_weights = sanitize_vocoder_weights(raw_base_weights)
    sanitized_weights = {
        key: value
        for key, value in sanitized_weights.items()
        if not key.endswith(".filter")
    }
    vocoder = AudioVocoder(
        resblock_kernel_sizes=list(runtime_vocoder_config.resblock_kernel_sizes),
        upsample_rates=list(runtime_vocoder_config.upsample_rates),
        upsample_kernel_sizes=list(runtime_vocoder_config.upsample_kernel_sizes),
        resblock_dilation_sizes=[
            list(block) for block in runtime_vocoder_config.resblock_dilation_sizes
        ],
        upsample_initial_channel=runtime_vocoder_config.upsample_initial_channel,
        stereo=runtime_vocoder_config.stereo,
        resblock=runtime_vocoder_config.resblock,
        output_sample_rate=runtime_vocoder_config.output_sample_rate,
        activation=runtime_vocoder_config.activation,
        use_tanh_at_final=runtime_vocoder_config.use_tanh_at_final,
        apply_final_activation=runtime_vocoder_config.apply_final_activation,
        use_bias_at_final=runtime_vocoder_config.use_bias_at_final,
    )
    vocoder.load_weights(list(sanitized_weights.items()), strict=False)
    backend_label = (
        "mlx_vocoder_amp1_base_only"
        if runtime_vocoder_config.uses_bwe
        else "mlx_vocoder"
    )
    return vocoder, runtime_vocoder_config.output_sample_rate, backend_label


@contextmanager
def _reference_path_on_sys_path() -> object:
    if not _REFERENCE_MLX_VIDEO_ROOT.is_dir():
        raise RuntimeError(
            "LTX real generation requires the repo-local reference checkout at "
            f"'{_REFERENCE_MLX_VIDEO_ROOT}'"
        )
    reference_path = str(_REFERENCE_MLX_VIDEO_ROOT)
    already_present = reference_path in sys.path
    if not already_present:
        sys.path.insert(0, reference_path)
    try:
        yield
    finally:
        if not already_present:
            try:
                sys.path.remove(reference_path)
            except ValueError:
                return None


def _decode_to_uint8_frames(
    decoded_video: object,
    *,
    padded_shape: _PaddedShape,
) -> npt.NDArray[np.uint8]:
    video = mx.squeeze(decoded_video, axis=0)
    video = mx.transpose(video, (1, 2, 3, 0))
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255.0).astype(mx.uint8)
    frames = np.asarray(video)
    if (
        padded_shape.internal_width != padded_shape.output_width
        or padded_shape.internal_height != padded_shape.output_height
    ):
        top = padded_shape.crop_top
        left = padded_shape.crop_left
        frames = frames[
            :,
            top : top + padded_shape.output_height,
            left : left + padded_shape.output_width,
            :,
        ]
    return frames


def _audio_waveform_to_numpy(audio_waveform: object) -> npt.NDArray[np.float32]:
    waveform = np.asarray(audio_waveform, dtype=np.float32)
    if (
        waveform.ndim == 2
        and waveform.shape[0] in {1, 2}
        and waveform.shape[1] > waveform.shape[0]
    ):
        waveform = np.transpose(waveform, (1, 0))
    if waveform.ndim not in {1, 2}:
        raise ValueError(
            "Decoded LTX audio must be mono [samples] or stereo [samples, channels]"
        )
    return _normalized_audio_waveform(waveform)


def _normalized_audio_waveform(
    waveform: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    audio = np.asarray(waveform, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32, copy=False)


def _debug_stage_dump_dir() -> Path | None:
    raw = os.environ.get("MLXR_LTX_DEBUG_STAGE_DUMPS_DIR")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _emit_debug_frame_snapshot(
    *,
    debug_dir: Path,
    stage_name: str,
    frames_uint8: npt.NDArray[np.uint8],
    metadata: dict[str, object],
) -> None:
    Image.fromarray(frames_uint8[0]).save(debug_dir / f"{stage_name}_frame_0001.png")
    (debug_dir / f"{stage_name}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _encode_conditioning_latent(
    *,
    imports: _ReferenceImports,
    vae_encoder: object,
    payload_path: Path,
    width: int,
    height: int,
    dtype: mx.Dtype,
) -> object:
    image = imports.load_image(
        str(payload_path), height=height, width=width, dtype=dtype
    )
    encoded = imports.prepare_image_for_encoding(image, height, width, dtype=dtype)
    latent = vae_encoder(encoded)
    mx.eval(latent)
    return latent


def _resolve_padded_shape(
    *, width: int, height: int, divisor: int = 64
) -> _PaddedShape:
    pad_w = (divisor - (width % divisor)) % divisor
    pad_h = (divisor - (height % divisor)) % divisor
    return _PaddedShape(
        output_width=width,
        output_height=height,
        internal_width=width + pad_w,
        internal_height=height + pad_h,
        crop_top=pad_h // 2,
        crop_left=pad_w // 2,
    )


def _half_resolution_padded_shape(
    *,
    padded_shape: _PaddedShape,
    width: int,
    height: int,
) -> _PaddedShape:
    output_width = max(1, width // 2)
    output_height = max(1, height // 2)
    internal_width = max(1, padded_shape.internal_width // 2)
    internal_height = max(1, padded_shape.internal_height // 2)
    return _PaddedShape(
        output_width=output_width,
        output_height=output_height,
        internal_width=internal_width,
        internal_height=internal_height,
        crop_top=padded_shape.crop_top // 2,
        crop_left=padded_shape.crop_left // 2,
    )


def _resolve_latent_frame_index(
    *, frame_index: int, num_frames: int, latent_frames: int
) -> int:
    if frame_index < latent_frames:
        return frame_index
    if num_frames <= 1 or latent_frames <= 1:
        return 0
    scaled = int((frame_index / (num_frames - 1) * (latent_frames - 1)) + 0.5)
    return int(max(0, min(latent_frames - 1, scaled)))


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def _debug_progress_enabled() -> bool:
    return os.environ.get("MLXR_LTX_DEBUG_PROGRESS") == "1"


def _debug_progress(message: str) -> None:
    if _debug_progress_enabled():
        print(f"[ltx] {message}", flush=True)


def _latent_stats(latents: object) -> dict[str, object]:
    latents_f32 = latents.astype(mx.float32)
    return {
        "shape": [int(size) for size in latents.shape],
        "mean": float(mx.mean(latents_f32).item()),
        "std": float(mx.std(latents_f32).item()),
        "min": float(mx.min(latents_f32).item()),
        "max": float(mx.max(latents_f32).item()),
    }


def _looks_like_metal_oom(exc: BaseException) -> bool:
    message = str(exc)
    needles = (
        "out of memory",
        "Out of memory",
        "OOM",
        "failed to allocate",
        "kIOGPU",
        "Command buffer execution failed",
        "Invalid Resource",
        "MTLCommandBufferError",
        "[METAL]",
    )
    return any(needle in message for needle in needles)


def _prompt_context_dtype(context: object) -> mx.Dtype:
    return getattr(context, "dtype", mx.bfloat16)


def _effective_seed(
    *,
    prompt_context: PromptEncodingResult,
    checkpoint_path: Path,
    spatial_upsampler_path: Path,
    seed: int | None,
) -> int:
    payload = "|".join(
        (
            prompt_context.prompt_text,
            str(prompt_context.token_count),
            str(prompt_context.sequence_length),
            checkpoint_path.name,
            str(checkpoint_path.stat().st_size),
            spatial_upsampler_path.name,
            str(spatial_upsampler_path.stat().st_size),
            str(seed if seed is not None else "auto"),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _prompt_signature(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]


def _validate_upsampler_layout(weights_path: Path) -> None:
    raw_weights = mx.load(str(weights_path))
    if not any(
        key.startswith("upsampler.conv.") or key.startswith("upsampler.0.")
        for key in raw_weights
    ):
        raise RuntimeError(
            f"LTX spatial upsampler '{weights_path}' is missing a supported x2 conv layout"
        )


def can_use_reference_backend(
    checkpoint_path: Path,
    spatial_upsampler_path: Path | None = None,
) -> bool:
    try:
        _validate_reference_backend_compatibility(checkpoint_path)
        if spatial_upsampler_path is not None:
            _validate_upsampler_layout(spatial_upsampler_path)
        if not _REFERENCE_MLX_VIDEO_ROOT.is_dir():
            return False
    except Exception:
        return False
    return True


def _required_transformer_string(
    raw_transformer_config: dict[str, object],
    checkpoint_path: Path,
    key: str,
) -> str:
    value = raw_transformer_config.get(key)
    if not isinstance(value, str):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing required transformer field '{key}'"
        )
    return value


def _required_transformer_bool(
    raw_transformer_config: dict[str, object],
    checkpoint_path: Path,
    key: str,
) -> bool:
    if key not in raw_transformer_config:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing required transformer field '{key}'"
        )
    return bool(raw_transformer_config[key])


def _resolved_transformer_flag(
    *,
    raw_transformer_config: dict[str, object],
    checkpoint_path: Path,
    key: str,
    inferred: bool,
) -> bool:
    explicit = raw_transformer_config.get(key)
    if explicit is None:
        return inferred
    explicit_bool = bool(explicit)
    if inferred and not explicit_bool:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has transformer metadata {key}=False "
            "but matching weights indicate the feature is required"
        )
    return explicit_bool


def _runtime_model_config(checkpoint_path: Path) -> _RuntimeModelConfig:
    metadata = _checkpoint_metadata(checkpoint_path)
    raw_transformer_config = metadata.get("transformer")
    if not isinstance(raw_transformer_config, dict):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing transformer metadata"
        )
    checkpoint_keys = _checkpoint_keys(checkpoint_path)
    default_audio_channels = 8 * 16

    def _int_or_default(value: object | None, default: int) -> int:
        if value is None:
            return default
        return int(value)

    audio_enabled = any(
        key.startswith("model.diffusion_model.audio_")
        or ".audio_" in key
        or "audio_to_video_attn" in key
        or "video_to_audio_attn" in key
        for key in checkpoint_keys
    )
    apply_gated_attention = _resolved_transformer_flag(
        raw_transformer_config=raw_transformer_config,
        checkpoint_path=checkpoint_path,
        key="apply_gated_attention",
        inferred=any(".to_gate_logits." in key for key in checkpoint_keys),
    )
    cross_attention_adaln = _resolved_transformer_flag(
        raw_transformer_config=raw_transformer_config,
        checkpoint_path=checkpoint_path,
        key="cross_attention_adaln",
        inferred=any(
            "prompt_scale_shift_table" in key or "prompt_adaln_single." in key
            for key in checkpoint_keys
        ),
    )
    return _RuntimeModelConfig(
        num_attention_heads=int(raw_transformer_config.get("num_attention_heads", 32)),
        attention_head_dim=int(raw_transformer_config.get("attention_head_dim", 128)),
        in_channels=int(raw_transformer_config.get("in_channels", 128)),
        out_channels=int(raw_transformer_config.get("out_channels", 128)),
        num_layers=int(raw_transformer_config.get("num_layers", 48)),
        cross_attention_dim=int(
            raw_transformer_config.get("cross_attention_dim", 4096)
        ),
        audio_enabled=audio_enabled,
        audio_num_attention_heads=_int_or_default(
            raw_transformer_config.get("audio_num_attention_heads"),
            32,
        ),
        audio_attention_head_dim=_int_or_default(
            raw_transformer_config.get("audio_attention_head_dim"),
            64,
        ),
        audio_in_channels=_int_or_default(
            raw_transformer_config.get("audio_in_channels"),
            default_audio_channels,
        ),
        audio_out_channels=_int_or_default(
            raw_transformer_config.get("audio_out_channels"),
            default_audio_channels,
        ),
        audio_cross_attention_dim=_int_or_default(
            raw_transformer_config.get("audio_cross_attention_dim"),
            2048,
        ),
        positional_embedding_theta=float(
            raw_transformer_config.get("positional_embedding_theta", 10000.0)
        ),
        positional_embedding_max_pos=list(
            raw_transformer_config.get("positional_embedding_max_pos", [20, 2048, 2048])
        ),
        audio_positional_embedding_max_pos=list(
            raw_transformer_config.get("audio_positional_embedding_max_pos", [20])
        ),
        use_middle_indices_grid=bool(
            raw_transformer_config.get("use_middle_indices_grid", True)
        ),
        rope_type=_required_transformer_string(
            raw_transformer_config, checkpoint_path, "rope_type"
        ),
        double_precision_rope=(
            _required_transformer_string(
                raw_transformer_config, checkpoint_path, "frequencies_precision"
            ).lower()
            == "float64"
        ),
        timestep_scale_multiplier=int(
            raw_transformer_config.get("timestep_scale_multiplier", 1000)
        ),
        av_ca_timestep_scale_multiplier=int(
            raw_transformer_config.get(
                "av_ca_timestep_scale_multiplier",
                raw_transformer_config.get("timestep_scale_multiplier", 1000),
            )
        ),
        norm_eps=float(raw_transformer_config.get("norm_eps", 1e-6)),
        apply_gated_attention=apply_gated_attention,
        cross_attention_adaln=cross_attention_adaln,
        caption_proj_before_connector=_required_transformer_bool(
            raw_transformer_config, checkpoint_path, "caption_proj_before_connector"
        ),
    )


def _validate_reference_backend_compatibility(
    checkpoint_path: Path,
) -> None:
    runtime_config = _runtime_model_config(checkpoint_path)
    _runtime_vae_config(checkpoint_path)
    checkpoint_keys = _checkpoint_keys(checkpoint_path)
    required_stats = (
        "vae.per_channel_statistics.mean-of-means",
        "vae.per_channel_statistics.std-of-means",
    )
    if not all(key in checkpoint_keys for key in required_stats):
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing required VAE per-channel statistics"
        )
    if runtime_config.audio_enabled:
        required_audio_weights = (
            "model.diffusion_model.audio_patchify_proj.weight",
            "model.diffusion_model.audio_proj_out.weight",
            "model.diffusion_model.transformer_blocks.0.audio_attn1.to_q.weight",
            "model.diffusion_model.transformer_blocks.0.audio_attn2.to_q.weight",
            "model.diffusion_model.transformer_blocks.0.audio_to_video_attn.to_q.weight",
            "model.diffusion_model.transformer_blocks.0.video_to_audio_attn.to_q.weight",
        )
        if not all(key in checkpoint_keys for key in required_audio_weights):
            raise RuntimeError(
                f"LTX checkpoint '{checkpoint_path}' is missing required audio-video transformer weights"
            )
        if runtime_config.cross_attention_adaln:
            required_audio_prompt = (
                "model.diffusion_model.audio_prompt_adaln_single.linear.weight",
                "model.diffusion_model.transformer_blocks.0.audio_prompt_scale_shift_table",
            )
            if not all(key in checkpoint_keys for key in required_audio_prompt):
                raise RuntimeError(
                    f"LTX checkpoint '{checkpoint_path}' is missing required audio prompt AdaLN weights"
                )


def _assert_prompt_runtime_contract(
    prompt_context: PromptEncodingResult,
    runtime_config: _RuntimeModelConfig,
) -> None:
    if prompt_context.context_representation != "post_connector":
        raise ValueError(
            "LTX real generation requires post-connector prompt context for the "
            "current 22B bridge"
        )
    context_width = _context_width(prompt_context.video_context)
    if prompt_context.transformer_context_dim is not None:
        if prompt_context.transformer_context_dim != context_width:
            raise ValueError(
                "LTX prompt contract mismatch: prompt context width "
                f"{context_width} does not match declared transformer_context_dim "
                f"{prompt_context.transformer_context_dim}"
            )
        if prompt_context.transformer_context_dim != runtime_config.cross_attention_dim:
            raise ValueError(
                "LTX prompt/generation config mismatch: prompt transformer_context_dim "
                f"{prompt_context.transformer_context_dim} != runtime cross_attention_dim "
                f"{runtime_config.cross_attention_dim}"
            )
    if prompt_context.audio_context is None:
        raise ValueError(
            "LTX real generation requires audio_context from prompt_encode for the "
            "current 22B audio-video bridge"
        )
    if _attention_mask(prompt_context) is not None:
        raise ValueError(
            "LTX real generation requires an all-valid post-connector prompt mask "
            "for the current 22B audio-video bridge"
        )
    audio_context_width = _context_width(prompt_context.audio_context)
    if audio_context_width != runtime_config.audio_cross_attention_dim:
        raise ValueError(
            "LTX prompt/generation config mismatch: audio context width "
            f"{audio_context_width} != runtime audio_cross_attention_dim "
            f"{runtime_config.audio_cross_attention_dim}"
        )
    if (
        prompt_context.caption_proj_before_connector
        != runtime_config.caption_proj_before_connector
    ):
        raise ValueError(
            "LTX prompt/generation config mismatch for caption_proj_before_connector"
        )
    if prompt_context.rope_type != runtime_config.rope_type:
        raise ValueError("LTX prompt/generation config mismatch for rope_type")
    if prompt_context.double_precision_rope != runtime_config.double_precision_rope:
        raise ValueError(
            "LTX prompt/generation config mismatch for double_precision_rope"
        )
    if (
        prompt_context.transformer_apply_gated_attention is not None
        and prompt_context.transformer_apply_gated_attention
        != runtime_config.apply_gated_attention
    ):
        raise ValueError(
            "LTX prompt/generation config mismatch for apply_gated_attention"
        )
    if (
        prompt_context.transformer_cross_attention_adaln is not None
        and prompt_context.transformer_cross_attention_adaln
        != runtime_config.cross_attention_adaln
    ):
        raise ValueError(
            "LTX prompt/generation config mismatch for cross_attention_adaln"
        )


def _patch_reference_modules(imports: _ReferenceImports) -> None:
    if getattr(imports.model_class, "_mlxr_22b_patch", False):
        return

    original_get_video_config = imports.model_config_class.get_video_config
    original_get_audio_config = imports.model_config_class.get_audio_config
    original_attention_init = imports.attention_class.__init__
    original_preprocessor_init = imports.preprocessor_class.__init__
    original_multi_preprocessor_init = imports.multi_preprocessor_class.__init__
    original_prepare_context = imports.preprocessor_class._prepare_context
    original_model_init_video = imports.model_class._init_video
    original_model_init_audio = imports.model_class._init_audio
    original_model_init_transformer_blocks = (
        imports.model_class._init_transformer_blocks
    )
    original_block_init = imports.BasicAVTransformerBlock.__init__
    original_model_call = imports.model_class.__call__

    def patched_get_video_config(config_self: object) -> object:
        video_config = original_get_video_config(config_self)
        if video_config is None:
            return None
        setattr(
            video_config,
            "apply_gated_attention",
            bool(getattr(config_self, "apply_gated_attention", False)),
        )
        setattr(
            video_config,
            "cross_attention_adaln",
            bool(getattr(config_self, "cross_attention_adaln", False)),
        )
        return video_config

    def patched_get_audio_config(config_self: object) -> object:
        audio_config = original_get_audio_config(config_self)
        if audio_config is None:
            return None
        setattr(
            audio_config,
            "apply_gated_attention",
            bool(getattr(config_self, "apply_gated_attention", False)),
        )
        setattr(
            audio_config,
            "cross_attention_adaln",
            bool(getattr(config_self, "cross_attention_adaln", False)),
        )
        return audio_config

    def patched_attention_init(
        attention_self: object,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: object = None,
        apply_gated_attention: bool = False,
    ) -> None:
        if rope_type is None:
            rope_type = imports.rope_type_enum.INTERLEAVED
        original_attention_init(
            attention_self,
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
            rope_type=rope_type,
        )
        setattr(attention_self, "dim_head", dim_head)
        if apply_gated_attention:
            attention_self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)

    def patched_attention_call(
        attention_self: object,
        x: object,
        context: object | None = None,
        mask: object | None = None,
        pe: object | None = None,
        k_pe: object | None = None,
    ) -> object:
        query = attention_self.to_q(x)
        context = x if context is None else context
        key = attention_self.to_k(context)
        value = attention_self.to_v(context)
        query = attention_self.q_norm(query)
        key = attention_self.k_norm(key)
        if pe is not None:
            query = imports.apply_rotary_emb(query, pe, attention_self.rope_type)
            key = imports.apply_rotary_emb(
                key,
                pe if k_pe is None else k_pe,
                attention_self.rope_type,
            )
        out = imports.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_self.heads,
            mask,
        )
        to_gate_logits = getattr(attention_self, "to_gate_logits", None)
        if to_gate_logits is not None:
            gates = 2.0 * mx.sigmoid(to_gate_logits(x))
            batch_size, seq_len, _ = out.shape
            heads = int(attention_self.heads)
            dim_head = int(getattr(attention_self, "dim_head"))
            reshaped = mx.reshape(out, (batch_size, seq_len, heads, dim_head))
            out = mx.reshape(
                reshaped * mx.expand_dims(gates, axis=-1),
                (batch_size, seq_len, heads * dim_head),
            )
        return attention_self.to_out(out)

    def patched_preprocessor_init(
        preprocessor_self: object,
        patchify_proj: object,
        adaln: object,
        caption_projection: object | None,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        positional_embedding_theta: float,
        rope_type: object,
        double_precision_rope: bool = False,
        prompt_adaln: object | None = None,
    ) -> None:
        original_preprocessor_init(
            preprocessor_self,
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
        )
        preprocessor_self.prompt_adaln = prompt_adaln

    def patched_multi_preprocessor_init(
        preprocessor_self: object,
        patchify_proj: object,
        adaln: object,
        caption_projection: object | None,
        cross_scale_shift_adaln: object,
        cross_gate_adaln: object,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        positional_embedding_theta: float,
        rope_type: object,
        av_ca_timestep_scale_multiplier: int,
        double_precision_rope: bool = False,
        prompt_adaln: object | None = None,
    ) -> None:
        original_multi_preprocessor_init(
            preprocessor_self,
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            cross_scale_shift_adaln=cross_scale_shift_adaln,
            cross_gate_adaln=cross_gate_adaln,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            cross_pe_max_pos=cross_pe_max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            audio_cross_attention_dim=audio_cross_attention_dim,
            timestep_scale_multiplier=timestep_scale_multiplier,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
        )
        preprocessor_self.simple_preprocessor.prompt_adaln = prompt_adaln

    def patched_multi_preprocessor_prepare(
        preprocessor_self: object,
        modality: _PatchedModality,
        cross_modality: _PatchedModality | None = None,
    ) -> _PatchedTransformerArgs:
        from dataclasses import replace as dataclass_replace

        transformer_args = preprocessor_self.simple_preprocessor.prepare(modality)
        if cross_modality is None:
            return transformer_args

        if cross_modality.timesteps.shape[0] != modality.timesteps.shape[0]:
            raise ValueError(
                "Cross modality timesteps must have the same batch size as the modality"
            )

        cross_pe = preprocessor_self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=preprocessor_self.audio_cross_attention_dim,
            max_pos=[preprocessor_self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=(
                preprocessor_self.simple_preprocessor.num_attention_heads
            ),
        )
        cross_scale_shift_timestep, cross_gate_timestep = (
            preprocessor_self._prepare_cross_attention_timestep(
                timestep=modality.timesteps,
                timestep_scale_multiplier=(
                    preprocessor_self.simple_preprocessor.timestep_scale_multiplier
                ),
                batch_size=transformer_args.x.shape[0],
                hidden_dtype=transformer_args.x.dtype,
            )
        )

        return dataclass_replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def patched_prepare_context(
        preprocessor_self: object,
        context: object,
        x: object,
        attention_mask: object | None = None,
    ) -> tuple[object, object | None]:
        caption_projection = getattr(preprocessor_self, "caption_projection", None)
        if caption_projection is None:
            if context.ndim != 3 or int(context.shape[-1]) != int(x.shape[-1]):
                raise ValueError(
                    "LTX prompt context must already be post-connector and transformer-width "
                    f"when caption projection is disabled; got {tuple(int(size) for size in context.shape)} "
                    f"for transformer width {int(x.shape[-1])}"
                )
            return context, attention_mask
        return original_prepare_context(
            preprocessor_self,
            context=context,
            x=x,
            attention_mask=attention_mask,
        )

    def patched_prepare(
        preprocessor_self: object, modality: _PatchedModality
    ) -> _PatchedTransformerArgs:
        x = preprocessor_self.patchify_proj(modality.latent)
        timesteps, embedded_timestep = preprocessor_self._prepare_timestep(
            modality.timesteps,
            x.shape[0],
            hidden_dtype=x.dtype,
        )
        prompt_timestep = None
        prompt_adaln = getattr(preprocessor_self, "prompt_adaln", None)
        if prompt_adaln is not None:
            sigma_scaled = modality.sigma * preprocessor_self.timestep_scale_multiplier
            prompt_values, _ = prompt_adaln(
                mx.reshape(sigma_scaled, (-1,)),
                hidden_dtype=modality.latent.dtype,
            )
            prompt_timestep = mx.reshape(
                prompt_values, (x.shape[0], -1, prompt_values.shape[-1])
            )
        context, attention_mask = preprocessor_self._prepare_context(
            modality.context,
            x,
            modality.context_mask,
        )
        attention_mask = preprocessor_self._prepare_attention_mask(
            attention_mask,
            modality.latent.dtype,
        )
        positional_embeddings = (
            modality.positional_embeddings
            if modality.positional_embeddings is not None
            else preprocessor_self._prepare_positional_embeddings(
                positions=modality.positions,
                inner_dim=preprocessor_self.inner_dim,
                max_pos=preprocessor_self.max_pos,
                use_middle_indices_grid=preprocessor_self.use_middle_indices_grid,
                num_attention_heads=preprocessor_self.num_attention_heads,
            )
        )
        return _PatchedTransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timesteps,
            embedded_timestep=embedded_timestep,
            positional_embeddings=positional_embeddings,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
        )

    def patched_model_init_video(model_self: object, config: object) -> None:
        original_model_init_video(model_self, config)
        adaln_coefficient = (
            9 if bool(getattr(config, "cross_attention_adaln", False)) else 6
        )
        model_self.adaln_single = imports.adaln_class(
            model_self.inner_dim,
            embedding_coefficient=adaln_coefficient,
        )
        if bool(getattr(config, "caption_proj_before_connector", False)):
            if hasattr(model_self, "caption_projection"):
                delattr(model_self, "caption_projection")
        if bool(getattr(config, "cross_attention_adaln", False)):
            model_self.prompt_adaln_single = imports.adaln_class(
                model_self.inner_dim,
                embedding_coefficient=2,
            )

    def patched_model_init_audio(model_self: object, config: object) -> None:
        original_model_init_audio(model_self, config)
        adaln_coefficient = (
            9 if bool(getattr(config, "cross_attention_adaln", False)) else 6
        )
        model_self.audio_adaln_single = imports.adaln_class(
            model_self.audio_inner_dim,
            embedding_coefficient=adaln_coefficient,
        )
        if bool(getattr(config, "caption_proj_before_connector", False)):
            if hasattr(model_self, "audio_caption_projection"):
                delattr(model_self, "audio_caption_projection")
        if bool(getattr(config, "cross_attention_adaln", False)):
            model_self.audio_prompt_adaln_single = imports.adaln_class(
                model_self.audio_inner_dim,
                embedding_coefficient=2,
            )

    def patched_model_init_preprocessors(
        model_self: object,
        config: object,
        cross_pe_max_pos: object = None,
    ) -> None:
        if model_self.model_type.is_audio_enabled():
            effective_cross_pe_max_pos = cross_pe_max_pos
            if effective_cross_pe_max_pos is None:
                effective_cross_pe_max_pos = max(
                    model_self.positional_embedding_max_pos[0],
                    model_self.audio_positional_embedding_max_pos[0],
                )
            model_self.video_args_preprocessor = imports.multi_preprocessor_class(
                patchify_proj=model_self.patchify_proj,
                adaln=model_self.adaln_single,
                caption_projection=getattr(model_self, "caption_projection", None),
                cross_scale_shift_adaln=model_self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=model_self.av_ca_a2v_gate_adaln_single,
                inner_dim=model_self.inner_dim,
                max_pos=model_self.positional_embedding_max_pos,
                num_attention_heads=model_self.num_attention_heads,
                cross_pe_max_pos=effective_cross_pe_max_pos,
                use_middle_indices_grid=model_self.use_middle_indices_grid,
                audio_cross_attention_dim=model_self.audio_cross_attention_dim,
                timestep_scale_multiplier=model_self.timestep_scale_multiplier,
                positional_embedding_theta=model_self.positional_embedding_theta,
                rope_type=model_self.rope_type,
                av_ca_timestep_scale_multiplier=model_self.av_ca_timestep_scale_multiplier,
                double_precision_rope=model_self.config.double_precision_rope,
                prompt_adaln=getattr(model_self, "prompt_adaln_single", None),
            )
            model_self.audio_args_preprocessor = imports.multi_preprocessor_class(
                patchify_proj=model_self.audio_patchify_proj,
                adaln=model_self.audio_adaln_single,
                caption_projection=getattr(
                    model_self, "audio_caption_projection", None
                ),
                cross_scale_shift_adaln=model_self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=model_self.av_ca_v2a_gate_adaln_single,
                inner_dim=model_self.audio_inner_dim,
                max_pos=model_self.audio_positional_embedding_max_pos,
                num_attention_heads=model_self.audio_num_attention_heads,
                cross_pe_max_pos=effective_cross_pe_max_pos,
                use_middle_indices_grid=model_self.use_middle_indices_grid,
                audio_cross_attention_dim=model_self.audio_cross_attention_dim,
                timestep_scale_multiplier=model_self.timestep_scale_multiplier,
                positional_embedding_theta=model_self.positional_embedding_theta,
                rope_type=model_self.rope_type,
                av_ca_timestep_scale_multiplier=model_self.av_ca_timestep_scale_multiplier,
                double_precision_rope=model_self.config.double_precision_rope,
                prompt_adaln=getattr(model_self, "audio_prompt_adaln_single", None),
            )
            return
        model_self.video_args_preprocessor = imports.preprocessor_class(
            patchify_proj=model_self.patchify_proj,
            adaln=model_self.adaln_single,
            caption_projection=getattr(model_self, "caption_projection", None),
            inner_dim=model_self.inner_dim,
            max_pos=model_self.positional_embedding_max_pos,
            num_attention_heads=model_self.num_attention_heads,
            use_middle_indices_grid=model_self.use_middle_indices_grid,
            timestep_scale_multiplier=model_self.timestep_scale_multiplier,
            positional_embedding_theta=model_self.positional_embedding_theta,
            rope_type=model_self.rope_type,
            double_precision_rope=model_self.config.double_precision_rope,
            prompt_adaln=getattr(model_self, "prompt_adaln_single", None),
        )

    def patched_model_init_transformer_blocks(
        model_self: object, config: object
    ) -> None:
        if model_self.model_type.is_audio_enabled():
            original_model_init_transformer_blocks(model_self, config)
            return
        video_config = config.get_video_config()
        model_self.transformer_blocks = {
            index: imports.BasicAVTransformerBlock(
                idx=index,
                video=video_config,
                rope_type=config.rope_type,
                norm_eps=config.norm_eps,
            )
            for index in range(config.num_layers)
        }

    def patched_block_init(
        block_self: object,
        idx: int,
        video: object | None = None,
        audio: object | None = None,
        rope_type: object = None,
        norm_eps: float = 1e-6,
    ) -> None:
        if rope_type is None:
            rope_type = imports.rope_type_enum.INTERLEAVED
        original_block_init(
            block_self,
            idx=idx,
            video=video,
            audio=audio,
            rope_type=rope_type,
            norm_eps=norm_eps,
        )
        if video is not None and bool(getattr(video, "apply_gated_attention", False)):
            block_self.attn1.to_gate_logits = nn.Linear(
                video.dim, video.heads, bias=True
            )
            block_self.attn2.to_gate_logits = nn.Linear(
                video.dim, video.heads, bias=True
            )
            if hasattr(block_self, "audio_to_video_attn"):
                block_self.audio_to_video_attn.to_gate_logits = nn.Linear(
                    video.dim,
                    audio.heads if audio is not None else video.heads,
                    bias=True,
                )
        if video is not None and bool(getattr(video, "cross_attention_adaln", False)):
            block_self.scale_shift_table = mx.zeros((9, video.dim))
            block_self.prompt_scale_shift_table = mx.zeros((2, video.dim))
        if audio is not None and bool(getattr(audio, "apply_gated_attention", False)):
            block_self.audio_attn1.to_gate_logits = nn.Linear(
                audio.dim, audio.heads, bias=True
            )
            block_self.audio_attn2.to_gate_logits = nn.Linear(
                audio.dim, audio.heads, bias=True
            )
            if hasattr(block_self, "video_to_audio_attn"):
                block_self.video_to_audio_attn.to_gate_logits = nn.Linear(
                    audio.dim, audio.heads, bias=True
                )
        if audio is not None and bool(getattr(audio, "cross_attention_adaln", False)):
            block_self.audio_scale_shift_table = mx.zeros((9, audio.dim))
            block_self.audio_prompt_scale_shift_table = mx.zeros((2, audio.dim))

    def apply_cross_attention_adaln(
        *,
        block: object,
        x: object,
        context: object,
        attn: object,
        scale_shift_table: object,
        prompt_scale_shift_table: object,
        timestep: object,
        prompt_timestep: object | None,
        context_mask: object | None,
        norm_eps: float,
    ) -> object:
        if prompt_timestep is None:
            raise ValueError(
                "LTX prompt timestep is required for cross-attention AdaLN"
            )
        q_shift, q_scale, q_gate = block.get_ada_values(
            scale_shift_table, x.shape[0], timestep, slice(6, 9)
        )
        batch_size = x.shape[0]
        prompt_values = prompt_scale_shift_table[None, None] + mx.reshape(
            prompt_timestep,
            (batch_size, prompt_timestep.shape[1], 2, -1),
        )
        shift_kv = prompt_values[:, :, 0, :]
        scale_kv = prompt_values[:, :, 1, :]
        attn_input = imports.rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
        return (
            attn(
                attn_input,
                context=encoder_hidden_states,
                mask=context_mask,
            )
            * q_gate
        )

    def patched_block_call(
        block_self: object,
        video: _PatchedTransformerArgs | None = None,
        audio: _PatchedTransformerArgs | None = None,
    ) -> tuple[object | None, object | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None
        run_vx = video is not None and video.enabled and vx.size > 0
        run_ax = audio is not None and audio.enabled and ax.size > 0
        run_a2v = run_vx and run_ax
        run_v2a = run_ax and run_vx

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = block_self.get_ada_values(
                block_self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = (
                imports.rms_norm(vx, eps=block_self.norm_eps) * (1 + vscale_msa)
                + vshift_msa
            )
            vx = (
                vx
                + block_self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa
            )
            if hasattr(block_self, "prompt_scale_shift_table"):
                vx = vx + apply_cross_attention_adaln(
                    block=block_self,
                    x=vx,
                    context=video.context,
                    attn=block_self.attn2,
                    scale_shift_table=block_self.scale_shift_table,
                    prompt_scale_shift_table=block_self.prompt_scale_shift_table,
                    timestep=video.timesteps,
                    prompt_timestep=video.prompt_timestep,
                    context_mask=video.context_mask,
                    norm_eps=block_self.norm_eps,
                )
            else:
                vx = vx + block_self.attn2(
                    imports.rms_norm(vx, eps=block_self.norm_eps),
                    context=video.context,
                    mask=video.context_mask,
                )

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = block_self.get_ada_values(
                block_self.audio_scale_shift_table,
                ax.shape[0],
                audio.timesteps,
                slice(0, 3),
            )
            norm_ax = (
                imports.rms_norm(ax, eps=block_self.norm_eps) * (1 + ascale_msa)
                + ashift_msa
            )
            ax = (
                ax
                + block_self.audio_attn1(norm_ax, pe=audio.positional_embeddings)
                * agate_msa
            )
            if hasattr(block_self, "audio_prompt_scale_shift_table"):
                ax = ax + apply_cross_attention_adaln(
                    block=block_self,
                    x=ax,
                    context=audio.context,
                    attn=block_self.audio_attn2,
                    scale_shift_table=block_self.audio_scale_shift_table,
                    prompt_scale_shift_table=block_self.audio_prompt_scale_shift_table,
                    timestep=audio.timesteps,
                    prompt_timestep=audio.prompt_timestep,
                    context_mask=audio.context_mask,
                    norm_eps=block_self.norm_eps,
                )
            else:
                ax = ax + block_self.audio_attn2(
                    imports.rms_norm(ax, eps=block_self.norm_eps),
                    context=audio.context,
                    mask=audio.context_mask,
                )

        if run_a2v or run_v2a:
            vx_norm3 = imports.rms_norm(vx, eps=block_self.norm_eps)
            ax_norm3 = imports.rms_norm(ax, eps=block_self.norm_eps)
            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_out_v2a,
            ) = block_self.get_av_ca_ada_values(
                block_self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )
            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_out_a2v,
            ) = block_self.get_av_ca_ada_values(
                block_self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )
            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                vx = vx + (
                    block_self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                )
            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                ax = ax + (
                    block_self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                )

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = block_self.get_ada_values(
                block_self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = (
                imports.rms_norm(vx, eps=block_self.norm_eps) * (1 + vscale_mlp)
                + vshift_mlp
            )
            vx = vx + block_self.ff(vx_scaled) * vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = block_self.get_ada_values(
                block_self.audio_scale_shift_table,
                ax.shape[0],
                audio.timesteps,
                slice(3, 6),
            )
            ax_scaled = (
                imports.rms_norm(ax, eps=block_self.norm_eps) * (1 + ascale_mlp)
                + ashift_mlp
            )
            ax = ax + block_self.audio_ff(ax_scaled) * agate_mlp

        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )

    def patched_model_call(
        model_self: object,
        video: _PatchedModality | None = None,
        audio: _PatchedModality | None = None,
    ) -> tuple[object | None, object | None]:
        if not model_self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not model_self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")
        if (
            not model_self.model_type.is_audio_enabled()
            or not model_self.model_type.is_video_enabled()
        ):
            return original_model_call(model_self, video=video, audio=audio)
        video_args = (
            model_self.video_args_preprocessor.prepare(video, audio)
            if video is not None
            else None
        )
        audio_args = (
            model_self.audio_args_preprocessor.prepare(audio, video)
            if audio is not None
            else None
        )
        video_out, audio_out = model_self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
        )
        vx = (
            model_self._process_output(
                model_self.scale_shift_table,
                model_self.norm_out,
                model_self.proj_out,
                video_out.x,
                video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )
        ax = (
            model_self._process_output(
                model_self.audio_scale_shift_table,
                model_self.audio_norm_out,
                model_self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax

    imports.model_config_class.get_video_config = patched_get_video_config
    imports.model_config_class.get_audio_config = patched_get_audio_config
    imports.attention_class.__init__ = patched_attention_init
    imports.attention_class.__call__ = patched_attention_call
    imports.preprocessor_class.__init__ = patched_preprocessor_init
    imports.multi_preprocessor_class.__init__ = patched_multi_preprocessor_init
    imports.multi_preprocessor_class.prepare = patched_multi_preprocessor_prepare
    imports.preprocessor_class._prepare_context = patched_prepare_context
    imports.preprocessor_class.prepare = patched_prepare
    imports.model_class._init_video = patched_model_init_video
    imports.model_class._init_audio = patched_model_init_audio
    imports.model_class._init_preprocessors = patched_model_init_preprocessors
    imports.model_class._init_transformer_blocks = patched_model_init_transformer_blocks
    imports.model_class.__call__ = patched_model_call
    imports.BasicAVTransformerBlock.__init__ = patched_block_init
    imports.BasicAVTransformerBlock.__call__ = patched_block_call
    imports.model_class._mlxr_22b_patch = True


def _denoise_distilled_audio_video(
    *,
    imports: _ReferenceImports,
    transformer: object,
    latents: object,
    positions: object,
    text_embeddings: object,
    audio_latents: object,
    audio_positions: object,
    audio_embeddings: object,
    sigmas: tuple[float, ...],
    state: object | None,
    runtime_config: _RuntimeModelConfig,
) -> tuple[object, object]:
    latents_dtype = latents.dtype
    batch_size, channels, frames, latent_h, latent_w = latents.shape
    num_tokens = int(frames * latent_h * latent_w)
    audio_batch, audio_channels, audio_frames, audio_bins = audio_latents.shape
    precomputed_rope = imports.precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=runtime_config.double_precision_rope,
    )
    precomputed_audio_rope = imports.precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=runtime_config.double_precision_rope,
    )
    if state is not None:
        denoise_mask = mx.reshape(state.denoise_mask, (batch_size, 1, frames, 1, 1))
        denoise_mask = mx.broadcast_to(
            denoise_mask, (batch_size, 1, frames, latent_h, latent_w)
        )
        video_timesteps_mask = mx.reshape(
            denoise_mask, (batch_size, num_tokens)
        ).astype(latents_dtype)
    else:
        video_timesteps_mask = mx.ones((batch_size, num_tokens), dtype=latents_dtype)
    audio_timesteps_mask = mx.ones((audio_batch, audio_frames), dtype=latents_dtype)
    total_steps = max(len(sigmas) - 1, 0)
    for step_index, (sigma_value, sigma_next_value) in enumerate(
        zip(sigmas[:-1], sigmas[1:]),
        start=1,
    ):
        if _debug_progress_enabled():
            print(
                "[ltx] denoise "
                f"step={step_index}/{total_steps} "
                f"sigma={float(sigma_value):.6f} "
                f"audio_frames={audio_frames}",
                flush=True,
            )
        sigma = mx.array(float(sigma_value), dtype=latents_dtype)
        sigma_next = mx.array(float(sigma_next_value), dtype=latents_dtype)
        flat_latents = mx.transpose(
            mx.reshape(latents, (batch_size, channels, -1)), (0, 2, 1)
        )
        modality = _PatchedModality(
            latent=flat_latents,
            sigma=mx.full((batch_size,), float(sigma_value), dtype=latents_dtype),
            timesteps=sigma * video_timesteps_mask,
            positions=positions,
            context=text_embeddings,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_rope,
        )
        audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
        audio_flat = mx.reshape(
            audio_flat, (audio_batch, audio_frames, audio_channels * audio_bins)
        )
        audio_modality = _PatchedModality(
            latent=audio_flat,
            sigma=mx.full((audio_batch,), float(sigma_value), dtype=latents_dtype),
            timesteps=sigma * audio_timesteps_mask,
            positions=audio_positions,
            context=audio_embeddings,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_audio_rope,
        )
        velocity, audio_velocity = transformer(video=modality, audio=audio_modality)
        velocity = mx.reshape(
            mx.transpose(velocity, (0, 2, 1)),
            (batch_size, channels, frames, latent_h, latent_w),
        )
        denoised = imports.to_denoised(latents, velocity, sigma)
        audio_velocity = mx.reshape(
            audio_velocity, (audio_batch, audio_frames, audio_channels, audio_bins)
        )
        audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
        audio_denoised = imports.to_denoised(audio_latents, audio_velocity, sigma)
        if state is not None:
            denoised = imports.apply_denoise_mask(
                denoised, state.clean_latent, state.denoise_mask
            )
        if float(sigma_next_value) == 0.0:
            latents = denoised.astype(latents_dtype)
            audio_latents = audio_denoised.astype(latents_dtype)
        else:
            latents = (
                denoised.astype(mx.float32)
                + sigma_next.astype(mx.float32)
                * (latents.astype(mx.float32) - denoised.astype(mx.float32))
                / sigma.astype(mx.float32)
            ).astype(latents_dtype)
            audio_latents = (
                audio_denoised.astype(mx.float32)
                + sigma_next.astype(mx.float32)
                * (audio_latents.astype(mx.float32) - audio_denoised.astype(mx.float32))
                / sigma.astype(mx.float32)
            ).astype(latents_dtype)
        mx.eval(latents, audio_latents)
    return latents, audio_latents


def _require_video_context(prompt_context: PromptEncodingResult) -> object:
    if not _looks_like_mlx_array(prompt_context.video_context):
        raise ValueError("LTX real generation requires an MLX video prompt context")
    return prompt_context.video_context


def _require_audio_context(prompt_context: PromptEncodingResult) -> object:
    if not _looks_like_mlx_array(prompt_context.audio_context):
        raise ValueError("LTX real generation requires an MLX audio prompt context")
    return prompt_context.audio_context


def _attention_mask(prompt_context: PromptEncodingResult) -> object | None:
    if not _looks_like_mlx_array(prompt_context.attention_mask):
        return None
    if int(mx.sum(prompt_context.attention_mask).item()) == int(
        prompt_context.attention_mask.size
    ):
        return None
    return prompt_context.attention_mask


def _context_width(video_context: object) -> int:
    if not _looks_like_mlx_array(video_context):
        raise ValueError("LTX real generation requires an MLX video prompt context")
    return int(video_context.shape[-1])


def _looks_like_mlx_array(value: object) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _base_frames(
    *,
    width: int,
    height: int,
    num_frames: int,
    prompt_context: PromptEncodingResult,
    seed: int,
) -> mx.array:
    prompt_scale = max(prompt_context.token_count, 1) / max(
        prompt_context.sequence_length, 1
    )
    seed_scale = ((seed % 997) + 1) / 997.0
    width_grid = mx.linspace(0.0, 1.0, width).reshape(1, 1, width, 1)
    height_grid = mx.linspace(0.0, 1.0, height).reshape(1, height, 1, 1)
    frame_grid = mx.linspace(0.0, 1.0, num_frames).reshape(num_frames, 1, 1, 1)
    tau = math.tau
    red = 0.52 + 0.48 * mx.sin(
        tau
        * (
            width_grid * (2.8 + prompt_scale * 4.2)
            + height_grid * 0.7
            + frame_grid * (1.6 + seed_scale * 2.1)
        )
    )
    green = 0.48 + 0.45 * mx.cos(
        tau
        * (
            width_grid * 0.9
            + height_grid * (2.4 + prompt_scale * 3.6)
            + frame_grid * (1.2 + seed_scale * 1.4)
        )
    )
    blue = 0.51 + 0.44 * mx.sin(
        tau
        * (
            width_grid * (1.2 + seed_scale * 2.7)
            + height_grid * (1.5 + prompt_scale * 2.0)
            + frame_grid * 0.75
        )
    )
    base = mx.concatenate([red, green, blue], axis=-1)
    noise = mx.random.uniform(shape=base.shape, low=-0.08, high=0.08)
    return base + noise


def _apply_conditioning(
    *,
    frames: mx.array,
    conditioning_input: ConditioningInput,
    width: int,
    height: int,
    num_frames: int,
) -> mx.array:
    image = _decode_conditioning_image(
        payload_path=conditioning_input.payload_path,
        width=width,
        height=height,
    )
    weights = np.zeros((num_frames, 1, 1, 1), dtype=np.float32)
    for offset, factor in ((0, 1.0), (-1, 0.35), (1, 0.35)):
        target_index = conditioning_input.frame_index + offset
        if 0 <= target_index < num_frames:
            blended_strength = float(conditioning_input.strength) * factor
            weights[target_index, 0, 0, 0] = max(
                weights[target_index, 0, 0, 0], blended_strength
            )
    image_tensor = mx.array(image).reshape(1, height, width, 3)
    weight_tensor = mx.array(weights)
    return frames * (1.0 - weight_tensor) + image_tensor * weight_tensor


def _decode_conditioning_image(
    *, payload_path: Path, width: int, height: int
) -> npt.NDArray[np.float32]:
    with Image.open(payload_path) as image:
        rgb = image.convert("RGB")
        resized = rgb.resize((width, height), resample=Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=np.float32) / np.float32(255.0)

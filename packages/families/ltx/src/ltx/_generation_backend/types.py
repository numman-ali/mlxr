from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Protocol, TypeAlias

import mlx.core as mx
import numpy as np
import numpy.typing as npt

MLXArray: TypeAlias = mx.array


class _ImageLoader(Protocol):
    def __call__(
        self,
        path: str | Path,
        *,
        height: int | None = ...,
        width: int | None = ...,
        dtype: mx.Dtype = ...,
    ) -> MLXArray: ...


class _PreparedImageEncoder(Protocol):
    def __call__(
        self,
        image: MLXArray,
        target_height: int,
        target_width: int,
        *,
        dtype: mx.Dtype = ...,
    ) -> MLXArray: ...


class _VAEEncoder(Protocol):
    def parameters(self) -> object: ...

    def __call__(self, image: MLXArray) -> MLXArray: ...


class _GeneratorModule(Protocol):
    def parameters(self) -> object: ...

    def load_weights(
        self, weights: list[tuple[str, MLXArray]], *, strict: bool = ...
    ) -> None: ...


class _PrecomputeFreqsCis(Protocol):
    def __call__(
        self,
        positions: MLXArray,
        *,
        dim: int,
        theta: float,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        rope_type: str,
        double_precision: bool,
    ) -> tuple[MLXArray, MLXArray]: ...


class _VariadicFactory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _RotaryEmbeddingFn(Protocol):
    def __call__(
        self,
        tensor: MLXArray,
        pe: tuple[MLXArray, MLXArray],
        rope_type: object,
    ) -> MLXArray: ...


class _RmsNormFn(Protocol):
    def __call__(self, x: MLXArray, *, eps: float) -> MLXArray: ...


class _ScaledDotProductAttentionFn(Protocol):
    def __call__(
        self,
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        heads: int,
        mask: MLXArray | None,
    ) -> MLXArray: ...


class _ToDenoised(Protocol):
    def __call__(
        self, latents: MLXArray, velocity: MLXArray, sigma: MLXArray | float
    ) -> MLXArray: ...


class _ApplyDenoiseMask(Protocol):
    def __call__(
        self, denoised: MLXArray, clean_latent: MLXArray, denoise_mask: MLXArray
    ) -> MLXArray: ...


class _LatentStateLike(Protocol):
    latent: MLXArray
    clean_latent: MLXArray
    denoise_mask: MLXArray

    def clone(self) -> "_LatentStateLike": ...


class _AudioVideoTransformer(Protocol):
    inner_dim: int
    positional_embedding_theta: float
    positional_embedding_max_pos: list[int]
    use_middle_indices_grid: bool
    num_attention_heads: int
    rope_type: str
    audio_inner_dim: int
    audio_positional_embedding_max_pos: list[int]
    audio_num_attention_heads: int
    transformer_blocks: dict[object, object]
    adaln_single: object
    audio_adaln_single: object

    def __call__(
        self, *, video: _PatchedModality, audio: _PatchedModality
    ) -> tuple[MLXArray, MLXArray]: ...

    def parameters(self) -> object: ...


class _VideoDecoderLike(_GeneratorModule, Protocol):
    latents_mean: MLXArray
    latents_std: MLXArray

    def __call__(
        self,
        sample: MLXArray,
        *,
        causal: bool = False,
        timestep: MLXArray | None = None,
        debug: bool = False,
        chunked_conv: bool = False,
    ) -> MLXArray: ...

    def decode_tiled(
        self,
        sample: MLXArray,
        *,
        tiling_config: _TilingConfigInstance | None = None,
        tiling_mode: str = "auto",
        causal: bool = False,
        timestep: MLXArray | None = None,
        debug: bool = False,
        on_frames_ready: object | None = None,
    ) -> MLXArray: ...


class _UpsamplerLike(_GeneratorModule, Protocol):
    def __call__(self, latent: MLXArray, debug: bool = False) -> MLXArray: ...


class _AudioEncoderLike(_GeneratorModule, Protocol):
    per_channel_statistics: "_PerChannelStatistics"
    in_channels: int

    def __call__(self, mel: MLXArray) -> MLXArray: ...

    def load_weights(
        self, weights: list[tuple[str, MLXArray]], *, strict: bool = ...
    ) -> None: ...


class _AudioDecoderLike(_GeneratorModule, Protocol):
    per_channel_statistics: "_PerChannelStatistics"


class _AudioProcessorLike(Protocol):
    sample_rate: int

    def waveform_to_mel(
        self, waveform: npt.NDArray[np.float32], sample_rate: int
    ) -> npt.NDArray[np.float32]: ...


class _VocoderLike(_GeneratorModule, Protocol):
    pass


class _PerChannelStatistics(Protocol):
    _mean_of_means: MLXArray
    _std_of_means: MLXArray


class _ModelTypeEnumLike(Protocol):
    AudioVideo: object
    INTERLEAVED: object


class _TransformerConfigLike(Protocol):
    apply_gated_attention: bool
    cross_attention_adaln: bool
    caption_proj_before_connector: bool
    rope_type: object
    norm_eps: float
    num_layers: int

    def get_video_config(self) -> object | None: ...

    def get_audio_config(self) -> object | None: ...


class _ModelConfigFactory(Protocol):
    def from_dict(self, raw_config: Mapping[str, object]) -> _TransformerConfigLike: ...


class _ModelFactory(Protocol):
    _mlxr_22b_patch: bool

    def _init_video(self, config: object) -> None: ...

    def _init_audio(self, config: object) -> None: ...

    def _init_preprocessors(
        self, config: object, cross_pe_max_pos: object = ...
    ) -> None: ...

    def _init_transformer_blocks(self, config: object) -> None: ...

    def __call__(
        self,
        *,
        video: _PatchedModality | None = ...,
        audio: _PatchedModality | None = ...,
    ) -> tuple[object | None, object | None]: ...

    def from_pretrained(
        self,
        checkpoint_path: Path,
        *,
        config: _TransformerConfigLike,
        strict: bool,
    ) -> _AudioVideoTransformer: ...


class _LoadVAEEncoder(Protocol):
    def __call__(self, checkpoint_path: str) -> _VAEEncoder: ...


class _LoadAudioDecoder(Protocol):
    def __call__(
        self,
        checkpoint_root: Path,
        *,
        unified_weights: dict[str, MLXArray],
    ) -> _AudioDecoderLike: ...


class _AudioEncoderFactory(Protocol):
    def __call__(
        self,
        *,
        ch: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int,
        attn_resolutions: set[int],
        dropout: float,
        resamp_with_conv: bool,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool,
        norm_type: object,
        causality_axis: object,
        mid_block_add_attention: bool,
        sample_rate: int,
        mel_hop_length: int,
        n_fft: int,
        mel_bins: int,
        is_causal: bool,
    ) -> _AudioEncoderLike: ...


class _AudioDecoderFactory(Protocol):
    def __call__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int,
        attn_resolutions: set[int],
        resolution: int,
        z_channels: int,
        norm_type: object,
        causality_axis: object,
        mel_bins: int,
        mid_block_add_attention: bool,
        sample_rate: int,
        mel_hop_length: int,
        is_causal: bool,
    ) -> _AudioDecoderLike: ...


class _AudioProcessorFactory(Protocol):
    def __call__(
        self,
        *,
        sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> _AudioProcessorLike: ...


class _AudioEnumFactory(Protocol):
    def __call__(self, value: str) -> object: ...


class _SanitizeAudioVAEWeights(Protocol):
    def __call__(self, weights: dict[str, MLXArray]) -> dict[str, MLXArray]: ...


class _SanitizeVocoderWeights(Protocol):
    def __call__(self, weights: dict[str, MLXArray]) -> dict[str, MLXArray]: ...


class _DecodeAudio(Protocol):
    def __call__(
        self,
        audio_latents: MLXArray,
        audio_decoder: _AudioDecoderLike,
        vocoder: _VocoderLike,
    ) -> MLXArray: ...


class _TilingConfigLike(Protocol):
    @staticmethod
    def auto(
        height: int, width: int, num_frames: int
    ) -> _TilingConfigInstance | None: ...


class _SpatialTilingConfigLike(Protocol):
    tile_size_in_pixels: int


class _TemporalTilingConfigLike(Protocol):
    tile_size_in_frames: int


class _TilingConfigInstance(Protocol):
    spatial_config: _SpatialTilingConfigLike | None
    temporal_config: _TemporalTilingConfigLike | None


class _AdalnFactory(Protocol):
    def __call__(self, dims: int, embedding_coefficient: int = 6) -> object: ...


_UpsampleLatents: TypeAlias = Callable[
    [MLXArray, _UpsamplerLike, MLXArray, MLXArray], MLXArray
]


class _ConditionLike(Protocol):
    @property
    def latent(self) -> MLXArray: ...

    @property
    def frame_idx(self) -> int: ...

    @property
    def strength(self) -> float: ...


class _ConditionFactory(Protocol):
    def __call__(
        self, *, latent: MLXArray, frame_idx: int, strength: float
    ) -> _ConditionLike: ...


class _LatentStateFactory(Protocol):
    def __call__(
        self, *, latent: MLXArray, clean_latent: MLXArray, denoise_mask: MLXArray
    ) -> _LatentStateLike: ...


class _ApplyConditioning(Protocol):
    def __call__(
        self, state: _LatentStateLike, conditionings: list[_ConditionLike]
    ) -> _LatentStateLike: ...


class _CreatePositionGrid(Protocol):
    def __call__(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        *,
        temporal_scale: int = ...,
        spatial_scale: int = ...,
        fps: float = ...,
        causal_fix: bool = ...,
    ) -> MLXArray: ...


class _CreateAudioPositionGrid(Protocol):
    def __call__(
        self,
        batch_size: int,
        audio_frames: int,
        *,
        sample_rate: int = ...,
        hop_length: int = ...,
        downsample_factor: int = ...,
        is_causal: bool = ...,
    ) -> MLXArray: ...


class _ComputeAudioFrames(Protocol):
    def __call__(self, num_frames: int, fps: float) -> int: ...


class _RuntimeHelperHost(Protocol):
    checkpoint_path: Path
    spatial_upsampler_path: Path
    _reference_imports: _ReferenceImports | None
    _transformer: _AudioVideoTransformer | None
    _vae_decoder: _VideoDecoderLike | None
    _vae_encoder: _VAEEncoder | None
    _upsampler: _UpsamplerLike | None
    _audio_encoder: _AudioEncoderLike | None
    _audio_decoder: _AudioDecoderLike | None
    _audio_processor: _AudioProcessorLike | None
    _vocoder: _VocoderLike | None
    _audio_output_sample_rate: int | None
    _audio_backend: str | None


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
    model_class: _ModelFactory
    model_config_class: _ModelConfigFactory
    model_type_enum: _ModelTypeEnumLike
    rope_type_enum: _ModelTypeEnumLike
    BasicAVTransformerBlock: _VariadicFactory
    attention_class: object
    preprocessor_class: _VariadicFactory
    multi_preprocessor_class: _VariadicFactory
    feed_forward_class: object
    adaln_class: _AdalnFactory
    apply_rotary_emb: _RotaryEmbeddingFn
    precompute_freqs_cis: _PrecomputeFreqsCis
    rms_norm: _RmsNormFn
    to_denoised: _ToDenoised
    scaled_dot_product_attention: _ScaledDotProductAttentionFn
    latent_state_class: _LatentStateFactory
    tiling_config_class: _TilingConfigLike
    video_decoder_module: ModuleType
    condition_class: _ConditionFactory
    stage_1_sigmas: tuple[float, ...]
    stage_2_sigmas: tuple[float, ...]
    apply_conditioning: _ApplyConditioning
    apply_denoise_mask: _ApplyDenoiseMask
    create_position_grid: _CreatePositionGrid
    create_audio_position_grid: _CreateAudioPositionGrid
    compute_audio_frames: _ComputeAudioFrames
    load_image: _ImageLoader
    upsampler_module: ModuleType
    load_vae_encoder: _LoadVAEEncoder
    load_audio_decoder: _LoadAudioDecoder
    audio_encoder_class: _AudioEncoderFactory
    audio_decoder_class: _AudioDecoderFactory
    audio_processor_class: _AudioProcessorFactory
    audio_norm_type_enum: _AudioEnumFactory
    audio_causality_axis_enum: _AudioEnumFactory
    decode_audio: _DecodeAudio
    sanitize_audio_vae_weights: _SanitizeAudioVAEWeights
    sanitize_vocoder_weights: _SanitizeVocoderWeights
    audio_vocoder_class: object
    prepare_image_for_encoding: _PreparedImageEncoder
    upsample_latents: _UpsampleLatents
    audio_latent_channels: int
    audio_mel_bins: int
    audio_sample_rate: int


@dataclass(frozen=True, slots=True)
class _ConditioningPlan:
    stage1: tuple[_ConditionLike, ...]
    stage2: tuple[_ConditionLike, ...]


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
class _RuntimeVocoderArchitectureConfig:
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


@dataclass(frozen=True, slots=True)
class _RuntimeBWEConfig:
    generator: _RuntimeVocoderArchitectureConfig
    input_sample_rate: int
    output_sample_rate: int
    hop_length: int
    n_fft: int
    win_size: int
    num_mels: int


@dataclass(frozen=True, slots=True)
class _RuntimeVocoderConfig:
    vocoder: _RuntimeVocoderArchitectureConfig
    bwe: _RuntimeBWEConfig | None

    @property
    def uses_bwe(self) -> bool:
        return self.bwe is not None

    @property
    def output_sample_rate(self) -> int:
        if self.bwe is not None:
            return self.bwe.input_sample_rate
        return self.vocoder.output_sample_rate

    @property
    def bwe_output_sample_rate(self) -> int | None:
        if self.bwe is None:
            return None
        return self.bwe.output_sample_rate


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
class _RuntimeAudioEncoderConfig:
    base_channels: int
    ch_mult: tuple[int, ...]
    num_res_blocks: int
    attn_resolutions: set[int]
    resolution: int
    latent_channels: int
    dropout: float
    in_channels: int
    norm_type: str
    causality_axis: str
    mid_block_add_attention: bool
    sample_rate: int
    mel_hop_length: int
    mel_bins: int
    n_fft: int
    is_causal: bool
    double_z: bool


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

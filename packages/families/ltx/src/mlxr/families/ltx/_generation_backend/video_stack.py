from __future__ import annotations

from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from .. import _nn_compat as nn
from .._audio_bwe import AudioMelSTFT, AudioVocoderWithBWE
from .._audio_vocoder import AudioVocoder
from .audio_autoencoder import (
    AudioCausalityAxis,
    AudioDecoderModel,
    AudioNormKind,
)
from .config import (
    _decoder_initial_feature_channels,
    _first_present,
    _int_value,
    _runtime_audio_encoder_config,
    _runtime_vae_config,
    _runtime_vocoder_config,
    _validate_upsampler_layout,
)
from .spatial_upsampler import LatentUpsampler
from .types import (
    MLXArray,
    _AudioDecoderLike,
    _RuntimeVocoderArchitectureConfig,
    _SanitizeAudioVAEWeights,
    _SanitizeVocoderWeights,
    _TilingConfigInstance,
    _UpsamplerLike,
    _VAEEncoder,
    _VideoDecoderLike,
    _VocoderLike,
)
from .video_decoder_blocks import (
    CausalConv3d,
    DepthToSpaceUpsample,
    PaddingModeType,
    PixArtAlphaTimestepEmbedder,
    ResBlockGroup,
)
from .video_encoder import LatentLogVarianceType, VideoEncoder
from .video_ops import unpatchify_video
from .video_tiling import TilingConfig, decode_with_tiling
from .weight_loading import align_module_dtype_to_weights
from .weight_store import CheckpointWeightStore


class _WrappedCausalConv3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        spatial_padding_mode: PaddingModeType,
    ) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array, *, causal: bool = False) -> mx.array:
        result: mx.array = self.conv(x, causal=causal)
        return result


class _RuntimeVocoderWrapper:
    def __init__(self, impl: AudioVocoder | AudioVocoderWithBWE) -> None:
        self._impl = impl

    def parameters(self) -> object:
        return self._impl.parameters()

    def __call__(self, decoded_audio: MLXArray) -> MLXArray:
        return self._impl(decoded_audio)


def _upsample_latents(
    latent: MLXArray,
    upsampler: _UpsamplerLike,
    latent_mean: MLXArray,
    latent_std: MLXArray,
) -> MLXArray:
    mean = latent_mean.reshape(1, -1, 1, 1, 1)
    std = latent_std.reshape(1, -1, 1, 1, 1)
    unnormalized = latent * std + mean
    upsampled: MLXArray = upsampler(unnormalized)
    return (upsampled - mean) / std


class _ConfiguredVideoDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        decoder_blocks: tuple[tuple[str, object], ...],
        base_channels: int,
        spatial_padding_mode: PaddingModeType,
        timestep_conditioning: bool,
        causal_decoder: bool,
    ) -> None:
        super().__init__()
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
            in_channels=feature_channels,
            out_channels=final_out_channels,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.act = nn.SiLU()
        self._final_feature_channels = feature_channels

        if timestep_conditioning:
            self.timestep_scale_multiplier = mx.array(1000.0)
            self.last_time_embedder = PixArtAlphaTimestepEmbedder(
                embedding_dim=feature_channels * 2
            )
            self.last_scale_shift_table = mx.zeros((2, feature_channels))

    def _make_block(
        self,
        *,
        block_name: str,
        block_config: dict[str, object],
        in_channels: int,
        spatial_padding_mode: PaddingModeType,
    ) -> tuple[object, int]:
        if block_name == "res_x":
            num_layers = _int_value(
                block_config.get("num_layers", 1),
                context="Expected integer LTX decoder num_layers metadata",
            )
            return (
                ResBlockGroup(
                    in_channels,
                    num_layers,
                    spatial_padding_mode,
                    self.timestep_conditioning,
                ),
                in_channels,
            )

        reduction = _int_value(
            block_config.get("multiplier", 1),
            context="Expected integer LTX decoder multiplier metadata",
        )
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
            DepthToSpaceUpsample(
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
            if isinstance(block, ResBlockGroup):
                x = block(x, causal=effective_causal, timestep=scaled_timestep)
            elif isinstance(block, DepthToSpaceUpsample):
                x = block(x, causal=effective_causal, chunked_conv=chunked_conv)
            else:
                if not callable(block):
                    raise TypeError("LTX decoder block must be callable")
                block_result: MLXArray = block(x, causal=effective_causal)
                x = block_result

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
        unpatchified = unpatchify_video(
            x,
            patch_size_hw=self.patch_size,
            patch_size_t=1,
        )
        return unpatchified

    def decode_tiled(
        self,
        sample: mx.array,
        *,
        tiling_config: _TilingConfigInstance | None = None,
        tiling_mode: str = "auto",
        causal: bool = False,
        timestep: mx.array | None = None,
        debug: bool = False,
        on_frames_ready: object | None = None,
    ) -> mx.array:
        effective_causal = causal or self.causal_decoder
        resolved_tiling_config: _TilingConfigInstance = (
            TilingConfig.default() if tiling_config is None else tiling_config
        )
        callback: Callable[[MLXArray, int], None] | None = (
            on_frames_ready if callable(on_frames_ready) else None
        )

        _, _, frames, latent_h, latent_w = sample.shape
        needs_spatial_tiling = False
        needs_temporal_tiling = False
        spatial_scale = 32
        temporal_scale = 8

        spatial_config = resolved_tiling_config.spatial_config
        if spatial_config is not None:
            tile_size_latent = spatial_config.tile_size_in_pixels // spatial_scale
            if latent_h > tile_size_latent or latent_w > tile_size_latent:
                needs_spatial_tiling = True

        temporal_config = resolved_tiling_config.temporal_config
        if temporal_config is not None:
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
            if callback is not None:
                try:
                    callback(decoded, 0)
                except Exception:
                    return decoded
            return decoded

        decoded_tiled = decode_with_tiling(
            decoder_fn=self,
            latents=sample,
            tiling_config=resolved_tiling_config,
            spatial_scale=32,
            temporal_scale=8,
            causal=effective_causal,
            timestep=timestep,
            chunked_conv=use_chunked_conv,
            on_frames_ready=callback,
        )
        return decoded_tiled


def _require_weight_mapping(weights: object, *, context: str) -> dict[str, MLXArray]:
    if not isinstance(weights, dict):
        raise RuntimeError(context)
    return {
        key: value
        for key, value in weights.items()
        if isinstance(key, str) and isinstance(value, mx.array)
    }


def _require_array(value: object, *, context: str) -> MLXArray:
    if not isinstance(value, mx.array):
        raise RuntimeError(context)
    return value


def _trainable_parameter_keys(module: object, *, context: str) -> set[str]:
    trainable_parameters = getattr(module, "trainable_parameters", None)
    if not callable(trainable_parameters):
        raise RuntimeError(context)

    def _collect_keys(tree: object, prefix: str = "") -> set[str]:
        if isinstance(tree, dict):
            collected: set[str] = set()
            for key, value in tree.items():
                if not isinstance(key, (str, int)):
                    raise RuntimeError(context)
                key_text = str(key)
                child_prefix = f"{prefix}.{key_text}" if prefix else key_text
                collected.update(_collect_keys(value, child_prefix))
            return collected
        if isinstance(tree, (list, tuple)):
            collected = set()
            for index, value in enumerate(tree):
                child_prefix = f"{prefix}.{index}" if prefix else str(index)
                collected.update(_collect_keys(value, child_prefix))
            return collected
        return {prefix} if prefix else set()

    keys = _collect_keys(trainable_parameters())
    if not keys:
        raise RuntimeError(context)
    return keys


def _require_weight_subset(
    weights: dict[str, MLXArray],
    *,
    expected_keys: set[str],
    context: str,
) -> dict[str, MLXArray]:
    missing = sorted(expected_keys - weights.keys())
    if missing:
        raise RuntimeError(
            f"{context} is missing required weights: {', '.join(missing[:5])}"
        )
    unexpected = sorted(weights.keys() - expected_keys)
    if unexpected:
        raise RuntimeError(
            f"{context} contains unexpected weights: {', '.join(unexpected[:5])}"
        )
    return {key: weights[key] for key in sorted(expected_keys)}


def _require_per_channel_statistics(
    weights: dict[str, MLXArray],
    *,
    expected_width: int,
    context: str,
) -> tuple[MLXArray, MLXArray]:
    mean = weights.get("per_channel_statistics._mean_of_means")
    std = weights.get("per_channel_statistics._std_of_means")
    if mean is None or std is None:
        raise RuntimeError(f"{context} is missing per-channel statistics")
    mean_array = _require_array(
        mean,
        context=f"{context} has invalid per-channel mean statistics",
    )
    std_array = _require_array(
        std,
        context=f"{context} has invalid per-channel std statistics",
    )
    expected_shape = (expected_width,)
    if tuple(int(size) for size in mean_array.shape) != expected_shape:
        raise RuntimeError(
            f"{context} has invalid per-channel mean statistics shape "
            f"{tuple(int(size) for size in mean_array.shape)!r}; expected {expected_shape!r}"
        )
    if tuple(int(size) for size in std_array.shape) != expected_shape:
        raise RuntimeError(
            f"{context} has invalid per-channel std statistics shape "
            f"{tuple(int(size) for size in std_array.shape)!r}; expected {expected_shape!r}"
        )
    std_values = np.asarray(std_array)
    if not np.isfinite(std_values).all() or np.any(std_values <= 0):
        raise RuntimeError(f"{context} has invalid per-channel std statistics values")
    mean_values = np.asarray(mean_array)
    if not np.isfinite(mean_values).all():
        raise RuntimeError(f"{context} has invalid per-channel mean statistics values")
    return mean_array, std_array


def _validate_bwe_stft_buffers(
    *,
    checkpoint_path: Path,
    mel_basis: MLXArray,
    forward_basis: MLXArray,
    inverse_basis: MLXArray,
    num_mels: int,
    n_fft: int,
) -> None:
    expected_mel_shape = (num_mels, n_fft // 2 + 1)
    mel_shape = tuple(int(size) for size in mel_basis.shape)
    if mel_shape != expected_mel_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid mel basis shape "
            f"{mel_shape!r}; expected {expected_mel_shape!r}"
        )

    expected_stft_shape = (n_fft + 2, 1, n_fft)
    forward_shape = tuple(int(size) for size in forward_basis.shape)
    if forward_shape != expected_stft_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid STFT forward basis shape "
            f"{forward_shape!r}; expected {expected_stft_shape!r}"
        )

    inverse_shape = tuple(int(size) for size in inverse_basis.shape)
    if inverse_shape != expected_stft_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid STFT inverse basis shape "
            f"{inverse_shape!r}; expected {expected_stft_shape!r}"
        )


def _load_configured_vae_decoder(
    checkpoint_path: Path,
    *,
    weight_store: CheckpointWeightStore | None = None,
) -> _VideoDecoderLike:
    vae_config = _runtime_vae_config(checkpoint_path)

    spatial_padding_mode = PaddingModeType(vae_config.spatial_padding_mode)
    decoder = _ConfiguredVideoDecoder(
        in_channels=vae_config.latent_channels,
        out_channels=vae_config.out_channels,
        patch_size=vae_config.patch_size,
        decoder_blocks=vae_config.decoder_blocks,
        base_channels=vae_config.base_channels,
        spatial_padding_mode=spatial_padding_mode,
        timestep_conditioning=vae_config.timestep_conditioning,
        causal_decoder=vae_config.causal_decoder,
    )

    weights = _require_weight_mapping(
        weight_store.all_weights()
        if weight_store is not None
        else mx.load(str(checkpoint_path)),
        context=f"LTX checkpoint '{checkpoint_path}' did not load into a decoder weight mapping",
    )
    decoder_weights: dict[str, MLXArray] = {}
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
    mean_array = _require_array(
        mean,
        context=f"LTX checkpoint '{checkpoint_path}' has invalid latents_mean weights",
    )
    std_array = _require_array(
        std,
        context=f"LTX checkpoint '{checkpoint_path}' has invalid latents_std weights",
    )
    expected_shape = (vae_config.latent_channels,)
    if tuple(int(size) for size in mean_array.shape) != expected_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid latents_mean shape "
            f"{tuple(int(size) for size in mean_array.shape)!r}; expected {expected_shape!r}"
        )
    if tuple(int(size) for size in std_array.shape) != expected_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid latents_std shape "
            f"{tuple(int(size) for size in std_array.shape)!r}; expected {expected_shape!r}"
        )
    decoder_weights["latents_mean"] = mean_array
    decoder_weights["latents_std"] = std_array
    align_module_dtype_to_weights(
        decoder,
        {
            key: value
            for key, value in decoder_weights.items()
            if key not in {"latents_mean", "latents_std"}
        },
        context="Owned LTX video decoder weights",
    )
    decoder.load_weights(list(decoder_weights.items()), strict=True)
    return decoder


def _load_runtime_vae_statistics(
    checkpoint_path: Path,
    *,
    weight_store: CheckpointWeightStore | None = None,
) -> tuple[MLXArray, MLXArray]:
    vae_config = _runtime_vae_config(checkpoint_path)
    weights = _require_weight_mapping(
        weight_store.all_weights()
        if weight_store is not None
        else mx.load(str(checkpoint_path)),
        context=f"LTX checkpoint '{checkpoint_path}' did not load into a VAE statistics weight mapping",
    )
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
    mean_array = _require_array(
        mean,
        context=f"LTX checkpoint '{checkpoint_path}' has invalid latents_mean weights",
    )
    std_array = _require_array(
        std,
        context=f"LTX checkpoint '{checkpoint_path}' has invalid latents_std weights",
    )
    expected_shape = (vae_config.latent_channels,)
    if tuple(int(size) for size in mean_array.shape) != expected_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid latents_mean shape "
            f"{tuple(int(size) for size in mean_array.shape)!r}; expected {expected_shape!r}"
        )
    if tuple(int(size) for size in std_array.shape) != expected_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid latents_std shape "
            f"{tuple(int(size) for size in std_array.shape)!r}; expected {expected_shape!r}"
        )
    return mean_array, std_array


def _load_configured_upsampler(
    weights_path: Path,
) -> _UpsamplerLike:
    _validate_upsampler_layout(weights_path)
    raw_weights = _require_weight_mapping(
        mx.load(str(weights_path)),
        context=f"LTX spatial upsampler '{weights_path}' did not load into a weight mapping",
    )
    sample_key = "res_blocks.0.conv1.weight"
    mid_channels = (
        int(raw_weights[sample_key].shape[0]) if sample_key in raw_weights else 1024
    )
    upsampler: _UpsamplerLike = LatentUpsampler(
        in_channels=128,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
    )

    sanitized: dict[str, MLXArray] = {}
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

    align_module_dtype_to_weights(
        upsampler,
        sanitized,
        context="Owned LTX spatial upsampler weights",
    )
    upsampler.load_weights(list(sanitized.items()), strict=False)
    return upsampler


def _load_runtime_vae_encoder(
    checkpoint_path: Path,
    *,
    weight_store: CheckpointWeightStore | None = None,
) -> _VAEEncoder:
    vae_config = _runtime_vae_config(checkpoint_path)
    padding_mode = PaddingModeType(vae_config.encoder_spatial_padding_mode)
    latent_log_var = LatentLogVarianceType(vae_config.latent_log_var)
    encoder = VideoEncoder(
        in_channels=vae_config.in_channels,
        out_channels=vae_config.latent_channels,
        encoder_blocks=list(vae_config.encoder_blocks),
        patch_size=vae_config.patch_size,
        latent_log_var=latent_log_var,
        encoder_spatial_padding_mode=padding_mode,
    )

    weights = _require_weight_mapping(
        weight_store.all_weights()
        if weight_store is not None
        else mx.load(str(checkpoint_path)),
        context=f"LTX checkpoint '{checkpoint_path}' did not load into an encoder weight mapping",
    )
    encoder_weights: dict[str, MLXArray] = {}
    for key, value in weights.items():
        if key.startswith("vae.encoder."):
            new_key = key[len("vae.encoder.") :]
        elif key.startswith("vae_encoder."):
            new_key = key[len("vae_encoder.") :]
        elif key.startswith("encoder."):
            new_key = key[len("encoder.") :]
        else:
            continue
        if value.ndim == 5 and "conv" in new_key and "weight" in new_key:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        encoder_weights[new_key] = value

    mean = _first_present(
        weights,
        (
            "vae.per_channel_statistics.mean-of-means",
            "vae.per_channel_statistics.mean",
            "per_channel_statistics.mean-of-means",
            "per_channel_statistics.mean",
        ),
    )
    std = _first_present(
        weights,
        (
            "vae.per_channel_statistics.std-of-means",
            "vae.per_channel_statistics.std",
            "per_channel_statistics.std-of-means",
            "per_channel_statistics.std",
        ),
    )
    if mean is None or std is None:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing VAE encoder per-channel statistics"
        )
    mean_array = _require_array(
        mean,
        context=f"LTX checkpoint '{checkpoint_path}' has invalid VAE encoder mean statistics",
    )
    std_array = _require_array(
        std,
        context=f"LTX checkpoint '{checkpoint_path}' has invalid VAE encoder std statistics",
    )
    expected_shape = (vae_config.latent_channels,)
    if tuple(int(size) for size in mean_array.shape) != expected_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid VAE encoder mean statistics shape "
            f"{tuple(int(size) for size in mean_array.shape)!r}; expected {expected_shape!r}"
        )
    if tuple(int(size) for size in std_array.shape) != expected_shape:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' has invalid VAE encoder std statistics shape "
            f"{tuple(int(size) for size in std_array.shape)!r}; expected {expected_shape!r}"
        )
    if not encoder_weights:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' is missing VAE encoder weights"
        )
    align_module_dtype_to_weights(
        encoder,
        encoder_weights,
        context="Owned LTX video encoder weights",
    )
    encoder.load_weights(list(encoder_weights.items()), strict=False)
    encoder.per_channel_statistics._mean_of_means = mean_array
    encoder.per_channel_statistics._std_of_means = std_array
    return encoder


def _load_runtime_audio_decoder(
    *,
    checkpoint_root: Path,
    sanitize_audio_vae_weights: _SanitizeAudioVAEWeights,
    unified_weights: dict[str, MLXArray],
) -> _AudioDecoderLike:
    audio_config = _runtime_audio_encoder_config(checkpoint_root)
    norm_kind = AudioNormKind.from_config_value(audio_config.norm_type)
    causality_axis = AudioCausalityAxis.from_config_value(audio_config.causality_axis)

    decoder = AudioDecoderModel(
        base_channels=audio_config.base_channels,
        out_channels=audio_config.in_channels,
        channel_multipliers=audio_config.ch_mult,
        num_res_blocks=audio_config.num_res_blocks,
        attn_resolutions=audio_config.attn_resolutions,
        resolution=audio_config.resolution,
        latent_channels=audio_config.latent_channels,
        norm_kind=norm_kind,
        causality_axis=causality_axis,
        mel_bins=audio_config.mel_bins,
        dropout=audio_config.dropout,
        sample_rate=audio_config.sample_rate,
        mel_hop_length=audio_config.mel_hop_length,
        mid_block_add_attention=audio_config.mid_block_add_attention,
        is_causal=audio_config.is_causal,
    )

    sanitized = sanitize_audio_vae_weights(unified_weights)
    decoder_weights = {
        key.replace("decoder.", ""): value
        for key, value in sanitized.items()
        if key.startswith("decoder.")
    }
    expected_keys = _trainable_parameter_keys(
        decoder,
        context="Expected owned LTX audio decoder trainable parameter contract",
    )
    required_decoder_weights = _require_weight_subset(
        decoder_weights,
        expected_keys=expected_keys,
        context="Owned LTX audio decoder weights",
    )
    align_module_dtype_to_weights(
        decoder,
        required_decoder_weights,
        context="Owned LTX audio decoder weights",
    )
    decoder.load_weights(list(required_decoder_weights.items()), strict=True)
    mean_array, std_array = _require_per_channel_statistics(
        sanitized,
        expected_width=int(decoder.per_channel_statistics._mean_of_means.shape[0]),
        context="Owned LTX audio decoder per-channel statistics",
    )
    decoder.per_channel_statistics._mean_of_means = mean_array
    decoder.per_channel_statistics._std_of_means = std_array
    return decoder


def _load_runtime_vocoder(
    *,
    checkpoint_path: Path,
    checkpoint_weights: dict[str, mx.array],
    sanitize_vocoder_weights: _SanitizeVocoderWeights,
) -> tuple[_VocoderLike, int, str]:
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

    def _build_vocoder(
        *,
        architecture: _RuntimeVocoderArchitectureConfig,
        raw_weights: dict[str, mx.array],
    ) -> AudioVocoder:
        sanitized_weights = sanitize_vocoder_weights(raw_weights)
        sanitized_weights = {
            key: value
            for key, value in sanitized_weights.items()
            if not key.endswith(".filter")
        }
        vocoder = AudioVocoder(
            resblock_kernel_sizes=list(architecture.resblock_kernel_sizes),
            upsample_rates=list(architecture.upsample_rates),
            upsample_kernel_sizes=list(architecture.upsample_kernel_sizes),
            resblock_dilation_sizes=[
                list(block) for block in architecture.resblock_dilation_sizes
            ],
            upsample_initial_channel=architecture.upsample_initial_channel,
            stereo=architecture.stereo,
            resblock=architecture.resblock,
            output_sample_rate=architecture.output_sample_rate,
            activation=architecture.activation,
            use_tanh_at_final=architecture.use_tanh_at_final,
            apply_final_activation=architecture.apply_final_activation,
            use_bias_at_final=architecture.use_bias_at_final,
        )
        vocoder.load_weights(list(sanitized_weights.items()), strict=False)
        return vocoder

    base_vocoder = _build_vocoder(
        architecture=runtime_vocoder_config.vocoder,
        raw_weights=raw_base_weights,
    )
    if runtime_vocoder_config.bwe is None:
        runtime_vocoder: _VocoderLike = _RuntimeVocoderWrapper(base_vocoder)
        return (
            runtime_vocoder,
            runtime_vocoder_config.output_sample_rate,
            "mlx_vocoder",
        )

    raw_bwe_weights = {
        key[len("vocoder.bwe_generator.") :]: value
        for key, value in checkpoint_weights.items()
        if key.startswith("vocoder.bwe_generator.")
    }
    if not raw_bwe_weights:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' declares BWE support but is missing BWE generator weights"
        )

    mel_basis = checkpoint_weights.get("vocoder.mel_stft.mel_basis")
    forward_basis = checkpoint_weights.get("vocoder.mel_stft.stft_fn.forward_basis")
    inverse_basis = checkpoint_weights.get("vocoder.mel_stft.stft_fn.inverse_basis")
    if mel_basis is None or forward_basis is None or inverse_basis is None:
        raise RuntimeError(
            f"LTX checkpoint '{checkpoint_path}' declares BWE support but is missing mel STFT buffers"
        )
    _validate_bwe_stft_buffers(
        checkpoint_path=checkpoint_path,
        mel_basis=mel_basis,
        forward_basis=forward_basis,
        inverse_basis=inverse_basis,
        num_mels=runtime_vocoder_config.bwe.num_mels,
        n_fft=runtime_vocoder_config.bwe.n_fft,
    )

    bwe_generator = _build_vocoder(
        architecture=runtime_vocoder_config.bwe.generator,
        raw_weights=raw_bwe_weights,
    )
    mel_stft = AudioMelSTFT(
        filter_length=runtime_vocoder_config.bwe.n_fft,
        hop_length=runtime_vocoder_config.bwe.hop_length,
        win_length=runtime_vocoder_config.bwe.win_size,
        n_mel_channels=runtime_vocoder_config.bwe.num_mels,
    )
    mel_stft.mel_basis = mel_basis.astype(mx.float32)
    mel_stft.stft_fn.forward_basis = mx.transpose(
        forward_basis.astype(mx.float32), (0, 2, 1)
    )
    mel_stft.stft_fn.inverse_basis = mx.transpose(
        inverse_basis.astype(mx.float32), (0, 2, 1)
    )
    vocoder_with_bwe = AudioVocoderWithBWE(
        vocoder=base_vocoder,
        bwe_generator=bwe_generator,
        mel_stft=mel_stft,
        input_sample_rate=runtime_vocoder_config.bwe.input_sample_rate,
        output_sample_rate=runtime_vocoder_config.bwe.output_sample_rate,
        hop_length=runtime_vocoder_config.bwe.hop_length,
    )
    runtime_vocoder_with_bwe: _VocoderLike = _RuntimeVocoderWrapper(vocoder_with_bwe)
    return (
        runtime_vocoder_with_bwe,
        runtime_vocoder_config.bwe.output_sample_rate,
        "mlxr_vocoder_with_bwe",
    )

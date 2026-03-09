from __future__ import annotations

from enum import Enum
from typing import Protocol

import mlx.core as mx

from .. import _nn_compat as nn
from .types import MLXArray, _PerChannelStatistics, _VAEEncoder
from .video_decoder_blocks import CausalConv3d, PaddingModeType, ResBlockGroup


class _EncoderBlock(Protocol):
    def __call__(self, x: MLXArray, *, causal: bool = True) -> MLXArray: ...


class LatentLogVarianceType(Enum):
    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


class PerChannelStatistics(nn.Module):
    _mean_of_means: MLXArray
    _std_of_means: MLXArray

    def __init__(self, *, latent_channels: int) -> None:
        super().__init__()
        self._mean_of_means: MLXArray = mx.zeros((latent_channels,))
        self._std_of_means: MLXArray = mx.ones((latent_channels,))

    def normalize(self, x: MLXArray) -> MLXArray:
        dtype = x.dtype
        mean = self._mean_of_means.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        std = self._std_of_means.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        return ((x - mean) / std).astype(dtype)


def patchify_video(
    x: MLXArray,
    *,
    patch_size_hw: int = 4,
    patch_size_t: int = 1,
) -> MLXArray:
    batch, channels, frames, height, width = x.shape
    if height % patch_size_hw != 0 or width % patch_size_hw != 0:
        raise ValueError("Input spatial dimensions must be divisible by patch size")
    if frames % patch_size_t != 0:
        raise ValueError("Input frame count must be divisible by temporal patch size")

    new_height = height // patch_size_hw
    new_width = width // patch_size_hw
    new_frames = frames // patch_size_t
    new_channels = channels * patch_size_hw * patch_size_hw * patch_size_t

    patched = mx.reshape(
        x,
        (
            batch,
            channels,
            new_frames,
            patch_size_t,
            new_height,
            patch_size_hw,
            new_width,
            patch_size_hw,
        ),
    )
    patched = mx.transpose(patched, (0, 1, 3, 7, 5, 2, 4, 6))
    return mx.reshape(
        patched,
        (batch, new_channels, new_frames, new_height, new_width),
    )


class _ConvWrapper(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: MLXArray, *, causal: bool = True) -> MLXArray:
        return self.conv(x, causal=causal)


class SpaceToDepthDownsample(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int],
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride
        self.out_channels = out_channels
        multiplier = stride[0] * stride[1] * stride[2]
        self.group_size = in_channels * multiplier // out_channels
        conv_out_channels = out_channels // multiplier
        self.conv = _ConvWrapper(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _space_to_depth(self, x: MLXArray) -> MLXArray:
        batch, channels, frames, height, width = x.shape
        stride_t, stride_h, stride_w = self.stride
        reshaped = mx.reshape(
            x,
            (
                batch,
                channels,
                frames // stride_t,
                stride_t,
                height // stride_h,
                stride_h,
                width // stride_w,
                stride_w,
            ),
        )
        transposed = mx.transpose(reshaped, (0, 1, 3, 5, 7, 2, 4, 6))
        new_channels = channels * stride_t * stride_h * stride_w
        return mx.reshape(
            transposed,
            (
                batch,
                new_channels,
                frames // stride_t,
                height // stride_h,
                width // stride_w,
            ),
        )

    def __call__(self, x: MLXArray, *, causal: bool = True) -> MLXArray:
        del causal
        _, channels, frames, height, width = x.shape
        stride_t, stride_h, stride_w = self.stride

        padded = x
        if stride_t == 2:
            padded = mx.concatenate([padded[:, :, :1, :, :], padded], axis=2)
            frames += 1

        pad_frames = (stride_t - (frames % stride_t)) % stride_t
        pad_height = (stride_h - (height % stride_h)) % stride_h
        pad_width = (stride_w - (width % stride_w)) % stride_w
        if pad_frames or pad_height or pad_width:
            padded = mx.pad(
                padded,
                [(0, 0), (0, 0), (0, pad_frames), (0, pad_height), (0, pad_width)],
            )

        residual = self._space_to_depth(padded)
        batch, _, out_frames, out_height, out_width = residual.shape
        residual = mx.reshape(
            residual,
            (
                batch,
                self.out_channels,
                self.group_size,
                out_frames,
                out_height,
                out_width,
            ),
        )
        residual = mx.mean(residual, axis=2)

        convolved = self.conv(padded)
        convolved = self._space_to_depth(convolved)
        return convolved + residual


class VideoEncoder(nn.Module, _VAEEncoder):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        encoder_blocks: list[tuple[str, object]],
        patch_size: int,
        latent_log_var: LatentLogVarianceType,
        encoder_spatial_padding_mode: PaddingModeType,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._per_channel_statistics = PerChannelStatistics(
            latent_channels=out_channels
        )

        feature_channels = out_channels
        patched_in_channels = in_channels * patch_size * patch_size
        self.conv_in = _ConvWrapper(
            in_channels=patched_in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks: list[_EncoderBlock] = []
        for index, (block_name, raw_params) in enumerate(encoder_blocks):
            params = (
                raw_params
                if isinstance(raw_params, dict)
                else {"num_layers": raw_params}
            )
            block, feature_channels = self._make_block(
                block_name=block_name,
                block_config=params,
                in_channels=feature_channels,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )
            del index
            self.down_blocks.append(block)

        self.conv_act = nn.SiLU()
        conv_out_channels = out_channels
        if latent_log_var == LatentLogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in (
            LatentLogVarianceType.UNIFORM,
            LatentLogVarianceType.CONSTANT,
        ):
            conv_out_channels += 1

        self.conv_out = _ConvWrapper(
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def _pixel_norm(self, x: MLXArray, eps: float = 1e-8) -> MLXArray:
        return x / mx.sqrt(mx.mean(x**2, axis=1, keepdims=True) + eps)

    def _make_block(
        self,
        *,
        block_name: str,
        block_config: dict[str, object],
        in_channels: int,
        spatial_padding_mode: PaddingModeType,
    ) -> tuple[_EncoderBlock, int]:
        if block_name == "res_x":
            num_layers_value = block_config.get("num_layers", 1)
            if not isinstance(num_layers_value, int) or isinstance(
                num_layers_value, bool
            ):
                raise ValueError("Expected integer LTX encoder num_layers metadata")
            return (
                ResBlockGroup(
                    channels=in_channels,
                    num_layers=num_layers_value,
                    spatial_padding_mode=spatial_padding_mode,
                    timestep_conditioning=False,
                ),
                in_channels,
            )

        multiplier_value = block_config.get("multiplier", 2)
        if not isinstance(multiplier_value, int) or isinstance(multiplier_value, bool):
            raise ValueError("Expected integer LTX encoder multiplier metadata")
        out_channels = in_channels * multiplier_value

        if block_name == "compress_space_res":
            stride = (1, 2, 2)
        elif block_name == "compress_time_res":
            stride = (2, 1, 1)
        elif block_name == "compress_all_res":
            stride = (2, 2, 2)
        elif block_name == "compress_space":
            return (
                _ConvWrapper(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=(1, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                ),
                in_channels,
            )
        elif block_name == "compress_time":
            return (
                _ConvWrapper(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=(2, 1, 1),
                    spatial_padding_mode=spatial_padding_mode,
                ),
                in_channels,
            )
        elif block_name == "compress_all":
            return (
                _ConvWrapper(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=(2, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                ),
                in_channels,
            )
        elif block_name == "compress_all_x_y":
            return (
                _ConvWrapper(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=(2, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                ),
                out_channels,
            )
        else:
            raise ValueError(f"Unsupported LTX encoder block '{block_name}'")

        return (
            SpaceToDepthDownsample(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                spatial_padding_mode=spatial_padding_mode,
            ),
            out_channels,
        )

    def __call__(self, sample: MLXArray) -> MLXArray:
        frame_count = int(sample.shape[2])
        if (frame_count - 1) % 8 != 0:
            raise ValueError(
                f"LTX encode input must have 1 + 8*n frames; got {frame_count} frames"
            )

        encoded = patchify_video(
            sample,
            patch_size_hw=self.patch_size,
            patch_size_t=1,
        )
        encoded = self.conv_in(encoded, causal=True)
        for block in self.down_blocks:
            encoded = block(encoded, causal=True)
        encoded = self._pixel_norm(encoded)
        encoded = self.conv_act(encoded)
        encoded = self.conv_out(encoded, causal=True)

        if self.latent_log_var == LatentLogVarianceType.UNIFORM:
            means = encoded[:, :-1, ...]
            logvar = encoded[:, -1:, ...]
            repeated_logvar = mx.tile(logvar, (1, int(means.shape[1]), 1, 1, 1))
            encoded = mx.concatenate([means, repeated_logvar], axis=1)
        elif self.latent_log_var == LatentLogVarianceType.CONSTANT:
            means = encoded[:, :-1, ...]
            approx_ln_0 = -30.0
            encoded = mx.concatenate(
                [
                    means,
                    mx.full(means.shape, approx_ln_0, dtype=means.dtype),
                ],
                axis=1,
            )

        means = encoded[:, : self.latent_channels, ...]
        return self.per_channel_statistics.normalize(means)

    @property
    def per_channel_statistics(self) -> _PerChannelStatistics:
        return self._per_channel_statistics

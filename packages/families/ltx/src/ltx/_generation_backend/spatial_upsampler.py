from __future__ import annotations

import mlx.core as mx

from .. import _nn_compat as nn_compat
from .types import MLXArray


class _Conv3DChannelsLast(nn_compat.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )
        stride_value = (stride, stride, stride) if isinstance(stride, int) else stride
        padding_value = (
            (padding, padding, padding) if isinstance(padding, int) else padding
        )
        dilation_value = (
            (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        )
        scale = 1.0 / (in_channels * kernel[0] * kernel[1] * kernel[2]) ** 0.5
        self.stride = stride_value
        self.padding = padding_value
        self.dilation = dilation_value
        self.groups = groups
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel[0], kernel[1], kernel[2], in_channels),
        )
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: MLXArray) -> MLXArray:
        result = mx.conv3d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is None:
            return result
        return result + self.bias


class _GroupNorm3DChannelsLast(nn_compat.Module):
    def __init__(self, *, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: MLXArray) -> MLXArray:
        batch, frames, height, width, channels = x.shape
        input_dtype = x.dtype
        normalized = x.astype(mx.float32)
        normalized = mx.reshape(
            normalized,
            (
                batch,
                frames * height * width,
                self.num_groups,
                channels // self.num_groups,
            ),
        )
        mean = mx.mean(normalized, axis=(1, 3), keepdims=True)
        variance = mx.var(normalized, axis=(1, 3), keepdims=True)
        normalized = (normalized - mean) / mx.sqrt(variance + self.eps)
        normalized = mx.reshape(normalized, (batch, frames, height, width, channels))
        normalized = normalized * self.weight.astype(mx.float32)
        normalized = normalized + self.bias.astype(mx.float32)
        return normalized.astype(input_dtype)


class _PixelShuffle2X(nn_compat.Module):
    def __call__(self, x: MLXArray) -> MLXArray:
        batch, height, width, channels = x.shape
        upscale = 2
        out_channels = channels // (upscale * upscale)
        shuffled = mx.reshape(x, (batch, height, width, out_channels, upscale, upscale))
        shuffled = mx.transpose(shuffled, (0, 1, 4, 2, 5, 3))
        return mx.reshape(
            shuffled, (batch, height * upscale, width * upscale, out_channels)
        )


class _FramewiseSpatialUpsampler(nn_compat.Module):
    def __init__(self, *, mid_channels: int) -> None:
        super().__init__()
        self.conv = nn_compat.Conv2d(
            mid_channels, 4 * mid_channels, kernel_size=3, padding=1
        )
        self.pixel_shuffle = _PixelShuffle2X()

    def __call__(self, x: MLXArray) -> MLXArray:
        batch, frames, height, width, channels = x.shape
        framewise = mx.reshape(x, (batch * frames, height, width, channels))
        framewise = self.conv(framewise)
        framewise = self.pixel_shuffle(framewise)
        return mx.reshape(framewise, (batch, frames, height * 2, width * 2, channels))


class _Residual3DBlock(nn_compat.Module):
    def __init__(self, *, channels: int) -> None:
        super().__init__()
        self.conv1 = _Conv3DChannelsLast(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.norm1 = _GroupNorm3DChannelsLast(
            num_groups=32,
            num_channels=channels,
        )
        self.conv2 = _Conv3DChannelsLast(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = _GroupNorm3DChannelsLast(
            num_groups=32,
            num_channels=channels,
        )

    def __call__(self, x: MLXArray) -> MLXArray:
        residual = x
        hidden = self.conv1(x)
        hidden = self.norm1(hidden)
        hidden = nn_compat.SiLU()(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        return nn_compat.SiLU()(hidden + residual)


class LatentUpsampler(nn_compat.Module):
    def __init__(
        self,
        *,
        in_channels: int = 128,
        mid_channels: int = 1024,
        num_blocks_per_stage: int = 4,
    ) -> None:
        super().__init__()
        self.initial_conv = _Conv3DChannelsLast(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
        )
        self.initial_norm = _GroupNorm3DChannelsLast(
            num_groups=32,
            num_channels=mid_channels,
        )
        self.res_blocks = {
            index: _Residual3DBlock(channels=mid_channels)
            for index in range(num_blocks_per_stage)
        }
        self.upsampler = _FramewiseSpatialUpsampler(mid_channels=mid_channels)
        self.post_upsample_res_blocks = {
            index: _Residual3DBlock(channels=mid_channels)
            for index in range(num_blocks_per_stage)
        }
        self.final_conv = _Conv3DChannelsLast(
            in_channels=mid_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, latent: MLXArray, debug: bool = False) -> MLXArray:
        del debug
        hidden = mx.transpose(latent, (0, 2, 3, 4, 1))
        hidden = self.initial_conv(hidden)
        hidden = self.initial_norm(hidden)
        hidden = nn_compat.SiLU()(hidden)
        for index in sorted(self.res_blocks):
            hidden = self.res_blocks[index](hidden)
        hidden = self.upsampler(hidden)
        for index in sorted(self.post_upsample_res_blocks):
            hidden = self.post_upsample_res_blocks[index](hidden)
        hidden = self.final_conv(hidden)
        return mx.transpose(hidden, (0, 4, 1, 2, 3))

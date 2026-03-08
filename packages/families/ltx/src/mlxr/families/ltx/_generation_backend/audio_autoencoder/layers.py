from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from ... import _nn_compat as nn
from .contracts import AudioCausalityAxis, AudioNormKind


class AudioFeatureLayer(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class PixelNorm(nn.Module):
    def __init__(self, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean_square = mx.mean(x**2, axis=-1, keepdims=True)
        return x / mx.sqrt(mean_square + self.eps)


class GroupNormChannelsLast(nn.Module):
    def __init__(
        self,
        dims: int,
        *,
        num_groups: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dims < 1:
            raise ValueError("GroupNormChannelsLast dims must be >= 1")
        group_count = min(num_groups, dims)
        while dims % group_count != 0:
            group_count -= 1
        if group_count < 1:
            raise ValueError(
                "GroupNormChannelsLast could not resolve a valid group count"
            )
        self.num_groups = group_count
        self.eps = eps
        self.weight = mx.ones((dims,), dtype=mx.float32)
        self.bias = mx.zeros((dims,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        batch, height, width, channels = (int(size) for size in x.shape)
        group_width = channels // self.num_groups
        grouped = x.reshape(batch, height, width, self.num_groups, group_width)
        mean = mx.mean(grouped, axis=(1, 2, 4), keepdims=True)
        variance = mx.mean((grouped - mean) ** 2, axis=(1, 2, 4), keepdims=True)
        normalized = (grouped - mean) / mx.sqrt(variance + self.eps)
        restored = normalized.reshape(batch, height, width, channels)
        weight = self.weight.astype(x.dtype).reshape(1, 1, 1, channels)
        bias = self.bias.astype(x.dtype).reshape(1, 1, 1, channels)
        return restored * weight + bias


def build_audio_normalization(
    dims: int,
    *,
    norm_kind: AudioNormKind,
    num_groups: int = 32,
) -> AudioFeatureLayer:
    if norm_kind is AudioNormKind.PIXEL:
        return PixelNorm()
    if norm_kind is AudioNormKind.GROUP:
        return GroupNormChannelsLast(dims, num_groups=num_groups)
    raise ValueError(f"Unsupported audio norm kind {norm_kind!r}")


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return value


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: AudioCausalityAxis = AudioCausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size)
        dilation_h, dilation_w = _pair(dilation)
        pad_h = (kernel_h - 1) * dilation_h
        pad_w = (kernel_w - 1) * dilation_w
        self.causality_axis = causality_axis
        if causality_axis is AudioCausalityAxis.NONE:
            self.padding = (
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
            )
        elif causality_axis in (
            AudioCausalityAxis.WIDTH,
            AudioCausalityAxis.WIDTH_COMPATIBILITY,
        ):
            self.padding = (pad_h // 2, pad_h - pad_h // 2, pad_w, 0)
        elif causality_axis is AudioCausalityAxis.HEIGHT:
            self.padding = (pad_h, 0, pad_w // 2, pad_w - pad_w // 2)
        else:
            raise ValueError(f"Unsupported audio causality axis {causality_axis!r}")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_h, kernel_w),
            stride=stride,
            padding=0,
            dilation=(dilation_h, dilation_w),
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = self.padding
        if any(value > 0 for value in self.padding):
            x = mx.pad(
                x,
                [
                    (0, 0),
                    (pad_h_top, pad_h_bottom),
                    (pad_w_left, pad_w_right),
                    (0, 0),
                ],
            )
        return self.conv(x)


def make_audio_conv2d(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int | tuple[int, int],
    stride: int = 1,
    padding: int | tuple[int, int] | None = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: AudioCausalityAxis | None = None,
) -> AudioFeatureLayer:
    if causality_axis is not None:
        return CausalConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causality_axis=causality_axis,
        )
    if padding is None:
        kernel_h, kernel_w = _pair(kernel_size)
        padding = (kernel_h // 2, kernel_w // 2)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=_pair(kernel_size),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, *, norm_kind: AudioNormKind) -> None:
        super().__init__()
        self.norm = build_audio_normalization(in_channels, norm_kind=norm_kind)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        normalized = self.norm(x)
        query = self.q(normalized)
        key = self.k(normalized)
        value = self.v(normalized)
        batch, height, width, channels = (int(size) for size in query.shape)
        query = query.reshape(batch, height * width, channels)
        key = key.reshape(batch, height * width, channels)
        value = value.reshape(batch, height * width, channels)
        weights = mx.softmax(
            mx.matmul(query, mx.transpose(key, (0, 2, 1))) * (float(channels) ** -0.5),
            axis=-1,
        )
        attended = mx.matmul(weights, value).reshape(batch, height, width, channels)
        return residual + self.proj_out(attended)


@dataclass(frozen=True, slots=True)
class StageShape:
    block_channels: int
    resolution: int

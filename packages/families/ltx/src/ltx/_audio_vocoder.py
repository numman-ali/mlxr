from __future__ import annotations

# mypy: ignore-errors
import math
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

LRELU_SLOPE = 0.1


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def _leaky_relu(x: mx.array, negative_slope: float = LRELU_SLOPE) -> mx.array:
    return mx.maximum(x, x * negative_slope)


def _sinc(x: mx.array) -> mx.array:
    return mx.where(
        x == 0,
        mx.array(1.0, dtype=x.dtype),
        mx.sin(math.pi * x) / math.pi / x,
    )


def _kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
) -> mx.array:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21.0) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0
    window = mx.array(np.kaiser(kernel_size, beta=beta))
    if even:
        time = mx.arange(-half_size, half_size) + 0.5
    else:
        time = mx.arange(kernel_size) - half_size
    if cutoff == 0:
        return mx.zeros_like(time).reshape(1, kernel_size, 1)
    filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
    filter_ = filter_ / filter_.sum()
    return filter_.reshape(1, kernel_size, 1)


class Snake(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        alpha: float = 1.0,
        alpha_logscale: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = (
            mx.zeros((in_features,), dtype=mx.float32)
            if alpha_logscale
            else mx.ones((in_features,), dtype=mx.float32) * alpha
        )

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha[None, None, :]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        return x + (1.0 / (alpha + 1e-9)) * mx.power(mx.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        alpha: float = 1.0,
        alpha_logscale: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        base = (
            mx.zeros((in_features,), dtype=mx.float32)
            if alpha_logscale
            else mx.ones((in_features,), dtype=mx.float32) * alpha
        )
        self.alpha = base
        self.beta = mx.array(base)

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha[None, None, :]
        beta = self.beta[None, None, :]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(x * alpha), 2)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        *,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "edge",
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.stride = stride
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.padding = padding
        self.padding_mode = padding_mode
        self.filter = _kaiser_sinc_filter1d(
            cutoff=cutoff,
            half_width=half_width,
            kernel_size=kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        _, _, channels = x.shape
        if self.padding:
            x = mx.pad(
                x,
                ((0, 0), (self.pad_left, self.pad_right), (0, 0)),
                mode=self.padding_mode,
            )
        expanded_filter = mx.broadcast_to(
            self.filter, (channels, self.filter.shape[1], self.filter.shape[2])
        )
        return mx.conv1d(x, expanded_filter, stride=self.stride, groups=channels)


class UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        self.filter = _kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        _, _, channels = x.shape
        x = mx.pad(x, ((0, 0), (self.pad, self.pad), (0, 0)), mode="edge")
        expanded_filter = mx.broadcast_to(
            self.filter, (channels, self.filter.shape[1], self.filter.shape[2])
        )
        x = self.ratio * mx.conv_transpose1d(
            x,
            expanded_filter,
            stride=self.stride,
            groups=channels,
        )
        return x[:, self.pad_left : -self.pad_right, :]


class DownSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=(
                int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            ),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        *,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


class ResBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.convs1 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=_get_padding(kernel_size, d),
            )
            for d in dilation
        ]
        self.convs2 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=_get_padding(kernel_size, 1),
            )
            for _ in dilation
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            xt = _leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = _leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = x + xt
        return x


class ResBlock2(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int] = (1, 3),
    ) -> None:
        super().__init__()
        self.convs = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=_get_padding(kernel_size, d),
            )
            for d in dilation
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for conv in self.convs:
            xt = _leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = x + xt
        return x


class AMPBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        activation: str = "snakebeta",
    ) -> None:
        super().__init__()
        act_cls = SnakeBeta if activation == "snakebeta" else Snake
        self.convs1 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=_get_padding(kernel_size, d),
            )
            for d in dilation
        ]
        self.convs2 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=_get_padding(kernel_size, 1),
            )
            for _ in dilation
        ]
        self.acts1 = [
            Activation1d(act_cls(channels, alpha_logscale=True)) for _ in dilation
        ]
        self.acts2 = [
            Activation1d(act_cls(channels, alpha_logscale=True)) for _ in dilation
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for conv1, conv2, act1, act2 in zip(
            self.convs1,
            self.convs2,
            self.acts1,
            self.acts2,
            strict=True,
        ):
            xt = act1(x)
            xt = conv1(xt)
            xt = act2(xt)
            xt = conv2(xt)
            x = x + xt
        return x


class AudioVocoder(nn.Module):
    def __init__(
        self,
        *,
        resblock_kernel_sizes: Iterable[int] | None = None,
        upsample_rates: Iterable[int] | None = None,
        upsample_kernel_sizes: Iterable[int] | None = None,
        resblock_dilation_sizes: Iterable[Iterable[int]] | None = None,
        upsample_initial_channel: int = 1024,
        stereo: bool = True,
        resblock: str = "1",
        output_sample_rate: int = 24000,
        activation: str = "snake",
        use_tanh_at_final: bool = True,
        apply_final_activation: bool = True,
        use_bias_at_final: bool = True,
    ) -> None:
        super().__init__()
        self.output_sample_rate = output_sample_rate
        kernel_sizes = list(resblock_kernel_sizes or [3, 7, 11])
        up_rates = list(upsample_rates or [6, 5, 2, 2, 2])
        up_kernels = list(upsample_kernel_sizes or [16, 15, 8, 4, 4])
        dilation_sizes = [
            tuple(int(v) for v in block)
            for block in (resblock_dilation_sizes or ([1, 3, 5], [1, 3, 5], [1, 3, 5]))
        ]
        self.num_kernels = len(kernel_sizes)
        self.num_upsamples = len(up_rates)
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation
        self.is_amp = resblock == "AMP1"

        in_channels = 128 if stereo else 64
        self.conv_pre = nn.Conv1d(
            in_channels,
            upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.ups = [
            nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            )
            for i, (stride, kernel_size) in enumerate(
                zip(up_rates, up_kernels, strict=True)
            )
        ]

        if resblock == "1":
            resblock_cls: type[nn.Module] = ResBlock1
        elif resblock == "2":
            resblock_cls = ResBlock2
        elif resblock == "AMP1":
            resblock_cls = AMPBlock1
        else:
            raise ValueError(f"Unsupported LTX vocoder resblock {resblock!r}")

        self.resblocks = []
        for i in range(len(self.ups)):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(
                kernel_sizes,
                dilation_sizes,
                strict=True,
            ):
                if resblock == "AMP1":
                    block = resblock_cls(
                        channels,
                        kernel_size=kernel_size,
                        dilation=tuple(int(v) for v in dilations),
                        activation=activation,
                    )
                else:
                    block = resblock_cls(
                        channels,
                        kernel_size=kernel_size,
                        dilation=tuple(int(v) for v in dilations),
                    )
                self.resblocks.append(block)

        final_channels = upsample_initial_channel // (2 ** len(self.ups))
        if self.is_amp:
            act_cls = SnakeBeta if activation == "snakebeta" else Snake
            self.act_post: nn.Module = Activation1d(
                act_cls(final_channels, alpha_logscale=True)
            )
        else:
            self.act_post = nn.LeakyReLU()

        out_channels = 2 if stereo else 1
        self.conv_post = nn.Conv1d(
            final_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=use_bias_at_final,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 1, 3, 2))
        if x.ndim == 4:
            batch, stereo_channels, mel_bins, time = x.shape
            x = x.reshape(batch, stereo_channels * mel_bins, time)
        x = mx.transpose(x, (0, 2, 1))
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            if not self.is_amp:
                x = _leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels
            block_outputs = [self.resblocks[idx](x) for idx in range(start, end)]
            x = mx.mean(mx.stack(block_outputs, axis=0), axis=0)

        x = self.act_post(x) if self.is_amp else nn.leaky_relu(x)
        x = self.conv_post(x)
        if self.apply_final_activation:
            x = mx.tanh(x) if self.use_tanh_at_final else mx.clip(x, -1.0, 1.0)
        return mx.transpose(x, (0, 2, 1))

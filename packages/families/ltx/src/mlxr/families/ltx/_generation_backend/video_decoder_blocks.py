from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import mlx.core as mx

from .. import _nn_compat as nn
from .types import MLXArray


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"


def _reflect_pad_2d(x: MLXArray, *, pad_h: int, pad_w: int) -> MLXArray:
    if pad_h == 0 and pad_w == 0:
        return x

    if pad_h > 0:
        top_pad = x[:, :, 1 : pad_h + 1, :, :][:, :, ::-1, :, :]
        bottom_pad = x[:, :, -pad_h - 1 : -1, :, :][:, :, ::-1, :, :]
        x = mx.concatenate([top_pad, x, bottom_pad], axis=2)

    if pad_w > 0:
        left_pad = x[:, :, :, 1 : pad_w + 1, :][:, :, :, ::-1, :]
        right_pad = x[:, :, :, -pad_w - 1 : -1, :][:, :, :, ::-1, :]
        x = mx.concatenate([left_pad, x, right_pad], axis=3)

    return x


class CausalConv3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | str = 0,
        causal: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        del padding
        super().__init__()
        self.causal = causal
        self.spatial_padding_mode = spatial_padding_mode

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.time_kernel_size = kernel_size[0]
        self.spatial_padding = (kernel_size[1] // 2, kernel_size[2] // 2)

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

    def __call__(self, x: MLXArray, *, causal: bool | None = None) -> MLXArray:
        use_causal = self.causal if causal is None else causal

        if self.time_kernel_size > 1:
            if use_causal:
                first_frame_pad = mx.repeat(
                    x[:, :, :1, :, :],
                    self.time_kernel_size - 1,
                    axis=2,
                )
                x = mx.concatenate([first_frame_pad, x], axis=2)
            else:
                pad_size = (self.time_kernel_size - 1) // 2
                if pad_size > 0:
                    first_frame_pad = mx.repeat(x[:, :, :1, :, :], pad_size, axis=2)
                    last_frame_pad = mx.repeat(x[:, :, -1:, :, :], pad_size, axis=2)
                    x = mx.concatenate([first_frame_pad, x, last_frame_pad], axis=2)

        x = mx.transpose(x, (0, 2, 3, 4, 1))

        pad_h, pad_w = self.spatial_padding
        if pad_h > 0 or pad_w > 0:
            if self.spatial_padding_mode == PaddingModeType.REFLECT:
                x = _reflect_pad_2d(x, pad_h=pad_h, pad_w=pad_w)
            else:
                x = mx.pad(
                    x,
                    [
                        (0, 0),
                        (0, 0),
                        (pad_h, pad_h),
                        (pad_w, pad_w),
                        (0, 0),
                    ],
                )

        x = self._chunked_conv3d(x)
        return mx.transpose(x, (0, 4, 1, 2, 3))

    def _chunked_conv3d(self, x: MLXArray) -> MLXArray:
        _, d, h, w, c = x.shape
        total_elements = d * h * w * c
        max_safe_elements = 30 * 192 * 192 * 128

        if total_elements <= max_safe_elements:
            return self.conv(x)

        elements_per_frame = h * w * c
        max_frames_per_chunk = max(1, max_safe_elements // elements_per_frame)
        chunk_size = min(max_frames_per_chunk, 24)
        overlap = self.time_kernel_size - 1
        expected_output_frames = d - overlap

        outputs: list[MLXArray] = []
        out_idx = 0
        in_start = 0
        while out_idx < expected_output_frames:
            remaining = expected_output_frames - out_idx
            out_frames_this_chunk = min(chunk_size, remaining)
            in_end = min(in_start + out_frames_this_chunk + overlap, d)
            chunk = x[:, in_start:in_end, :, :, :]
            chunk_out = self.conv(chunk)
            mx.eval(chunk_out)
            outputs.append(chunk_out)
            out_idx += int(chunk_out.shape[1])
            in_start += int(chunk_out.shape[1])

        if len(outputs) == 1:
            return outputs[0]
        return mx.concatenate(outputs, axis=1)


def _get_timestep_embedding(
    timesteps: MLXArray,
    *,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> MLXArray:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = scale * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, *, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.act = nn.SiLU()

    def __call__(self, sample: MLXArray) -> MLXArray:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class PixArtAlphaTimestepEmbedder(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def __call__(
        self,
        timestep: MLXArray,
        *,
        hidden_dtype: mx.Dtype = mx.float32,
    ) -> MLXArray:
        timestep_proj = _get_timestep_embedding(
            timestep,
            embedding_dim=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        return self.timestep_embedder(timestep_proj.astype(hidden_dtype))


class _ConvWrapper(nn.Module):
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

    def __call__(self, x: MLXArray, *, causal: bool = False) -> MLXArray:
        return self.conv(x, causal=causal)


class ResnetBlock3DSimple(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.timestep_conditioning = timestep_conditioning
        self.conv1 = _ConvWrapper(
            in_channels=channels,
            out_channels=channels,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv2 = _ConvWrapper(
            in_channels=channels,
            out_channels=channels,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.act = nn.SiLU()
        if timestep_conditioning:
            self.scale_shift_table = mx.zeros((4, channels))

    def pixel_norm(self, x: MLXArray, eps: float = 1e-8) -> MLXArray:
        return x / mx.sqrt(mx.mean(x**2, axis=1, keepdims=True) + eps)

    def __call__(
        self,
        x: MLXArray,
        *,
        causal: bool = False,
        timestep_embed: Optional[MLXArray] = None,
    ) -> MLXArray:
        residual = x
        batch_size = int(x.shape[0])

        x = self.pixel_norm(x)
        shift1 = scale1 = shift2 = scale2 = None
        if self.timestep_conditioning and timestep_embed is not None:
            ada_values = self.scale_shift_table[None, :, :, None, None, None]
            channels = int(self.scale_shift_table.shape[1])
            reshaped = timestep_embed.reshape(batch_size, 4, channels, 1, 1, 1)
            ada_values = ada_values + reshaped
            shift1 = ada_values[:, 0]
            scale1 = ada_values[:, 1]
            shift2 = ada_values[:, 2]
            scale2 = ada_values[:, 3]
            x = x * (1 + scale1) + shift1

        x = self.act(x)
        x = self.conv1(x, causal=causal)
        x = self.pixel_norm(x)

        if self.timestep_conditioning and timestep_embed is not None:
            if shift2 is None or scale2 is None:
                raise RuntimeError("Expected timestep-conditioned residual modulation")
            x = x * (1 + scale2) + shift2

        x = self.act(x)
        x = self.conv2(x, causal=causal)
        return x + residual


class ResBlockGroup(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int = 5,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaTimestepEmbedder(embedding_dim=channels * 4)
        self.res_blocks = [
            ResnetBlock3DSimple(
                channels=channels,
                spatial_padding_mode=spatial_padding_mode,
                timestep_conditioning=timestep_conditioning,
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        x: MLXArray,
        *,
        causal: bool = False,
        timestep: Optional[MLXArray] = None,
    ) -> MLXArray:
        timestep_embed = None
        if self.timestep_conditioning and timestep is not None:
            batch_size = int(x.shape[0])
            timestep_embed = self.time_embedder(
                timestep.flatten(),
                hidden_dtype=x.dtype,
            )
            timestep_embed = timestep_embed.reshape(batch_size, -1, 1, 1, 1)

        for res_block in self.res_blocks:
            x = res_block(x, causal=causal, timestep_embed=timestep_embed)
        return x


class DepthToSpaceUpsample(nn.Module):
    def __init__(
        self,
        *,
        dims: int,
        in_channels: int,
        stride: int | tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        del dims
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
        multiplier = stride[0] * stride[1] * stride[2]
        self.out_channels = in_channels // out_channels_reduction_factor
        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=self.out_channels * multiplier,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _depth_to_space(self, x: MLXArray) -> MLXArray:
        b, packed_channels, d, h, w = x.shape
        st, sh, sw = self.stride
        channels = packed_channels // (st * sh * sw)
        x = mx.reshape(x, (b, channels, st, sh, sw, d, h, w))
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        return mx.reshape(x, (b, channels, d * st, h * sh, w * sw))

    def __call__(
        self,
        x: MLXArray,
        *,
        causal: bool = True,
        chunked_conv: bool = False,
    ) -> MLXArray:
        del chunked_conv
        _, _, _, _, _ = x.shape
        st, sh, sw = self.stride

        x_residual = None
        if self.residual:
            x_residual = self._depth_to_space(x)
            num_repeat = (st * sh * sw) // self.out_channels_reduction_factor
            x_residual = mx.tile(x_residual, (1, num_repeat, 1, 1, 1))
            if st > 1:
                x_residual = x_residual[:, :, 1:, :, :]

        x = self.conv(x, causal=causal)
        x = self._depth_to_space(x)
        if st > 1:
            x = x[:, :, 1:, :, :]
        if self.residual and x_residual is not None:
            x = x + x_residual
        return x

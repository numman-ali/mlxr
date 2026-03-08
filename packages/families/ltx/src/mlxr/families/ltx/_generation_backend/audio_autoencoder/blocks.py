from __future__ import annotations

import mlx.core as mx

from ... import _nn_compat as nn
from .contracts import AudioCausalityAxis, AudioNormKind
from .layers import (
    AudioFeatureLayer,
    SelfAttention2d,
    build_audio_normalization,
    make_audio_conv2d,
)

LRELU_SLOPE = 0.1


class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 0,
        norm_kind: AudioNormKind,
        causality_axis: AudioCausalityAxis,
    ) -> None:
        super().__init__()
        resolved_out_channels = in_channels if out_channels is None else out_channels
        if (
            causality_axis is not AudioCausalityAxis.NONE
            and norm_kind is AudioNormKind.GROUP
        ):
            raise ValueError(
                "Causal audio residual blocks do not support group normalization"
            )
        self.in_channels = in_channels
        self.out_channels = resolved_out_channels
        self.temb_channels = temb_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1: AudioFeatureLayer = build_audio_normalization(
            in_channels,
            norm_kind=norm_kind,
        )
        self.conv1 = make_audio_conv2d(
            in_channels,
            resolved_out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, resolved_out_channels)
        self.norm2: AudioFeatureLayer = build_audio_normalization(
            resolved_out_channels,
            norm_kind=norm_kind,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.conv2 = make_audio_conv2d(
            resolved_out_channels,
            resolved_out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )
        if in_channels != resolved_out_channels:
            if conv_shortcut:
                self.conv_shortcut = make_audio_conv2d(
                    in_channels,
                    resolved_out_channels,
                    kernel_size=3,
                    stride=1,
                    causality_axis=causality_axis,
                )
            else:
                self.nin_shortcut = make_audio_conv2d(
                    in_channels,
                    resolved_out_channels,
                    kernel_size=1,
                    stride=1,
                    causality_axis=causality_axis,
                )

    def __call__(self, x: mx.array, temb: mx.array | None = None) -> mx.array:
        hidden = nn.SiLU()(self.norm1(x))
        hidden = self.conv1(hidden)
        if temb is not None and self.temb_channels > 0:
            projected = nn.SiLU()(self.temb_proj(temb))
            hidden = hidden + projected[:, None, None, :]
        hidden = nn.SiLU()(self.norm2(hidden))
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)
        residual = x
        if self.in_channels != self.out_channels:
            if hasattr(self, "conv_shortcut"):
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        return residual + hidden


class Downsample2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        with_conv: bool,
        causality_axis: AudioCausalityAxis,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        self.conv: nn.Conv2d | None = None
        if causality_axis is not AudioCausalityAxis.NONE and not with_conv:
            raise ValueError("Causal audio downsampling requires convolution")
        if with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    def __call__(self, x: mx.array) -> mx.array:
        if not self.with_conv:
            batch, height, width, channels = (int(size) for size in x.shape)
            return mx.mean(
                x.reshape(batch, height // 2, 2, width // 2, 2, channels),
                axis=(2, 4),
            )
        if self.causality_axis is AudioCausalityAxis.NONE:
            pad = [(0, 0), (0, 1), (0, 1), (0, 0)]
        elif self.causality_axis is AudioCausalityAxis.WIDTH:
            pad = [(0, 0), (0, 1), (2, 0), (0, 0)]
        elif self.causality_axis is AudioCausalityAxis.HEIGHT:
            pad = [(0, 0), (2, 0), (0, 1), (0, 0)]
        else:
            pad = [(0, 0), (0, 1), (1, 0), (0, 0)]
        if self.conv is None:
            raise RuntimeError("Downsample2d convolution is missing for with_conv path")
        return self.conv(mx.pad(x, pad, constant_values=0))


class Upsample2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        with_conv: bool,
        causality_axis: AudioCausalityAxis,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        self.conv: AudioFeatureLayer | None = None
        if with_conv:
            self.conv = make_audio_conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )

    def __call__(self, x: mx.array) -> mx.array:
        batch, height, width, channels = (int(size) for size in x.shape)
        upsampled = mx.broadcast_to(
            x[:, :, None, :, None, :],
            (batch, height, 2, width, 2, channels),
        ).reshape(batch, height * 2, width * 2, channels)
        if not self.with_conv:
            return upsampled
        if self.conv is None:
            raise RuntimeError("Upsample2d convolution is missing for with_conv path")
        convolved = self.conv(upsampled)
        if self.causality_axis is AudioCausalityAxis.HEIGHT:
            return convolved[:, 1:, :, :]
        if self.causality_axis is AudioCausalityAxis.WIDTH:
            return convolved[:, :, 1:, :]
        return convolved


class EncoderStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block: dict[int, ResidualBlock2d] = {}
        self.attn: dict[int, SelfAttention2d] = {}
        self.downsample: Downsample2d | None = None


class DecoderStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block: dict[int, ResidualBlock2d] = {}
        self.attn: dict[int, SelfAttention2d] = {}
        self.upsample: Upsample2d | None = None


class MidBlock(nn.Module):
    def __init__(
        self,
        *,
        block_1: ResidualBlock2d,
        attn_1: SelfAttention2d | None,
        block_2: ResidualBlock2d,
    ) -> None:
        super().__init__()
        self.block_1 = block_1
        self.attn_1 = attn_1
        self.block_2 = block_2


def build_mid_block(
    *,
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_kind: AudioNormKind,
    causality_axis: AudioCausalityAxis,
    add_attention: bool,
) -> MidBlock:
    return MidBlock(
        block_1=ResidualBlock2d(
            in_channels=channels,
            out_channels=channels,
            temb_channels=temb_channels,
            dropout=dropout,
            norm_kind=norm_kind,
            causality_axis=causality_axis,
        ),
        attn_1=SelfAttention2d(channels, norm_kind=norm_kind)
        if add_attention
        else None,
        block_2=ResidualBlock2d(
            in_channels=channels,
            out_channels=channels,
            temb_channels=temb_channels,
            dropout=dropout,
            norm_kind=norm_kind,
            causality_axis=causality_axis,
        ),
    )


def run_mid_block(mid: MidBlock, features: mx.array) -> mx.array:
    hidden = mid.block_1(features, temb=None)
    if mid.attn_1 is not None:
        hidden = mid.attn_1(hidden)
    return mid.block_2(hidden, temb=None)


def build_downsampling_stages(
    *,
    ch: int,
    ch_mult: tuple[int, ...],
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_kind: AudioNormKind,
    causality_axis: AudioCausalityAxis,
    attn_resolutions: set[int],
    resample_with_conv: bool,
) -> tuple[dict[int, EncoderStage], int]:
    stages: dict[int, EncoderStage] = {}
    current_resolution = resolution
    input_multipliers = (1, *ch_mult)
    block_in = ch
    for level in range(len(ch_mult)):
        stage = EncoderStage()
        block_in = ch * input_multipliers[level]
        block_out = ch * ch_mult[level]
        for block_index in range(num_res_blocks):
            stage.block[block_index] = ResidualBlock2d(
                in_channels=block_in,
                out_channels=block_out,
                temb_channels=temb_channels,
                dropout=dropout,
                norm_kind=norm_kind,
                causality_axis=causality_axis,
            )
            block_in = block_out
            if current_resolution in attn_resolutions:
                stage.attn[block_index] = SelfAttention2d(
                    block_in,
                    norm_kind=norm_kind,
                )
        if level != len(ch_mult) - 1:
            stage.downsample = Downsample2d(
                block_in,
                with_conv=resample_with_conv,
                causality_axis=causality_axis,
            )
            current_resolution //= 2
        stages[level] = stage
    return stages, block_in


def build_upsampling_stages(
    *,
    ch: int,
    ch_mult: tuple[int, ...],
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_kind: AudioNormKind,
    causality_axis: AudioCausalityAxis,
    attn_resolutions: set[int],
    resample_with_conv: bool,
    initial_block_channels: int,
) -> tuple[dict[int, DecoderStage], int]:
    stages: dict[int, DecoderStage] = {}
    current_resolution = resolution // (2 ** (len(ch_mult) - 1))
    block_in = initial_block_channels
    for level in reversed(range(len(ch_mult))):
        stage = DecoderStage()
        block_out = ch * ch_mult[level]
        for block_index in range(num_res_blocks + 1):
            stage.block[block_index] = ResidualBlock2d(
                in_channels=block_in,
                out_channels=block_out,
                temb_channels=temb_channels,
                dropout=dropout,
                norm_kind=norm_kind,
                causality_axis=causality_axis,
            )
            block_in = block_out
            if current_resolution in attn_resolutions:
                stage.attn[block_index] = SelfAttention2d(
                    block_in,
                    norm_kind=norm_kind,
                )
        if level != 0:
            stage.upsample = Upsample2d(
                block_in,
                with_conv=resample_with_conv,
                causality_axis=causality_axis,
            )
            current_resolution *= 2
        stages[level] = stage
    return stages, block_in

from __future__ import annotations

import math

import mlx.core as mx

from .. import _nn_compat as nn
from .config import AutoencoderConfig
from .constants import _PATCH_SIZE
from .embeddings import _silu


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=groups,
            dims=in_channels,
            eps=1.0e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=groups,
            dims=out_channels,
            eps=1.0e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_shortcut = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x)
        hidden = self.conv1(_silu(self.norm1(x)))
        hidden = self.conv2(_silu(self.norm2(hidden)))
        return residual + hidden


class AttentionBlock2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(
            num_groups=32,
            dims=channels,
            eps=1.0e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, height, width, channels = (int(size) for size in x.shape)
        hidden = self.group_norm(x).reshape(batch_size, 1, height * width, channels)
        q = self.to_q(hidden)
        k = self.to_k(hidden)
        v = self.to_v(hidden)
        attended = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=channels**-0.5,
        )
        attended = self.to_out(attended).reshape(batch_size, height, width, channels)
        return x + attended


class Downsample2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)]))


class Upsample2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        hidden = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        return self.conv(hidden)


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        add_downsample: bool,
        groups: int,
    ) -> None:
        super().__init__()
        self.resnets: list[ResnetBlock2D] = []
        current_channels = in_channels
        for _ in range(num_layers):
            block = ResnetBlock2D(current_channels, out_channels, groups)
            self.resnets.append(block)
            current_channels = out_channels
        self.downsamplers = [Downsample2D(out_channels)] if add_downsample else []

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.resnets:
            x = block(x)
        for downsample in self.downsamplers:
            x = downsample(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        add_upsample: bool,
        groups: int,
    ) -> None:
        super().__init__()
        self.resnets: list[ResnetBlock2D] = []
        current_channels = in_channels
        for _ in range(num_layers):
            block = ResnetBlock2D(current_channels, out_channels, groups)
            self.resnets.append(block)
            current_channels = out_channels
        self.upsamplers = [Upsample2D(out_channels)] if add_upsample else []

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.resnets:
            x = block(x)
        for upsample in self.upsamplers:
            x = upsample(x)
        return x


class UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.resnets = [
            ResnetBlock2D(channels, channels, groups),
            ResnetBlock2D(channels, channels, groups),
        ]
        self.attentions = [AttentionBlock2D(channels)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(
            config.in_channels,
            config.block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down_blocks: list[DownEncoderBlock2D] = []
        current_channels = config.block_out_channels[0]
        for index, out_channels in enumerate(config.block_out_channels):
            block = DownEncoderBlock2D(
                current_channels,
                out_channels,
                num_layers=config.layers_per_block,
                add_downsample=index < len(config.block_out_channels) - 1,
                groups=config.norm_num_groups,
            )
            self.down_blocks.append(block)
            current_channels = out_channels
        self.mid_block = UNetMidBlock2D(current_channels, config.norm_num_groups)
        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups,
            dims=current_channels,
            eps=1.0e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv_out = nn.Conv2d(
            current_channels,
            config.latent_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.conv_in(x)
        for block in self.down_blocks:
            hidden = block(hidden)
        hidden = self.mid_block(hidden)
        return self.conv_out(_silu(self.conv_norm_out(hidden)))


class Decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        current_channels = config.block_out_channels[-1]
        self.conv_in = nn.Conv2d(
            config.latent_channels,
            current_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.mid_block = UNetMidBlock2D(current_channels, config.norm_num_groups)
        self.up_blocks: list[UpDecoderBlock2D] = []
        reversed_channels = list(reversed(config.block_out_channels))
        for index, out_channels in enumerate(reversed_channels):
            block = UpDecoderBlock2D(
                current_channels,
                out_channels,
                num_layers=config.layers_per_block + 1,
                add_upsample=index < len(reversed_channels) - 1,
                groups=config.norm_num_groups,
            )
            self.up_blocks.append(block)
            current_channels = out_channels
        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups,
            dims=current_channels,
            eps=1.0e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv_out = nn.Conv2d(
            current_channels,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, z: mx.array) -> mx.array:
        hidden = self.conv_in(z)
        hidden = self.mid_block(hidden)
        for block in self.up_blocks:
            hidden = block(hidden)
        return self.conv_out(_silu(self.conv_norm_out(hidden)))


class BatchNormState(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.running_mean = mx.zeros((channels,), dtype=mx.float32)
        self.running_var = mx.ones((channels,), dtype=mx.float32)
        self.num_batches_tracked = mx.array(0, dtype=mx.int64)


class AutoencoderKLFlux2(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = nn.Conv2d(
            config.latent_channels * 2,
            config.latent_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.post_quant_conv = nn.Conv2d(
            config.latent_channels,
            config.latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        packed_channels = config.latent_channels * math.prod(config.patch_size)
        self.bn = BatchNormState(packed_channels)
        self.bn_eps = 1.0e-4

    def normalize(self, z: mx.array) -> mx.array:
        mean = self.bn.running_mean.reshape(1, 1, 1, -1).astype(mx.float32)
        var = self.bn.running_var.reshape(1, 1, 1, -1).astype(mx.float32)
        return ((z.astype(mx.float32) - mean) / mx.sqrt(var + self.bn_eps)).astype(
            z.dtype
        )

    def inv_normalize(self, z: mx.array) -> mx.array:
        mean = self.bn.running_mean.reshape(1, 1, 1, -1).astype(mx.float32)
        var = self.bn.running_var.reshape(1, 1, 1, -1).astype(mx.float32)
        return z.astype(mx.float32) * mx.sqrt(var + self.bn_eps) + mean

    def encode(self, x: mx.array) -> mx.array:
        moments = self.quant_conv(self.encoder(x.astype(mx.float32)))
        mean, _ = mx.split(moments, 2, axis=-1)
        batch_size, height, width, channels = (int(size) for size in mean.shape)
        mean = mean.reshape(
            batch_size,
            height // _PATCH_SIZE,
            _PATCH_SIZE,
            width // _PATCH_SIZE,
            _PATCH_SIZE,
            channels,
        )
        mean = mean.transpose(0, 1, 3, 5, 2, 4).reshape(
            batch_size,
            height // _PATCH_SIZE,
            width // _PATCH_SIZE,
            channels * (_PATCH_SIZE * _PATCH_SIZE),
        )
        return self.normalize(mean)

    def decode(self, z: mx.array) -> mx.array:
        batch_size, height, width, channels = (int(size) for size in z.shape)
        z = self.inv_normalize(z)
        z = z.reshape(
            batch_size,
            height,
            width,
            channels // (_PATCH_SIZE * _PATCH_SIZE),
            _PATCH_SIZE,
            _PATCH_SIZE,
        )
        z = z.transpose(0, 1, 4, 2, 5, 3).reshape(
            batch_size,
            height * _PATCH_SIZE,
            width * _PATCH_SIZE,
            channels // (_PATCH_SIZE * _PATCH_SIZE),
        )
        return self.decoder(self.post_quant_conv(z.astype(mx.float32)))

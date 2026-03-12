from __future__ import annotations

import math

import mlx.core as mx

from .. import _nn_compat as nn
from .config import AutoencoderConfig
from .embeddings import silu


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = mx.zeros(
            (out_channels, kernel_size, kernel_size, in_channels),
            dtype=mx.float32,
        )
        self.bias = mx.zeros((out_channels,), dtype=mx.float32) if bias else None
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        hidden = mx.conv2d(
            x,
            self.weight.astype(x.dtype),
            stride=self.stride,
            padding=self.padding,
        )
        if self.bias is not None:
            hidden = hidden + self.bias.astype(hidden.dtype)
        return hidden


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.weight = mx.zeros(
            (
                out_channels,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
                in_channels,
            ),
            dtype=mx.float32,
        )
        self.bias = mx.zeros((out_channels,), dtype=mx.float32) if bias else None
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        time_pad, height_pad, width_pad = self.padding
        if time_pad or height_pad or width_pad:
            x = mx.pad(
                x,
                [
                    (0, 0),
                    (2 * time_pad, 0),
                    (height_pad, height_pad),
                    (width_pad, width_pad),
                    (0, 0),
                ],
            )
        hidden = mx.conv3d(
            x,
            self.weight.astype(x.dtype),
            stride=self.stride,
            padding=(0, 0, 0),
        )
        if self.bias is not None:
            hidden = hidden + self.bias.astype(hidden.dtype)
        return hidden


class QwenImageRMSNorm(nn.Module):
    def __init__(self, dims: int, *, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.gamma = mx.ones((dims,), dtype=mx.float32)
        self.scale = math.sqrt(dims)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(
            mx.sum(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True) + self.eps
        )
        normalized = x.astype(mx.float32) * (self.scale / rms)
        return (normalized * self.gamma.astype(normalized.dtype)).astype(x.dtype)


class QwenImageResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        del dropout
        super().__init__()
        self.norm1 = QwenImageRMSNorm(in_dim)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMSNorm(out_dim)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x)
        hidden = self.conv1(silu(self.norm1(x)))
        hidden = self.conv2(silu(self.norm2(hidden)))
        return hidden + residual


class QwenImageAttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = QwenImageRMSNorm(dim)
        self.to_qkv = Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.proj = Conv2d(dim, dim, kernel_size=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, frames, height, width, channels = (int(size) for size in x.shape)
        hidden = self.norm(x).reshape(batch_size * frames, height, width, channels)
        qkv = self.to_qkv(hidden).reshape(
            batch_size * frames, 1, height * width, channels * 3
        )
        q, k, v = mx.split(qkv, 3, axis=-1)
        attended = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=channels**-0.5,
        )
        attended = attended.reshape(batch_size * frames, height, width, channels)
        attended = self.proj(attended).reshape(
            batch_size, frames, height, width, channels
        )
        return x + attended


class QwenImageMidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, *, num_layers: int = 1) -> None:
        super().__init__()
        self.resnets = [QwenImageResidualBlock(dim, dim, dropout)]
        self.attentions = []
        for _ in range(num_layers):
            self.attentions.append(QwenImageAttentionBlock(dim))
            self.resnets.append(QwenImageResidualBlock(dim, dim, dropout))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        for attention, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            x = attention(x)
            x = resnet(x)
        return x


class QwenImageResample(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.resample_conv = None
        self.time_conv = None
        if mode == "upsample2d":
            self.resample_conv = Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        elif mode == "upsample3d":
            self.resample_conv = Conv2d(dim, dim // 2, kernel_size=3, padding=1)
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample_conv = Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
        elif mode == "downsample3d":
            self.resample_conv = Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
            # The upstream plain encode path keeps images on the 2D branch; the
            # temporal conv is only needed for future cached streaming support.
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        if self.mode == "upsample3d":
            if self.time_conv is None:
                raise RuntimeError("Qwen-Image upsample3d requires a temporal conv")
            x = self.time_conv(x)
            batch_size, frames, height, width, doubled_channels = (
                int(size) for size in x.shape
            )
            channels = doubled_channels // 2
            x = x.reshape(batch_size, frames, height, width, 2, channels)
            x = x.transpose(0, 1, 4, 2, 3, 5).reshape(
                batch_size,
                frames * 2,
                height,
                width,
                channels,
            )
        batch_size, frames, height, width, channels = (int(size) for size in x.shape)
        hidden = x.reshape(batch_size * frames, height, width, channels)
        if self.mode in {"upsample2d", "upsample3d"}:
            hidden = mx.repeat(mx.repeat(hidden, 2, axis=1), 2, axis=2)
        elif self.mode in {"downsample2d", "downsample3d"}:
            hidden = mx.pad(hidden, [(0, 0), (0, 1), (0, 1), (0, 0)])
        if self.resample_conv is None:
            raise RuntimeError("Qwen-Image resample mode requires a resample conv")
        hidden = self.resample_conv(hidden)
        new_height = int(hidden.shape[1])
        new_width = int(hidden.shape[2])
        new_channels = int(hidden.shape[3])
        return hidden.reshape(batch_size, frames, new_height, new_width, new_channels)


class QwenImageUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_res_blocks: int,
        dropout: float,
        upsample_mode: str | None,
    ) -> None:
        super().__init__()
        self.resnets: list[QwenImageResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.upsamplers = (
            [QwenImageResample(out_dim, upsample_mode)]
            if upsample_mode is not None
            else []
        )

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x


class QwenImageEncoder3D(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        dims = [config.base_dim * value for value in [1, *config.dim_mult]]
        current_scale = 1.0
        current_dim = dims[0]
        self.conv_in = CausalConv3d(config.input_channels, current_dim, 3, padding=1)
        self.down_blocks: list[
            QwenImageResidualBlock | QwenImageAttentionBlock | QwenImageResample
        ] = []
        for index, out_dim in enumerate(dims[1:]):
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(
                    QwenImageResidualBlock(current_dim, out_dim, config.dropout)
                )
                if current_scale in config.attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                current_dim = out_dim
            if index != len(config.dim_mult) - 1:
                downsample_mode = (
                    "downsample3d"
                    if config.temperal_downsample[index]
                    else "downsample2d"
                )
                self.down_blocks.append(QwenImageResample(current_dim, downsample_mode))
                current_scale /= 2.0
        self.mid_block = QwenImageMidBlock(
            current_dim,
            config.dropout,
            num_layers=1,
        )
        self.norm_out = QwenImageRMSNorm(current_dim)
        self.conv_out = CausalConv3d(current_dim, config.z_dim * 2, 3, padding=1)

    def encode(self, images: mx.array) -> mx.array:
        hidden = images.transpose(0, 2, 3, 4, 1)
        hidden = self.conv_in(hidden)
        for layer in self.down_blocks:
            hidden = layer(hidden)
        hidden = self.mid_block(hidden)
        hidden = silu(self.norm_out(hidden))
        encoded = self.conv_out(hidden)
        return encoded.transpose(0, 4, 1, 2, 3)


class QwenImageDecoder3D(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        dims = [
            config.base_dim * value
            for value in [config.dim_mult[-1], *reversed(config.dim_mult)]
        ]
        self.conv_in = CausalConv3d(config.z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock(dims[0], config.dropout)
        self.up_blocks: list[QwenImageUpBlock] = []
        for index, (in_dim, out_dim) in enumerate(
            zip(dims[:-1], dims[1:], strict=True)
        ):
            if index > 0:
                in_dim = in_dim // 2
            upsample_mode = None
            if index != len(config.dim_mult) - 1:
                upsample_mode = (
                    "upsample3d" if config.temporal_upsample[index] else "upsample2d"
                )
            self.up_blocks.append(
                QwenImageUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=config.num_res_blocks,
                    dropout=config.dropout,
                    upsample_mode=upsample_mode,
                )
            )
        self.norm_out = QwenImageRMSNorm(out_dim)
        self.conv_out = CausalConv3d(out_dim, config.input_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.conv_in(x)
        hidden = self.mid_block(hidden)
        for up_block in self.up_blocks:
            hidden = up_block(hidden)
        hidden = silu(self.norm_out(hidden))
        return self.conv_out(hidden)


class QwenImageAutoencoderDecoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = QwenImageEncoder3D(config)
        self.quant_conv = CausalConv3d(config.z_dim * 2, config.z_dim * 2, 1)
        self.post_quant_conv = CausalConv3d(config.z_dim, config.z_dim, 1)
        self.decoder = QwenImageDecoder3D(config)
        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def enable_tiling(
        self,
        *,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_stride_height: int | None = None,
        tile_sample_stride_width: int | None = None,
    ) -> None:
        self.use_tiling = True
        if tile_sample_min_height is not None:
            self.tile_sample_min_height = int(tile_sample_min_height)
        if tile_sample_min_width is not None:
            self.tile_sample_min_width = int(tile_sample_min_width)
        if tile_sample_stride_height is not None:
            self.tile_sample_stride_height = int(tile_sample_stride_height)
        if tile_sample_stride_width is not None:
            self.tile_sample_stride_width = int(tile_sample_stride_width)

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def _encode_monolithic(self, images: mx.array) -> mx.array:
        encoded = self.encoder.encode(images).transpose(0, 2, 3, 4, 1)
        moments = self.quant_conv(encoded).transpose(0, 4, 1, 2, 3)
        mean, _logvar = mx.split(moments, 2, axis=1)
        return mean

    def _decode_monolithic(self, latents: mx.array) -> mx.array:
        hidden = latents.transpose(0, 2, 3, 4, 1)
        hidden = self.post_quant_conv(hidden)
        decoded = self.decoder(hidden)
        decoded = mx.clip(decoded, -1.0, 1.0)
        return decoded.transpose(0, 4, 1, 2, 3)

    def _should_tile_encode(self, images: mx.array) -> bool:
        return self.use_tiling and (
            int(images.shape[-1]) > self.tile_sample_min_width
            or int(images.shape[-2]) > self.tile_sample_min_height
        )

    def _should_tile_decode(self, latents: mx.array) -> bool:
        tile_latent_min_height = self.tile_sample_min_height // self.config.scale_factor
        tile_latent_min_width = self.tile_sample_min_width // self.config.scale_factor
        return self.use_tiling and (
            int(latents.shape[-1]) > tile_latent_min_width
            or int(latents.shape[-2]) > tile_latent_min_height
        )

    def _blend_v(
        self, above: mx.array, current: mx.array, blend_extent: int
    ) -> mx.array:
        extent = min(int(above.shape[-2]), int(current.shape[-2]), int(blend_extent))
        if extent <= 0:
            return current
        weights = mx.arange(extent, dtype=current.dtype).reshape(
            1, 1, 1, extent, 1
        ) / float(extent)
        top = (
            above[:, :, :, -extent:, :] * (1.0 - weights)
            + current[:, :, :, :extent, :] * weights
        )
        return mx.concatenate([top, current[:, :, :, extent:, :]], axis=-2)

    def _blend_h(
        self, left: mx.array, current: mx.array, blend_extent: int
    ) -> mx.array:
        extent = min(int(left.shape[-1]), int(current.shape[-1]), int(blend_extent))
        if extent <= 0:
            return current
        weights = mx.arange(extent, dtype=current.dtype).reshape(
            1, 1, 1, 1, extent
        ) / float(extent)
        overlap = (
            left[:, :, :, :, -extent:] * (1.0 - weights)
            + current[:, :, :, :, :extent] * weights
        )
        return mx.concatenate([overlap, current[:, :, :, :, extent:]], axis=-1)

    def encode(self, images: mx.array) -> mx.array:
        if self.use_slicing and int(images.shape[0]) > 1:
            parts = [
                self.encode(images[index : index + 1])
                for index in range(int(images.shape[0]))
            ]
            return mx.concatenate(parts, axis=0)
        if self._should_tile_encode(images):
            return self.tiled_encode(images)
        return self._encode_monolithic(images)

    def decode(self, latents: mx.array) -> mx.array:
        if self.use_slicing and int(latents.shape[0]) > 1:
            parts = [
                self.decode(latents[index : index + 1])
                for index in range(int(latents.shape[0]))
            ]
            return mx.concatenate(parts, axis=0)
        if self._should_tile_decode(latents):
            return self.tiled_decode(latents)
        return self._decode_monolithic(latents)

    def tiled_encode(self, images: mx.array) -> mx.array:
        _, _, _, height, width = (int(size) for size in images.shape)
        latent_height = height // self.config.scale_factor
        latent_width = width // self.config.scale_factor
        tile_latent_min_height = self.tile_sample_min_height // self.config.scale_factor
        tile_latent_min_width = self.tile_sample_min_width // self.config.scale_factor
        tile_latent_stride_height = (
            self.tile_sample_stride_height // self.config.scale_factor
        )
        tile_latent_stride_width = (
            self.tile_sample_stride_width // self.config.scale_factor
        )
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        rows: list[list[mx.array]] = []
        for top in range(0, height, self.tile_sample_stride_height):
            row: list[mx.array] = []
            for left in range(0, width, self.tile_sample_stride_width):
                tile = images[
                    :,
                    :,
                    :,
                    top : top + self.tile_sample_min_height,
                    left : left + self.tile_sample_min_width,
                ]
                row.append(self._encode_monolithic(tile))
            rows.append(row)

        result_rows: list[mx.array] = []
        for row_index, row in enumerate(rows):
            stitched_row: list[mx.array] = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend_v(
                        rows[row_index - 1][column_index],
                        tile,
                        blend_height,
                    )
                    row[column_index] = tile
                if column_index > 0:
                    tile = self._blend_h(row[column_index - 1], tile, blend_width)
                    row[column_index] = tile
                stitched_row.append(
                    tile[
                        :,
                        :,
                        :,
                        :tile_latent_stride_height,
                        :tile_latent_stride_width,
                    ]
                )
            result_rows.append(mx.concatenate(stitched_row, axis=-1))

        encoded = mx.concatenate(result_rows, axis=-2)
        return encoded[:, :, :, :latent_height, :latent_width]

    def tiled_decode(self, latents: mx.array) -> mx.array:
        _, _, _, height, width = (int(size) for size in latents.shape)
        sample_height = height * self.config.scale_factor
        sample_width = width * self.config.scale_factor
        tile_latent_min_height = self.tile_sample_min_height // self.config.scale_factor
        tile_latent_min_width = self.tile_sample_min_width // self.config.scale_factor
        tile_latent_stride_height = (
            self.tile_sample_stride_height // self.config.scale_factor
        )
        tile_latent_stride_width = (
            self.tile_sample_stride_width // self.config.scale_factor
        )
        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        rows: list[list[mx.array]] = []
        for top in range(0, height, tile_latent_stride_height):
            row: list[mx.array] = []
            for left in range(0, width, tile_latent_stride_width):
                tile = latents[
                    :,
                    :,
                    :,
                    top : top + tile_latent_min_height,
                    left : left + tile_latent_min_width,
                ]
                row.append(self._decode_monolithic(tile))
            rows.append(row)

        result_rows: list[mx.array] = []
        for row_index, row in enumerate(rows):
            stitched_row: list[mx.array] = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend_v(
                        rows[row_index - 1][column_index],
                        tile,
                        blend_height,
                    )
                    row[column_index] = tile
                if column_index > 0:
                    tile = self._blend_h(row[column_index - 1], tile, blend_width)
                    row[column_index] = tile
                stitched_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : self.tile_sample_stride_height,
                        : self.tile_sample_stride_width,
                    ]
                )
            result_rows.append(mx.concatenate(stitched_row, axis=-1))

        decoded = mx.concatenate(result_rows, axis=-2)
        return decoded[:, :, :, :sample_height, :sample_width]

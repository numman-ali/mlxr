from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlxr.core.runtime import TraceRecorder, mlx_memory_snapshot
from PIL import Image

from . import _nn_compat as nn
from .debug import debug_trace_enabled, debug_trace_sync_enabled
from .generation import GeneratedImage, ImageGenerator
from .prompt_encoding import PromptEncodingResult

_SEQ_MULTI_OF = 32
_ADALN_EMBED_DIM = 256
_FREQUENCY_EMBEDDING_SIZE = 256
_MAX_PERIOD = 10_000
_BASE_IMAGE_SEQ_LEN = 256
_MAX_IMAGE_SEQ_LEN = 4_096
_BASE_SHIFT = 0.5
_MAX_SHIFT = 1.15


@dataclass(frozen=True, slots=True)
class ZImageTransformerConfig:
    all_patch_size: tuple[int, ...] = (2,)
    all_f_patch_size: tuple[int, ...] = (1,)
    in_channels: int = 16
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    norm_eps: float = 1.0e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (1536, 512, 512)

    @classmethod
    def from_path(cls, config_path: Path) -> ZImageTransformerConfig:
        raw = json.loads(config_path.read_text("utf-8"))
        return cls(
            all_patch_size=tuple(int(value) for value in raw["all_patch_size"]),
            all_f_patch_size=tuple(int(value) for value in raw["all_f_patch_size"]),
            in_channels=int(raw["in_channels"]),
            dim=int(raw["dim"]),
            n_layers=int(raw["n_layers"]),
            n_refiner_layers=int(raw["n_refiner_layers"]),
            n_heads=int(raw["n_heads"]),
            n_kv_heads=int(raw["n_kv_heads"]),
            norm_eps=float(raw["norm_eps"]),
            qk_norm=bool(raw.get("qk_norm", True)),
            cap_feat_dim=int(raw["cap_feat_dim"]),
            rope_theta=float(raw["rope_theta"]),
            t_scale=float(raw["t_scale"]),
            axes_dims=tuple(int(value) for value in raw["axes_dims"]),
            axes_lens=tuple(int(value) for value in raw["axes_lens"]),
        )


@dataclass(frozen=True, slots=True)
class AutoencoderConfig:
    in_channels: int = 3
    out_channels: int = 3
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    latent_channels: int = 16
    norm_num_groups: int = 32
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    force_upcast: bool = True
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False

    @classmethod
    def from_path(cls, config_path: Path) -> AutoencoderConfig:
        raw = json.loads(config_path.read_text("utf-8"))
        shift_factor = raw.get("shift_factor")
        return cls(
            in_channels=int(raw["in_channels"]),
            out_channels=int(raw["out_channels"]),
            block_out_channels=tuple(int(value) for value in raw["block_out_channels"]),
            layers_per_block=int(raw["layers_per_block"]),
            latent_channels=int(raw["latent_channels"]),
            norm_num_groups=int(raw["norm_num_groups"]),
            scaling_factor=float(raw["scaling_factor"]),
            shift_factor=float(shift_factor if shift_factor is not None else 0.0),
            force_upcast=bool(raw.get("force_upcast", True)),
            use_quant_conv=bool(raw.get("use_quant_conv", False)),
            use_post_quant_conv=bool(raw.get("use_post_quant_conv", False)),
        )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    num_train_timesteps: int = 1000
    shift: float = 3.0
    use_dynamic_shifting: bool = False

    @classmethod
    def from_path(cls, config_path: Path) -> SchedulerConfig:
        raw = json.loads(config_path.read_text("utf-8"))
        return cls(
            num_train_timesteps=int(raw["num_train_timesteps"]),
            shift=float(raw["shift"]),
            use_dynamic_shifting=bool(raw.get("use_dynamic_shifting", False)),
        )


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return value


def _resolved_group_count(channels: int, requested_groups: int) -> int:
    group_count = min(channels, requested_groups)
    while channels % group_count != 0:
        group_count -= 1
    if group_count < 1:
        raise ValueError(f"Could not resolve a valid group count for {channels=}")
    return group_count


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _frequency_embedding(
    timesteps: mx.array, dim: int, *, max_period: int = _MAX_PERIOD
) -> mx.array:
    timesteps = timesteps.astype(mx.float32)
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = timesteps[:, None] * freqs[None, :]
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        embedding = mx.concatenate(
            [embedding, mx.zeros((embedding.shape[0], 1), dtype=embedding.dtype)],
            axis=-1,
        )
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        out_size: int,
        *,
        mid_size: int | None = None,
        frequency_embedding_size: int = _FREQUENCY_EMBEDDING_SIZE,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        hidden_size = out_size if mid_size is None else mid_size
        self.mlp_in = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp_act = nn.SiLU()
        self.mlp_out = nn.Linear(hidden_size, out_size, bias=True)

    def __call__(self, timesteps: mx.array) -> mx.array:
        embedded = _frequency_embedding(
            timesteps, self.frequency_embedding_size, max_period=_MAX_PERIOD
        )
        hidden = self.mlp_in(embedded)
        hidden = self.mlp_act(hidden)
        return self.mlp_out(hidden)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight.astype(x.dtype)
        return mx.fast.rms_norm(x, weight, self.eps)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(_silu(self.w1(x)) * self.w3(x))


class RopeEmbedder:
    def __init__(
        self,
        *,
        theta: float,
        axes_dims: tuple[int, ...],
        axes_lens: tuple[int, ...],
    ) -> None:
        if len(axes_dims) != len(axes_lens):
            raise ValueError("axes_dims and axes_lens must have the same length")
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self._cos_tables: tuple[mx.array, ...] | None = None
        self._sin_tables: tuple[mx.array, ...] | None = None

    def __call__(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        if ids.ndim == 3:
            batch, length, axes = (int(size) for size in ids.shape)
            cos, sin = self(ids.reshape(batch * length, axes))
            return (
                cos.reshape(batch, length, -1),
                sin.reshape(batch, length, -1),
            )
        if ids.ndim != 2 or int(ids.shape[-1]) != len(self.axes_dims):
            raise ValueError(
                "RopeEmbedder expects ids shaped (sequence, axes) or (batch, sequence, axes)"
            )
        if self._cos_tables is None or self._sin_tables is None:
            self._cos_tables, self._sin_tables = self._precompute_tables()
        cos_parts: list[mx.array] = []
        sin_parts: list[mx.array] = []
        for axis_index, (cos_table, sin_table) in enumerate(
            zip(self._cos_tables, self._sin_tables, strict=True)
        ):
            positions = ids[:, axis_index]
            cos_parts.append(mx.take(cos_table, positions, axis=0))
            sin_parts.append(mx.take(sin_table, positions, axis=0))
        return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)

    def _precompute_tables(self) -> tuple[tuple[mx.array, ...], tuple[mx.array, ...]]:
        cos_tables: list[mx.array] = []
        sin_tables: list[mx.array] = []
        for axis_dim, axis_len in zip(self.axes_dims, self.axes_lens, strict=True):
            freqs = 1.0 / (
                self.theta
                ** (mx.arange(0, axis_dim, 2, dtype=mx.float32) / float(axis_dim))
            )
            steps = mx.arange(axis_len, dtype=mx.float32)
            angles = steps[:, None] * freqs[None, :]
            cos_tables.append(mx.cos(angles))
            sin_tables.append(mx.sin(angles))
        return tuple(cos_tables), tuple(sin_tables)


def _apply_rotary_emb(
    x: mx.array, rotary: tuple[mx.array, mx.array] | None
) -> mx.array:
    if rotary is None:
        return x
    cos, sin = rotary
    cos = mx.expand_dims(cos.astype(x.dtype), axis=1)
    sin = mx.expand_dims(sin.astype(x.dtype), axis=1)
    reshaped = x.reshape(*x.shape[:-1], int(x.shape[-1]) // 2, 2)
    first = reshaped[..., 0]
    second = reshaped[..., 1]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return mx.stack([rotated_first, rotated_second], axis=-1).reshape(x.shape)


class ZImageAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        qk_norm: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = [nn.Linear(n_heads * self.head_dim, dim, bias=False)]
        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array | None = None,
        rotary: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.to_q(hidden_states).reshape(
            batch_size, sequence_length, self.n_heads, self.head_dim
        )
        key = self.to_k(hidden_states).reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        )
        value = self.to_v(hidden_states).reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        )
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)
        query = _apply_rotary_emb(query, rotary)
        key = _apply_rotary_emb(key, rotary)
        if self.repeats > 1:
            key = mx.repeat(key, self.repeats, axis=1)
            value = mx.repeat(value, self.repeats, axis=1)

        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
        hidden = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask=mask,
        )
        hidden = hidden.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.n_heads * self.head_dim
        )
        return self.to_out[0](hidden)


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool,
    ) -> None:
        del layer_id
        super().__init__()
        self.dim = dim
        self.modulation = modulation
        self.attention = ZImageAttention(
            dim,
            n_heads,
            n_kv_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
        )
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = (
            [nn.Linear(min(dim, _ADALN_EMBED_DIM), 4 * dim, bias=True)]
            if modulation
            else None
        )

    def __call__(
        self,
        x: mx.array,
        attention_mask: mx.array,
        rotary: tuple[mx.array, mx.array],
        adaln_input: mx.array | None = None,
    ) -> mx.array:
        if self.modulation:
            if adaln_input is None or self.adaLN_modulation is None:
                raise ValueError("Modulated transformer blocks require adaLN input")
            modulation = self.adaLN_modulation[0](adaln_input)
            modulation = mx.expand_dims(modulation, axis=1)
            scale_msa = 1.0 + modulation[:, :, 0 : self.dim]
            gate_msa = mx.tanh(modulation[:, :, self.dim : 2 * self.dim])
            scale_mlp = 1.0 + modulation[:, :, 2 * self.dim : 3 * self.dim]
            gate_mlp = mx.tanh(modulation[:, :, 3 * self.dim : 4 * self.dim])
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attention_mask,
                rotary=rotary,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            mlp_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            return x + gate_mlp * self.ffn_norm2(mlp_out)

        attn_out = self.attention(
            self.attention_norm1(x),
            attention_mask=attention_mask,
            rotary=rotary,
        )
        x = x + self.attention_norm2(attn_out)
        return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, eps=1.0e-6, affine=False, bias=False
        )
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_act = nn.SiLU()
        self.adaln_proj = nn.Linear(
            min(hidden_size, _ADALN_EMBED_DIM), hidden_size, bias=True
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaln_proj(self.adaln_act(c))
        hidden = self.norm_final(x) * mx.expand_dims(scale.astype(x.dtype), axis=1)
        return self.linear(hidden)


class ZImageTransformer2DModel(nn.Module):
    def __init__(self, config: ZImageTransformerConfig) -> None:
        super().__init__()
        if len(config.all_patch_size) != 1 or len(config.all_f_patch_size) != 1:
            raise ValueError(
                "The native MLX backend currently supports one patch-size pair per checkpoint"
            )
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.patch_size = config.all_patch_size[0]
        self.f_patch_size = config.all_f_patch_size[0]
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.t_scale = config.t_scale
        self.x_embedder = nn.Linear(
            self.f_patch_size * self.patch_size * self.patch_size * self.in_channels,
            config.dim,
            bias=True,
        )
        self.final_layer = FinalLayer(
            config.dim,
            self.patch_size * self.patch_size * self.f_patch_size * self.out_channels,
        )
        self.noise_refiner = [
            ZImageTransformerBlock(
                layer_id=1000 + layer_id,
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                norm_eps=config.norm_eps,
                qk_norm=config.qk_norm,
                modulation=True,
            )
            for layer_id in range(config.n_refiner_layers)
        ]
        self.context_refiner = [
            ZImageTransformerBlock(
                layer_id=layer_id,
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                norm_eps=config.norm_eps,
                qk_norm=config.qk_norm,
                modulation=False,
            )
            for layer_id in range(config.n_refiner_layers)
        ]
        self.t_embedder = TimestepEmbedder(
            min(config.dim, _ADALN_EMBED_DIM),
            mid_size=1024,
        )
        self.cap_norm = RMSNorm(config.cap_feat_dim, eps=config.norm_eps)
        self.cap_proj = nn.Linear(config.cap_feat_dim, config.dim, bias=True)
        self.x_pad_token = mx.zeros((1, config.dim), dtype=mx.float32)
        self.cap_pad_token = mx.zeros((1, config.dim), dtype=mx.float32)
        self.layers = [
            ZImageTransformerBlock(
                layer_id=layer_id,
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                norm_eps=config.norm_eps,
                qk_norm=config.qk_norm,
                modulation=True,
            )
            for layer_id in range(config.n_layers)
        ]
        head_dim = config.dim // config.n_heads
        if head_dim != sum(config.axes_dims):
            raise ValueError(
                "Transformer head dimension must equal the configured RoPE axes total"
            )
        self.rope_embedder = RopeEmbedder(
            theta=config.rope_theta,
            axes_dims=config.axes_dims,
            axes_lens=config.axes_lens,
        )

    @property
    def dtype(self) -> mx.Dtype:
        return self.x_embedder.weight.dtype

    @staticmethod
    def create_coordinate_grid(
        size: tuple[int, int, int], *, start: tuple[int, int, int]
    ) -> mx.array:
        axes = [
            np.arange(offset, offset + span, dtype=np.int32)
            for offset, span in zip(start, size, strict=True)
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        return mx.array(np.stack(grids, axis=-1), dtype=mx.int32)

    def patchify_and_embed(
        self,
        all_image: tuple[mx.array, ...],
        all_cap_feats: tuple[mx.array, ...],
    ) -> tuple[
        tuple[mx.array, ...],
        tuple[mx.array, ...],
        tuple[tuple[int, int, int], ...],
        tuple[mx.array, ...],
        tuple[mx.array, ...],
    ]:
        p_h = self.patch_size
        p_w = self.patch_size
        p_f = self.f_patch_size
        image_out: list[mx.array] = []
        cap_out: list[mx.array] = []
        image_sizes: list[tuple[int, int, int]] = []
        image_pos_ids: list[mx.array] = []
        cap_pos_ids: list[mx.array] = []

        for image, cap_feat in zip(all_image, all_cap_feats, strict=True):
            if image.ndim != 4:
                raise ValueError(
                    "Expected latent samples shaped (channels, frames, height, width)"
                )
            if cap_feat.ndim != 2:
                raise ValueError(
                    "Expected prompt embeddings shaped (tokens, hidden_size)"
                )
            cap_ori_len = int(cap_feat.shape[0])
            if cap_ori_len <= 0:
                raise ValueError("Prompt embeddings must contain at least one token")
            cap_padding_len = (-cap_ori_len) % _SEQ_MULTI_OF
            cap_pos_grid = self.create_coordinate_grid(
                (cap_ori_len + cap_padding_len, 1, 1), start=(1, 0, 0)
            ).reshape(cap_ori_len + cap_padding_len, 3)
            if cap_padding_len > 0:
                cap_feat = mx.concatenate(
                    [
                        cap_feat,
                        mx.repeat(cap_feat[-1:, :], cap_padding_len, axis=0),
                    ],
                    axis=0,
                )
            cap_out.append(cap_feat)
            cap_pos_ids_out = cap_pos_grid

            channels, frames, height, width = (int(size) for size in image.shape)
            image_sizes.append((frames, height, width))
            frame_tokens = frames // p_f
            height_tokens = height // p_h
            width_tokens = width // p_w
            image = image.reshape(
                channels,
                frame_tokens,
                p_f,
                height_tokens,
                p_h,
                width_tokens,
                p_w,
            )
            image = image.transpose(1, 3, 5, 2, 4, 6, 0).reshape(
                frame_tokens * height_tokens * width_tokens,
                p_f * p_h * p_w * channels,
            )
            image_ori_len = int(image.shape[0])
            image_padding_len = (-image_ori_len) % _SEQ_MULTI_OF
            image_ori_pos_ids = self.create_coordinate_grid(
                (frame_tokens, height_tokens, width_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
            ).reshape(image_ori_len, 3)
            if image_padding_len > 0:
                image = mx.concatenate(
                    [
                        image,
                        mx.repeat(image[-1:, :], image_padding_len, axis=0),
                    ],
                    axis=0,
                )
                pad_pos = mx.repeat(
                    self.create_coordinate_grid((1, 1, 1), start=(0, 0, 0)).reshape(
                        1, 3
                    ),
                    image_padding_len,
                    axis=0,
                )
                image_pos_ids_out = mx.concatenate([image_ori_pos_ids, pad_pos], axis=0)
            else:
                image_pos_ids_out = image_ori_pos_ids
            image_out.append(image)
            image_pos_ids.append(image_pos_ids_out)
            cap_pos_ids.append(cap_pos_ids_out)

        return (
            tuple(image_out),
            tuple(cap_out),
            tuple(image_sizes),
            tuple(image_pos_ids),
            tuple(cap_pos_ids),
        )

    def unpatchify(
        self,
        patched: tuple[mx.array, ...],
        sizes: tuple[tuple[int, int, int], ...],
    ) -> tuple[mx.array, ...]:
        p_h = self.patch_size
        p_w = self.patch_size
        p_f = self.f_patch_size
        restored: list[mx.array] = []
        for tokens, (frames, height, width) in zip(patched, sizes, strict=True):
            original_len = (frames // p_f) * (height // p_h) * (width // p_w)
            tokens = tokens[:original_len]
            image = tokens.reshape(
                frames // p_f,
                height // p_h,
                width // p_w,
                p_f,
                p_h,
                p_w,
                self.out_channels,
            )
            image = image.transpose(6, 0, 3, 1, 4, 2, 5).reshape(
                self.out_channels, frames, height, width
            )
            restored.append(image)
        return tuple(restored)

    def __call__(
        self,
        x: tuple[mx.array, ...],
        timesteps: mx.array,
        cap_feats: tuple[mx.array, ...],
    ) -> tuple[tuple[mx.array, ...], dict[str, object]]:
        (
            image_tokens,
            cap_tokens,
            image_sizes,
            image_pos_ids,
            cap_pos_ids,
        ) = self.patchify_and_embed(x, cap_feats)
        image_lengths = [int(tokens.shape[0]) for tokens in image_tokens]
        cap_lengths = [int(tokens.shape[0]) for tokens in cap_tokens]

        projected_images: list[mx.array] = []
        for tokens in image_tokens:
            projected = self.x_embedder(tokens)
            projected_images.append(projected)
        adaln_input = self.t_embedder(timesteps * self.t_scale).astype(
            projected_images[0].dtype
        )
        for index, tokens in enumerate(projected_images):
            original_len = (
                (image_sizes[index][0] // self.f_patch_size)
                * (image_sizes[index][1] // self.patch_size)
                * (image_sizes[index][2] // self.patch_size)
            )
            padding_len = int(tokens.shape[0]) - original_len
            if padding_len > 0:
                projected_images[index] = mx.concatenate(
                    [
                        tokens[:original_len],
                        mx.repeat(
                            self.x_pad_token.astype(tokens.dtype), padding_len, axis=0
                        ),
                    ],
                    axis=0,
                )
        image_rotary = [self.rope_embedder(pos_ids) for pos_ids in image_pos_ids]
        image_batch = _pad_sequence(projected_images)
        image_rotary_batch = _pad_rotary(image_rotary)
        image_attention_mask = _sequence_mask(image_lengths)
        for layer in self.noise_refiner:
            image_batch = layer(
                image_batch,
                image_attention_mask,
                image_rotary_batch,
                adaln_input,
            )

        projected_caps: list[mx.array] = []
        for tokens in cap_tokens:
            projected = self.cap_proj(self.cap_norm(tokens))
            projected_caps.append(projected)
        for index, tokens in enumerate(projected_caps):
            padding_len = int(tokens.shape[0]) - int(cap_feats[index].shape[0])
            if padding_len > 0:
                projected_caps[index] = mx.concatenate(
                    [
                        tokens[:-padding_len],
                        mx.repeat(
                            self.cap_pad_token.astype(tokens.dtype), padding_len, axis=0
                        ),
                    ],
                    axis=0,
                )
        cap_rotary = [self.rope_embedder(pos_ids) for pos_ids in cap_pos_ids]
        cap_batch = _pad_sequence(projected_caps)
        cap_rotary_batch = _pad_rotary(cap_rotary)
        cap_attention_mask = _sequence_mask(cap_lengths)
        for layer in self.context_refiner:
            cap_batch = layer(
                cap_batch,
                cap_attention_mask,
                cap_rotary_batch,
                None,
            )

        unified_tokens: list[mx.array] = []
        unified_rotary: list[tuple[mx.array, mx.array]] = []
        unified_lengths: list[int] = []
        for index in range(len(projected_images)):
            image_len = image_lengths[index]
            cap_len = cap_lengths[index]
            unified_tokens.append(
                mx.concatenate(
                    [
                        image_batch[index, :image_len, :],
                        cap_batch[index, :cap_len, :],
                    ],
                    axis=0,
                )
            )
            image_cos, image_sin = image_rotary_batch
            cap_cos, cap_sin = cap_rotary_batch
            unified_rotary.append(
                (
                    mx.concatenate(
                        [image_cos[index, :image_len, :], cap_cos[index, :cap_len, :]],
                        axis=0,
                    ),
                    mx.concatenate(
                        [image_sin[index, :image_len, :], cap_sin[index, :cap_len, :]],
                        axis=0,
                    ),
                )
            )
            unified_lengths.append(image_len + cap_len)

        unified_batch = _pad_sequence(unified_tokens)
        unified_rotary_batch = _pad_rotary(unified_rotary)
        unified_attention_mask = _sequence_mask(unified_lengths)
        for layer in self.layers:
            unified_batch = layer(
                unified_batch,
                unified_attention_mask,
                unified_rotary_batch,
                adaln_input,
            )
        unified_batch = self.final_layer(unified_batch, adaln_input)
        unbound = tuple(
            unified_batch[index, : unified_lengths[index], :]
            for index in range(len(unified_lengths))
        )
        return self.unpatchify(unbound, image_sizes), {}


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        groups: int,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        out_dims = in_channels if out_channels is None else out_channels
        norm1_groups = _resolved_group_count(in_channels, groups)
        norm2_groups = _resolved_group_count(out_dims, groups)
        self.in_channels = in_channels
        self.out_channels = out_dims
        self.norm1 = nn.GroupNorm(
            norm1_groups,
            in_channels,
            eps=eps,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv1 = nn.Conv2d(in_channels, out_dims, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            norm2_groups,
            out_dims,
            eps=eps,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv2 = nn.Conv2d(out_dims, out_dims, kernel_size=3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_dims, kernel_size=1, padding=0)
            if in_channels != out_dims
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.conv1(_silu(self.norm1(x)))
        hidden = self.conv2(_silu(self.norm2(hidden)))
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x)
        return residual + hidden


class Attention2d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        groups: int,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        norm_groups = _resolved_group_count(channels, groups)
        self.group_norm = nn.GroupNorm(
            norm_groups,
            channels,
            eps=eps,
            affine=True,
            pytorch_compatible=True,
        )
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        batch, height, width, channels = (int(size) for size in x.shape)
        residual = x
        hidden = self.group_norm(x).reshape(batch, height * width, channels)
        query = self.to_q(hidden)
        key = self.to_k(hidden)
        value = self.to_v(hidden)
        weights = mx.softmax(
            mx.matmul(query, key.transpose(0, 2, 1)) * (float(channels) ** -0.5),
            axis=-1,
        )
        hidden = mx.matmul(weights, value)
        hidden = self.to_out[0](hidden).reshape(batch, height, width, channels)
        return residual + hidden


class Downsample2D(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_dims = channels if out_channels is None else out_channels
        self.conv = nn.Conv2d(channels, out_dims, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_dims = channels if out_channels is None else out_channels
        self.conv = nn.Conv2d(channels, out_dims, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        upsampled = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        return self.conv(upsampled)


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        resnet_groups: int,
        add_downsample: bool,
    ) -> None:
        super().__init__()
        self.resnets = [
            ResnetBlock2D(
                in_channels if index == 0 else out_channels,
                out_channels,
                groups=resnet_groups,
            )
            for index in range(num_layers)
        ]
        self.downsamplers = (
            [Downsample2D(out_channels, out_channels)] if add_downsample else None
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden = hidden_states
        for resnet in self.resnets:
            hidden = resnet(hidden)
        if self.downsamplers is not None:
            hidden = mx.pad(hidden, [(0, 0), (0, 1), (0, 1), (0, 0)])
            hidden = self.downsamplers[0](hidden)
        return hidden


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        resnet_groups: int,
        add_upsample: bool,
    ) -> None:
        super().__init__()
        self.resnets = [
            ResnetBlock2D(
                in_channels if index == 0 else out_channels,
                out_channels,
                groups=resnet_groups,
            )
            for index in range(num_layers)
        ]
        self.upsamplers = (
            [Upsample2D(out_channels, out_channels)] if add_upsample else None
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden = hidden_states
        for resnet in self.resnets:
            hidden = resnet(hidden)
        if self.upsamplers is not None:
            hidden = self.upsamplers[0](hidden)
        return hidden


class UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, *, resnet_groups: int) -> None:
        super().__init__()
        self.resnets = [
            ResnetBlock2D(channels, channels, groups=resnet_groups),
            ResnetBlock2D(channels, channels, groups=resnet_groups),
        ]
        self.attentions = [Attention2d(channels, groups=resnet_groups)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden = self.resnets[0](hidden_states)
        hidden = self.attentions[0](hidden)
        return self.resnets[1](hidden)


class Encoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(
            config.in_channels,
            config.block_out_channels[0],
            kernel_size=3,
            padding=1,
        )
        self.down_blocks = []
        output_channel = config.block_out_channels[0]
        for index, block_out_channel in enumerate(config.block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            self.down_blocks.append(
                DownEncoderBlock2D(
                    input_channel,
                    output_channel,
                    num_layers=config.layers_per_block,
                    resnet_groups=config.norm_num_groups,
                    add_downsample=index != len(config.block_out_channels) - 1,
                )
            )
        self.mid_block = UNetMidBlock2D(
            config.block_out_channels[-1],
            resnet_groups=config.norm_num_groups,
        )
        out_groups = _resolved_group_count(
            config.block_out_channels[-1], config.norm_num_groups
        )
        self.conv_norm_out = nn.GroupNorm(
            out_groups,
            config.block_out_channels[-1],
            eps=1.0e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            config.block_out_channels[-1],
            2 * config.latent_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.conv_in(x)
        for block in self.down_blocks:
            hidden = block(hidden)
        hidden = self.mid_block(hidden)
        hidden = self.conv_out(self.conv_act(self.conv_norm_out(hidden)))
        return hidden


class Decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(
            config.latent_channels,
            config.block_out_channels[-1],
            kernel_size=3,
            padding=1,
        )
        self.mid_block = UNetMidBlock2D(
            config.block_out_channels[-1],
            resnet_groups=config.norm_num_groups,
        )
        reversed_channels = list(reversed(config.block_out_channels))
        self.up_blocks = []
        output_channel = reversed_channels[0]
        for index, block_out_channel in enumerate(reversed_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            self.up_blocks.append(
                UpDecoderBlock2D(
                    input_channel,
                    output_channel,
                    num_layers=config.layers_per_block + 1,
                    resnet_groups=config.norm_num_groups,
                    add_upsample=index != len(reversed_channels) - 1,
                )
            )
        out_groups = _resolved_group_count(
            config.block_out_channels[0], config.norm_num_groups
        )
        self.conv_norm_out = nn.GroupNorm(
            out_groups,
            config.block_out_channels[0],
            eps=1.0e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            config.block_out_channels[0],
            config.out_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.conv_in(x)
        hidden = self.mid_block(hidden)
        for block in self.up_blocks:
            hidden = block(hidden)
        hidden = self.conv_out(self.conv_act(self.conv_norm_out(hidden)))
        return hidden


class AutoencoderKL(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = (
            nn.Conv2d(
                2 * config.latent_channels,
                2 * config.latent_channels,
                kernel_size=1,
                padding=0,
            )
            if config.use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(
                config.latent_channels,
                config.latent_channels,
                kernel_size=1,
                padding=0,
            )
            if config.use_post_quant_conv
            else None
        )

    @property
    def dtype(self) -> mx.Dtype:
        return self.decoder.conv_in.weight.dtype

    def decode(self, z: mx.array) -> mx.array:
        hidden = mx.transpose(z, (0, 2, 3, 1))
        if self.post_quant_conv is not None:
            hidden = self.post_quant_conv(hidden)
        decoded = self.decoder(hidden)
        return mx.transpose(decoded, (0, 3, 1, 2))


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = {
            "num_train_timesteps": config.num_train_timesteps,
            "shift": config.shift,
            "use_dynamic_shifting": config.use_dynamic_shifting,
            "base_image_seq_len": _BASE_IMAGE_SEQ_LEN,
            "max_image_seq_len": _MAX_IMAGE_SEQ_LEN,
            "base_shift": _BASE_SHIFT,
            "max_shift": _MAX_SHIFT,
        }
        self.num_train_timesteps = config.num_train_timesteps
        self.shift = config.shift
        self.use_dynamic_shifting = config.use_dynamic_shifting
        timesteps = np.linspace(
            1, self.num_train_timesteps, self.num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        sigmas = timesteps / self.num_train_timesteps
        if not self.use_dynamic_shifting:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = mx.array(sigmas * self.num_train_timesteps, dtype=mx.float32)
        self.sigmas = mx.array(sigmas, dtype=mx.float32)
        self.sigma_min = float(sigmas[-1])
        self.sigma_max = float(sigmas[0])
        self._step_index: int | None = None

    def set_timesteps(
        self, num_inference_steps: int, *, mu: float | None = None
    ) -> None:
        timesteps = np.linspace(
            self.sigma_max * self.num_train_timesteps,
            self.sigma_min * self.num_train_timesteps,
            num_inference_steps + 1,
            dtype=np.float32,
        )[:-1]
        sigmas = timesteps / self.num_train_timesteps
        if self.use_dynamic_shifting:
            if mu is None:
                raise ValueError("Dynamic shifting requires mu")
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = mx.array(sigmas * self.num_train_timesteps, dtype=mx.float32)
        self.sigmas = mx.array(
            np.concatenate([sigmas, np.zeros((1,), dtype=np.float32)])
        )
        self._step_index = None

    def step(
        self, model_output: mx.array, timestep: float, sample: mx.array
    ) -> mx.array:
        if self._step_index is None:
            matches = np.where(np.isclose(np.asarray(self.timesteps), timestep))[0]
            if matches.size == 0:
                raise ValueError(f"Unknown scheduler timestep {timestep}")
            self._step_index = int(matches[0])
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        dt = sigma_next - sigma
        prev_sample = sample.astype(mx.float32) + dt * model_output.astype(mx.float32)
        self._step_index += 1
        return prev_sample

    def time_shift(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


@dataclass(slots=True)
class _RuntimeImageGenerator(ImageGenerator):
    transformer_path: Path
    vae_path: Path
    scheduler_path: Path
    _transformer: ZImageTransformer2DModel
    _vae: AutoencoderKL
    _scheduler: FlowMatchEulerDiscreteScheduler

    def generate(
        self,
        *,
        prompt_context: PromptEncodingResult,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        cfg_normalization: float = 0.0,
        cfg_truncation: float = 1.0,
        seed: int | None = None,
    ) -> GeneratedImage:
        if width % _vae_scale(self._vae.config) != 0:
            raise ValueError(
                f"Width must be divisible by {_vae_scale(self._vae.config)} for Z-Image generation"
            )
        if height % _vae_scale(self._vae.config) != 0:
            raise ValueError(
                f"Height must be divisible by {_vae_scale(self._vae.config)} for Z-Image generation"
            )
        positive_embeddings = _embedding_tuple(prompt_context.prompt_embeddings)
        if len(positive_embeddings) != 1:
            raise ValueError(
                "The current Z-Image runtime path only supports one prompt per job"
            )
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            negative = prompt_context.negative_prompt_embeddings
            if negative is None:
                raise ValueError(
                    "guidance_scale > 1.0 requires negative prompt embeddings in the current Z-Image runtime path"
                )
            negative_embeddings = _embedding_tuple(negative)
            if len(negative_embeddings) != 1:
                raise ValueError(
                    "The current Z-Image runtime path only supports one negative prompt per job"
                )
        else:
            negative_embeddings = ()

        resolved_seed = 0 if seed is None else int(seed)
        mx.random.seed(resolved_seed)
        trace_recorder = TraceRecorder(enabled=debug_trace_enabled())
        trace_sync = debug_trace_sync_enabled()

        with trace_recorder.span(
            "zimage.prepare_latents", snapshot=mlx_memory_snapshot
        ):
            latent_height = 2 * (height // _vae_scale(self._vae.config))
            latent_width = 2 * (width // _vae_scale(self._vae.config))
            latents = mx.random.normal(
                (1, self._transformer.in_channels, latent_height, latent_width),
                dtype=mx.float32,
            )
            image_seq_len = (latent_height // self._transformer.patch_size) * (
                latent_width // self._transformer.patch_size
            )
        with trace_recorder.span(
            "zimage.prepare_scheduler",
            attributes={
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": float(guidance_scale),
                "cfg_normalization": float(cfg_normalization),
                "cfg_truncation": float(cfg_truncation),
            },
            snapshot=mlx_memory_snapshot,
        ):
            mu = _calculate_shift(
                image_seq_len,
                base_seq_len=int(self._scheduler.config["base_image_seq_len"]),
                max_seq_len=int(self._scheduler.config["max_image_seq_len"]),
                base_shift=float(self._scheduler.config["base_shift"]),
                max_shift=float(self._scheduler.config["max_shift"]),
            )
            self._scheduler.sigma_min = 0.0
            self._scheduler.set_timesteps(num_inference_steps, mu=mu)
            timesteps = np.asarray(self._scheduler.timesteps)

        for index, timestep_value in enumerate(timesteps):
            if index == len(timesteps) - 1 and float(timestep_value) == 0.0:
                continue
            timestep = mx.array(
                [(1000.0 - float(timestep_value)) / 1000.0], dtype=mx.float32
            )
            timestep_normalized = float(timestep[0].item())
            current_guidance_scale = float(guidance_scale)
            if (
                do_classifier_free_guidance
                and float(cfg_truncation) <= 1.0
                and timestep_normalized > float(cfg_truncation)
            ):
                current_guidance_scale = 0.0
            apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0.0
            step_context = (
                trace_recorder.span(
                    "zimage.denoise.step",
                    attributes={
                        "step_index": index + 1,
                        "total_steps": len(timesteps),
                        "timestep": round(float(timestep_value), 6),
                        "t_normalized": round(timestep_normalized, 6),
                        "cfg_applied": apply_cfg,
                        "guidance_scale": current_guidance_scale,
                    },
                    snapshot=mlx_memory_snapshot,
                    sync=(lambda: mx.eval(latents)) if trace_sync else None,
                )
                if trace_recorder.enabled
                else nullcontext()
            )
            with step_context:
                if apply_cfg:
                    latent_model_input = mx.concatenate([latents, latents], axis=0)
                    prompt_embeddings = positive_embeddings + negative_embeddings
                    timestep_model_input = mx.concatenate([timestep, timestep], axis=0)
                else:
                    latent_model_input = latents
                    prompt_embeddings = positive_embeddings
                    timestep_model_input = timestep
                latent_model_input = mx.expand_dims(latent_model_input, axis=2)
                latent_list = tuple(
                    latent_model_input[sample_index]
                    for sample_index in range(int(latent_model_input.shape[0]))
                )
                model_out = self._transformer(
                    latent_list,
                    timestep_model_input,
                    prompt_embeddings,
                )[0]
                noise_pred = mx.stack(list(model_out), axis=0).astype(mx.float32)
                if apply_cfg:
                    pos = noise_pred[:1]
                    neg = noise_pred[1:]
                    noise_pred = pos + current_guidance_scale * (pos - neg)
                    if cfg_normalization > 0.0:
                        noise_pred = _apply_cfg_normalization(
                            positive=pos,
                            guided=noise_pred,
                            limit=float(cfg_normalization),
                        )
                noise_pred = -mx.squeeze(noise_pred, axis=2)
                latents = self._scheduler.step(
                    noise_pred,
                    float(timestep_value),
                    latents,
                )

        with trace_recorder.span(
            "zimage.decode_vae",
            snapshot=mlx_memory_snapshot,
            sync=(lambda: mx.eval(decoded)) if trace_sync else None,
        ):
            latents = (
                latents.astype(self._vae.dtype) / self._vae.config.scaling_factor
            ) + (self._vae.config.shift_factor or 0.0)
            decoded = self._vae.decode(latents).astype(mx.float32)
        pixels = _postprocess_image(decoded[0])
        metadata: dict[str, object] = {
            "width": int(pixels.shape[1]),
            "height": int(pixels.shape[0]),
            "num_inference_steps": num_inference_steps,
            "guidance_scale": float(guidance_scale),
            "cfg_normalization": float(cfg_normalization),
            "cfg_truncation": float(cfg_truncation),
            "family_runtime": "z_image_native_mlx",
        }
        if trace_recorder.enabled:
            metadata["trace"] = trace_recorder.to_metadata()
        return GeneratedImage(
            pixels=pixels,
            seed=resolved_seed,
            backend="native_mlx_z_image",
            prompt_signature=_prompt_signature(prompt_context.prompt_text),
            metadata=metadata,
        )

    def close(self) -> None:
        return None


def create_image_generator(
    *,
    transformer_path: Path,
    vae_path: Path,
    scheduler_path: Path,
) -> ImageGenerator:
    transformer = load_local_zimage_transformer(transformer_path)
    vae = load_local_autoencoder(vae_path)
    scheduler = load_local_scheduler(scheduler_path)
    return _RuntimeImageGenerator(
        transformer_path=transformer_path,
        vae_path=vae_path,
        scheduler_path=scheduler_path,
        _transformer=transformer,
        _vae=vae,
        _scheduler=scheduler,
    )


def encode_png_image(image: GeneratedImage, output_path: Path) -> None:
    Image.fromarray(image.pixels).save(output_path, format="PNG")


def encode_jpg_image(image: GeneratedImage, output_path: Path) -> None:
    Image.fromarray(image.pixels).save(output_path, format="JPEG", quality=95)


def load_local_zimage_transformer(component_path: Path) -> ZImageTransformer2DModel:
    config = ZImageTransformerConfig.from_path(component_path / "config.json")
    model = ZImageTransformer2DModel(config)
    weights = list(
        _load_component_weights(
            component_path,
            alias_map={
                "all_x_embedder.2-1.": "x_embedder.",
                "all_final_layer.2-1.": "final_layer.",
                "t_embedder.mlp.0.": "t_embedder.mlp_in.",
                "t_embedder.mlp.2.": "t_embedder.mlp_out.",
                "cap_embedder.0.": "cap_norm.",
                "cap_embedder.1.": "cap_proj.",
                "final_layer.adaLN_modulation.1.": "final_layer.adaln_proj.",
            },
        )
    )
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def load_local_autoencoder(component_path: Path) -> AutoencoderKL:
    config = AutoencoderConfig.from_path(component_path / "config.json")
    model = AutoencoderKL(config)
    weight_dtype = mx.float32 if config.force_upcast else None
    weights = list(_load_component_weights(component_path, weight_dtype=weight_dtype))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def load_local_scheduler(component_path: Path) -> FlowMatchEulerDiscreteScheduler:
    config = SchedulerConfig.from_path(component_path / "scheduler_config.json")
    return FlowMatchEulerDiscreteScheduler(config)


def _load_component_weights(
    component_path: Path,
    *,
    alias_map: dict[str, str] | None = None,
    weight_dtype: mx.Dtype | None = None,
) -> list[tuple[str, mx.array]]:
    items: list[tuple[str, mx.array]] = []
    for weight_file in _weight_files(component_path):
        shard = mx.load(str(weight_file))
        if not isinstance(shard, dict):
            raise RuntimeError(
                f"Weight shard '{weight_file}' did not load into a weight mapping"
            )
        for key, value in shard.items():
            normalized_key = str(key)
            if alias_map is not None:
                changed = True
                while changed:
                    changed = False
                    for source, target in alias_map.items():
                        if normalized_key.startswith(source):
                            normalized_key = target + normalized_key[len(source) :]
                            changed = True
                            break
            if normalized_key.endswith(".weight") and getattr(value, "ndim", None) == 4:
                value = mx.transpose(value, (0, 2, 3, 1))
            if weight_dtype is not None and mx.issubdtype(value.dtype, mx.floating):
                value = value.astype(weight_dtype)
            items.append((normalized_key, value))
    return items


def _weight_files(component_path: Path) -> tuple[Path, ...]:
    index_path = component_path / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        raw_index = json.loads(index_path.read_text("utf-8"))
        weight_map = raw_index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight index at '{index_path}'")
        filenames = tuple(sorted({str(value) for value in weight_map.values()}))
        return tuple(component_path / name for name in filenames)
    single_file = component_path / "diffusion_pytorch_model.safetensors"
    if single_file.exists():
        return (single_file,)
    legacy_file = component_path / "model.safetensors"
    if legacy_file.exists():
        return (legacy_file,)
    raise ValueError(
        f"Could not find diffusers safetensors weights under '{component_path}'"
    )


def _pad_sequence(sequences: list[mx.array]) -> mx.array:
    max_length = max(int(sequence.shape[0]) for sequence in sequences)
    padded: list[mx.array] = []
    for sequence in sequences:
        pad_length = max_length - int(sequence.shape[0])
        if pad_length > 0:
            pad_width = [(0, pad_length)] + [(0, 0)] * (sequence.ndim - 1)
            sequence = mx.pad(sequence, pad_width)
        padded.append(sequence)
    return mx.stack(padded, axis=0)


def _pad_rotary(
    sequences: list[tuple[mx.array, mx.array]],
) -> tuple[mx.array, mx.array]:
    cos_batch = _pad_sequence([sequence[0] for sequence in sequences])
    sin_batch = _pad_sequence([sequence[1] for sequence in sequences])
    return cos_batch, sin_batch


def _sequence_mask(lengths: list[int]) -> mx.array:
    max_length = max(lengths)
    mask = np.zeros((len(lengths), max_length), dtype=np.bool_)
    for index, length in enumerate(lengths):
        mask[index, :length] = True
    return mx.array(mask)


def _embedding_tuple(values: tuple[object, ...]) -> tuple[mx.array, ...]:
    embeddings: list[mx.array] = []
    for value in values:
        if isinstance(value, np.ndarray):
            embeddings.append(mx.array(value))
            continue
        if isinstance(value, mx.array):
            embeddings.append(value)
            continue
        raise ValueError("Prompt embeddings must be MLX arrays or numpy arrays")
    return tuple(embeddings)


def _calculate_shift(
    image_seq_len: int,
    *,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def _vae_scale(config: AutoencoderConfig) -> int:
    return int(2 ** (len(config.block_out_channels) - 1) * 2)


def _postprocess_image(decoded: mx.array) -> np.ndarray:
    image = mx.clip((decoded / 2.0) + 0.5, 0.0, 1.0)
    image_np = np.asarray(mx.transpose(image, (1, 2, 0)))
    return np.rint(image_np * 255.0).astype(np.uint8)


def _apply_cfg_normalization(
    *,
    positive: mx.array,
    guided: mx.array,
    limit: float,
) -> mx.array:
    if limit <= 0.0:
        return guided
    positive_f32 = positive.astype(mx.float32)
    guided_f32 = guided.astype(mx.float32)
    positive_norm = float(mx.sqrt(mx.sum(mx.square(positive_f32))).item())
    guided_norm = float(mx.sqrt(mx.sum(mx.square(guided_f32))).item())
    max_guided_norm = positive_norm * limit
    if guided_norm <= 0.0 or guided_norm <= max_guided_norm:
        return guided
    scale = max_guided_norm / guided_norm
    return guided * mx.array(scale, dtype=guided.dtype)


def _prompt_signature(prompt_text: str) -> str:
    digest = hashlib.sha256(prompt_text.strip().encode("utf-8")).hexdigest()
    return digest[:16]

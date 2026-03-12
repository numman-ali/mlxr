from __future__ import annotations

import mlx.core as mx

from .. import _nn_compat as nn
from .config import Flux2TransformerConfig
from .embeddings import EmbedND, TimestepGuidanceEmbeddings, _silu


def _rope(pos: mx.array, dim: int, theta: float) -> mx.array:
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / float(dim)
    omega = 1.0 / (theta**scale)
    angles = pos[..., None].astype(mx.float32) * omega
    rope = mx.stack(
        [mx.cos(angles), -mx.sin(angles), mx.sin(angles), mx.cos(angles)],
        axis=-1,
    )
    return rope.reshape(*rope.shape[:-1], 2, 2)


def _apply_rope(x: mx.array, rope: mx.array) -> mx.array:
    original_dtype = x.dtype
    shape = x.shape
    x = x.reshape(*shape[:-1], -1, 1, 2).astype(mx.float32)
    rotated = x[..., 0] * rope[..., 0] + x[..., 1] * rope[..., 1]
    return rotated.reshape(shape).astype(original_dtype)


def _attention(q: mx.array, k: mx.array, v: mx.array, rope: mx.array) -> mx.array:
    head_dim = int(q.shape[-1])
    q = _apply_rope(q, rope)
    k = _apply_rope(k, rope)
    attended = mx.fast.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=head_dim**-0.5,
    )
    return attended.transpose(0, 2, 1, 3).reshape(
        int(attended.shape[0]), int(attended.shape[2]), -1
    )


class Modulation(nn.Module):
    def __init__(self, hidden_size: int, *, double: bool) -> None:
        super().__init__()
        self.multiplier = 6 if double else 3
        self.linear = nn.Linear(
            hidden_size,
            self.multiplier * hidden_size,
            bias=False,
        )

    def __call__(self, vec: mx.array) -> tuple[mx.array, ...]:
        outputs = self.linear(_silu(vec))[:, None, :]
        return tuple(mx.split(outputs, self.multiplier, axis=-1))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dim,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        normalized = mx.fast.rms_norm(
            x.astype(mx.float32),
            self.weight.astype(mx.float32),
            self.eps,
        )
        return normalized.astype(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.linear_in = nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False)
        self.linear_out = nn.Linear(mlp_hidden_dim, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, value = mx.split(self.linear_in(x), 2, axis=-1)
        return self.linear_out(_silu(gate) * value)


class DoubleStreamAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.add_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.add_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.add_v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)
        self.norm_added_q = RMSNorm(self.head_dim)
        self.norm_added_k = RMSNorm(self.head_dim)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_add_out = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        image_tokens: mx.array,
        context_tokens: mx.array,
        rope: mx.array,
    ) -> tuple[mx.array, mx.array]:
        batch_size = int(image_tokens.shape[0])
        image_length = int(image_tokens.shape[1])
        context_length = int(context_tokens.shape[1])

        image_q = (
            self.to_q(image_tokens)
            .reshape(batch_size, image_length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        image_k = (
            self.to_k(image_tokens)
            .reshape(batch_size, image_length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        image_v = (
            self.to_v(image_tokens)
            .reshape(batch_size, image_length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        context_q = (
            self.add_q_proj(context_tokens)
            .reshape(batch_size, context_length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        context_k = (
            self.add_k_proj(context_tokens)
            .reshape(batch_size, context_length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        context_v = (
            self.add_v_proj(context_tokens)
            .reshape(batch_size, context_length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        image_q = self.norm_q(image_q)
        image_k = self.norm_k(image_k)
        context_q = self.norm_added_q(context_q)
        context_k = self.norm_added_k(context_k)

        q = mx.concatenate([context_q, image_q], axis=2)
        k = mx.concatenate([context_k, image_k], axis=2)
        v = mx.concatenate([context_v, image_v], axis=2)
        attended = _attention(q, k, v, rope)
        context_attended = attended[:, :context_length, :]
        image_attended = attended[:, context_length:, :]
        return self.to_out(image_attended), self.to_add_out(context_attended)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.image_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1.0e-6)
        self.image_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1.0e-6)
        self.context_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1.0e-6)
        self.context_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1.0e-6)
        self.attn = DoubleStreamAttention(hidden_size, num_heads)
        self.ff = FeedForward(hidden_size, mlp_hidden_dim)
        self.ff_context = FeedForward(hidden_size, mlp_hidden_dim)

    def __call__(
        self,
        image_tokens: mx.array,
        context_tokens: mx.array,
        rope: mx.array,
        image_modulation: tuple[mx.array, ...],
        context_modulation: tuple[mx.array, ...],
    ) -> tuple[mx.array, mx.array]:
        (
            image_shift_1,
            image_scale_1,
            image_gate_1,
            image_shift_2,
            image_scale_2,
            image_gate_2,
        ) = image_modulation
        (
            context_shift_1,
            context_scale_1,
            context_gate_1,
            context_shift_2,
            context_scale_2,
            context_gate_2,
        ) = context_modulation

        image_attended, context_attended = self.attn(
            (1.0 + image_scale_1) * self.image_norm1(image_tokens) + image_shift_1,
            (1.0 + context_scale_1) * self.context_norm1(context_tokens)
            + context_shift_1,
            rope,
        )
        image_tokens = image_tokens + image_gate_1 * image_attended
        context_tokens = context_tokens + context_gate_1 * context_attended
        image_tokens = image_tokens + image_gate_2 * self.ff(
            (1.0 + image_scale_2) * self.image_norm2(image_tokens) + image_shift_2
        )
        context_tokens = context_tokens + context_gate_2 * self.ff_context(
            (1.0 + context_scale_2) * self.context_norm2(context_tokens)
            + context_shift_2
        )
        return image_tokens, context_tokens


class SingleStreamAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * 3.0)
        self.to_qkv_mlp_proj = nn.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim * 2,
            bias=False,
        )
        self.to_out = nn.Linear(
            hidden_size + self.mlp_hidden_dim,
            hidden_size,
            bias=False,
        )
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

    def __call__(self, tokens: mx.array, rope: mx.array) -> mx.array:
        batch_size = int(tokens.shape[0])
        length = int(tokens.shape[1])
        qkv_mlp = self.to_qkv_mlp_proj(tokens)
        q, k, v, mlp = mx.split(
            qkv_mlp,
            [
                self.hidden_size,
                self.hidden_size * 2,
                self.hidden_size * 3,
            ],
            axis=-1,
        )
        q = q.reshape(batch_size, length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(batch_size, length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(batch_size, length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        q = self.norm_q(q)
        k = self.norm_k(k)
        attended = _attention(q, k, v, rope)
        mlp_gate, mlp_value = mx.split(mlp, 2, axis=-1)
        return self.to_out(
            mx.concatenate([attended, _silu(mlp_gate) * mlp_value], axis=-1)
        )


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_size, affine=False, eps=1.0e-6)
        self.attn = SingleStreamAttention(hidden_size, num_heads)

    def __call__(
        self,
        tokens: mx.array,
        rope: mx.array,
        modulation: tuple[mx.array, ...],
    ) -> mx.array:
        shift, scale, gate = modulation
        hidden = (1.0 + scale) * self.pre_norm(tokens) + shift
        return tokens + gate * self.attn(hidden, rope)


class FinalModulation(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size * 2, bias=False)

    def __call__(self, vec: mx.array) -> tuple[mx.array, mx.array]:
        return tuple(mx.split(self.linear(_silu(vec)), 2, axis=-1))  # type: ignore[return-value]


class Flux2Transformer2DModel(nn.Module):
    def __init__(self, config: Flux2TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.x_embedder = nn.Linear(config.in_channels, self.hidden_size, bias=False)
        self.context_embedder = nn.Linear(
            config.joint_attention_dim, self.hidden_size, bias=False
        )
        self.time_guidance_embed = TimestepGuidanceEmbeddings(
            in_channels=config.timestep_guidance_channels,
            hidden_size=self.hidden_size,
            guidance_embeds=config.guidance_embeds,
        )
        self.position_embedder = EmbedND(
            theta=config.rope_theta,
            axes_dims=config.axes_dims_rope,
        )
        self.double_stream_modulation_img = Modulation(self.hidden_size, double=True)
        self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True)
        self.single_stream_modulation = Modulation(self.hidden_size, double=False)
        self.transformer_blocks: list[DoubleStreamBlock] = [
            DoubleStreamBlock(
                self.hidden_size,
                config.num_attention_heads,
                config.mlp_ratio,
            )
            for _ in range(config.num_layers)
        ]
        self.single_transformer_blocks: list[SingleStreamBlock] = [
            SingleStreamBlock(
                self.hidden_size,
                config.num_attention_heads,
            )
            for _ in range(config.num_single_layers)
        ]
        self.final_norm = nn.LayerNorm(self.hidden_size, affine=False, eps=config.eps)
        self.norm_out = FinalModulation(self.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.in_channels, bias=False)

    def __call__(
        self,
        *,
        x: mx.array,
        x_ids: mx.array,
        timesteps: mx.array,
        ctx: mx.array,
        ctx_ids: mx.array,
        guidance: mx.array | None,
    ) -> mx.array:
        image_tokens = self.x_embedder(x)
        context_tokens = self.context_embedder(ctx)
        timestep = timesteps.astype(image_tokens.dtype) * 1000.0
        guidance_input = (
            guidance.astype(image_tokens.dtype) * 1000.0
            if guidance is not None
            else None
        )
        vec = self.time_guidance_embed(timestep, guidance_input)
        rope = self.position_embedder(mx.concatenate([ctx_ids, x_ids], axis=1))
        image_modulation = self.double_stream_modulation_img(vec)
        context_modulation = self.double_stream_modulation_txt(vec)
        single_modulation = self.single_stream_modulation(vec)
        for double_block in self.transformer_blocks:
            image_tokens, context_tokens = double_block(
                image_tokens,
                context_tokens,
                rope,
                image_modulation,
                context_modulation,
            )
        combined = mx.concatenate([context_tokens, image_tokens], axis=1)
        for single_block in self.single_transformer_blocks:
            combined = single_block(combined, rope, single_modulation)
        image_tokens = combined[:, int(context_tokens.shape[1]) :, :]
        scale, shift = self.norm_out(vec)
        image_tokens = (1.0 + scale[:, None, :]) * self.final_norm(
            image_tokens
        ) + shift[:, None, :]
        return self.proj_out(image_tokens)

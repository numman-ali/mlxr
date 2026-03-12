from __future__ import annotations

import mlx.core as mx

from .. import _nn_compat as nn
from .config import QwenImageTransformerConfig
from .embeddings import QwenEmbedRope, TimestepProjection, apply_rotary_emb_qwen, silu


class RMSNorm(nn.Module):
    def __init__(self, dims: int, *, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dims,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        normalized = mx.fast.rms_norm(
            x.astype(mx.float32),
            self.weight.astype(mx.float32),
            self.eps,
        )
        return normalized.astype(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        inner_dim = hidden_size * 4
        self.proj_in = nn.Linear(hidden_size, inner_dim, bias=True)
        self.proj_out = nn.Linear(inner_dim, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj_out(nn.gelu_approx(self.proj_in(x)))


class DoubleStreamAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.add_q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.add_k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.add_v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm_q = RMSNorm(head_dim)
        self.norm_k = RMSNorm(head_dim)
        self.norm_added_q = RMSNorm(head_dim)
        self.norm_added_k = RMSNorm(head_dim)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_add_out = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(
        self,
        image_tokens: mx.array,
        context_tokens: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        batch_size = int(image_tokens.shape[0])
        image_length = int(image_tokens.shape[1])
        context_length = int(context_tokens.shape[1])

        image_q = self._reshape_heads(self.to_q(image_tokens), batch_size, image_length)
        image_k = self._reshape_heads(self.to_k(image_tokens), batch_size, image_length)
        image_v = self._reshape_heads(self.to_v(image_tokens), batch_size, image_length)
        context_q = self._reshape_heads(
            self.add_q_proj(context_tokens),
            batch_size,
            context_length,
        )
        context_k = self._reshape_heads(
            self.add_k_proj(context_tokens),
            batch_size,
            context_length,
        )
        context_v = self._reshape_heads(
            self.add_v_proj(context_tokens),
            batch_size,
            context_length,
        )

        image_q = self.norm_q(image_q)
        image_k = self.norm_k(image_k)
        context_q = self.norm_added_q(context_q)
        context_k = self.norm_added_k(context_k)

        image_freqs, text_freqs = image_rotary_emb
        image_q = apply_rotary_emb_qwen(image_q, image_freqs)
        image_k = apply_rotary_emb_qwen(image_k, image_freqs)
        context_q = apply_rotary_emb_qwen(context_q, text_freqs)
        context_k = apply_rotary_emb_qwen(context_k, text_freqs)

        q = mx.concatenate([context_q, image_q], axis=2)
        k = mx.concatenate([context_k, image_k], axis=2)
        v = mx.concatenate([context_v, image_v], axis=2)
        attended = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.head_dim**-0.5,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size,
            context_length + image_length,
            self.hidden_size,
        )
        context_attended = attended[:, :context_length, :]
        image_attended = attended[:, context_length:, :]
        return self.to_out(image_attended), self.to_add_out(context_attended)

    def _reshape_heads(
        self,
        tensor: mx.array,
        batch_size: int,
        sequence_length: int,
    ) -> mx.array:
        return tensor.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)


class QwenImageTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.img_mod = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, eps=1.0e-6, affine=False)
        self.attn = DoubleStreamAttention(hidden_size, num_heads, head_dim)
        self.img_norm2 = nn.LayerNorm(hidden_size, eps=1.0e-6, affine=False)
        self.img_mlp = FeedForward(hidden_size)
        self.txt_mod = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, eps=1.0e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(hidden_size, eps=1.0e-6, affine=False)
        self.txt_mlp = FeedForward(hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        modulate_index: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        img_mod1, img_mod2 = mx.split(self.img_mod(silu(temb)), 2, axis=-1)
        txt_temb = temb
        if int(temb.shape[0]) != int(encoder_hidden_states.shape[0]):
            txt_temb = mx.split(temb, 2, axis=0)[0]
        txt_mod1, txt_mod2 = mx.split(self.txt_mod(silu(txt_temb)), 2, axis=-1)

        img_modulated, img_gate1 = self._modulate(
            self.img_norm1(hidden_states), img_mod1, modulate_index
        )
        txt_modulated, txt_gate1 = self._modulate(
            self.txt_norm1(encoder_hidden_states),
            txt_mod1,
        )
        img_attended, txt_attended = self.attn(
            img_modulated,
            txt_modulated,
            image_rotary_emb,
        )
        hidden_states = hidden_states + img_gate1 * img_attended
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attended

        img_modulated2, img_gate2 = self._modulate(
            self.img_norm2(hidden_states),
            img_mod2,
            modulate_index,
        )
        txt_modulated2, txt_gate2 = self._modulate(
            self.txt_norm2(encoder_hidden_states),
            txt_mod2,
        )
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(
            txt_modulated2
        )
        return encoder_hidden_states, hidden_states

    def _modulate(
        self,
        hidden: mx.array,
        params: mx.array,
        modulate_index: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        shift, scale, gate = mx.split(params, 3, axis=-1)
        if modulate_index is None:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
            gate = gate[:, None, :]
            return hidden * (1.0 + scale) + shift, gate
        batch_size = int(hidden.shape[0])
        if int(shift.shape[0]) != batch_size * 2:
            raise ValueError(
                "Qwen-Image zero_cond_t modulation expects doubled timestep embeddings"
            )
        shift_zero, shift_one = mx.split(shift, 2, axis=0)
        scale_zero, scale_one = mx.split(scale, 2, axis=0)
        gate_zero, gate_one = mx.split(gate, 2, axis=0)
        mask = modulate_index.astype(hidden.dtype)[..., None]
        shift_result = (
            shift_zero[:, None, :] * (1.0 - mask) + shift_one[:, None, :] * mask
        )
        scale_result = (
            scale_zero[:, None, :] * (1.0 - mask) + scale_one[:, None, :] * mask
        )
        gate_result = gate_zero[:, None, :] * (1.0 - mask) + gate_one[:, None, :] * mask
        return hidden * (1.0 + scale_result) + shift_result, gate_result


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size * 2, bias=True)
        self.norm = nn.LayerNorm(hidden_size, eps=1.0e-6, affine=False)

    def __call__(self, hidden: mx.array, conditioning: mx.array) -> mx.array:
        projected = self.linear(silu(conditioning).astype(hidden.dtype))
        scale, shift = mx.split(projected, 2, axis=-1)
        return self.norm(hidden) * (1.0 + scale[:, None, :]) + shift[:, None, :]


class QwenImageTransformer2DModel(nn.Module):
    def __init__(self, config: QwenImageTransformerConfig) -> None:
        super().__init__()
        if config.guidance_embeds:
            raise ValueError(
                "Guidance-distilled Qwen-Image rows are not implemented yet"
            )
        if config.use_additional_t_cond or config.use_layer3d_rope:
            raise ValueError(
                "Owned Qwen-Image backend only supports the released 2512/edit transformer contract"
            )
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.attention_head_dim
        self.img_in = nn.Linear(config.in_channels, self.hidden_size, bias=True)
        self.txt_norm = RMSNorm(config.joint_attention_dim)
        self.txt_in = nn.Linear(config.joint_attention_dim, self.hidden_size, bias=True)
        self.time_text_embed = TimestepProjection(self.hidden_size)
        self.pos_embed = QwenEmbedRope(config)
        self.transformer_blocks = [
            QwenImageTransformerBlock(
                self.hidden_size,
                config.num_attention_heads,
                self.head_dim,
            )
            for _ in range(config.num_layers)
        ]
        self.norm_out = AdaLayerNormContinuous(self.hidden_size)
        self.proj_out = nn.Linear(
            self.hidden_size,
            config.patch_size * config.patch_size * config.out_channels,
            bias=True,
        )

    def __call__(
        self,
        *,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        timestep: mx.array,
        image_shapes: list[tuple[int, int, int]],
    ) -> mx.array:
        if encoder_hidden_states_mask is not None:
            mask_min = float(
                mx.min(encoder_hidden_states_mask.astype(mx.float32)).item()
            )
            if mask_min < 1.0:
                raise ValueError(
                    "Owned Qwen-Image backend currently supports fully-trimmed prompt masks only"
                )
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        if self.config.zero_cond_t:
            timestep = mx.concatenate(
                [
                    timestep.astype(hidden_states.dtype),
                    mx.zeros(
                        tuple(int(size) for size in timestep.shape),
                        dtype=hidden_states.dtype,
                    ),
                ],
                axis=0,
            )
            modulate_index = _build_modulate_index(
                image_shapes=image_shapes,
                batch_size=int(hidden_states.shape[0]),
                dtype=hidden_states.dtype,
            )
        else:
            modulate_index = None
        temb = self.time_text_embed(timestep.astype(hidden_states.dtype))
        image_rotary_emb = self.pos_embed(
            image_shapes,
            max_text_seq_len=int(encoder_hidden_states.shape[1]),
        )
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                modulate_index=modulate_index,
            )
        norm_temb = temb
        if self.config.zero_cond_t:
            norm_temb = mx.split(temb, 2, axis=0)[0]
        hidden_states = self.norm_out(hidden_states, norm_temb)
        return self.proj_out(hidden_states)


def _build_modulate_index(
    *,
    image_shapes: list[tuple[int, int, int]],
    batch_size: int,
    dtype: mx.Dtype,
) -> mx.array:
    if not image_shapes:
        raise ValueError("Qwen-Image zero_cond_t requires at least one image shape")
    token_counts = [frame * height * width for frame, height, width in image_shapes]
    row = [0.0] * token_counts[0]
    for count in token_counts[1:]:
        row.extend([1.0] * count)
    return mx.array([row] * batch_size, dtype=dtype)

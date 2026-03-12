from __future__ import annotations

from dataclasses import dataclass

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


def _sdpa(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    head_dim = int(q.shape[-1])
    attended = mx.fast.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=head_dim**-0.5,
    )
    return attended.transpose(0, 2, 1, 3).reshape(
        int(attended.shape[0]), int(attended.shape[2]), -1
    )


def _attention(q: mx.array, k: mx.array, v: mx.array, rope: mx.array) -> mx.array:
    q = _apply_rope(q, rope)
    k = _apply_rope(k, rope)
    return _sdpa(q, k, v)


@dataclass(slots=True)
class Flux2KVLayerCache:
    k_ref: mx.array | None = None
    v_ref: mx.array | None = None

    def store(self, *, k_ref: mx.array, v_ref: mx.array) -> None:
        self.k_ref = k_ref
        self.v_ref = v_ref

    def get(self) -> tuple[mx.array, mx.array]:
        if self.k_ref is None or self.v_ref is None:
            raise RuntimeError("FLUX.2 KV cache has not been populated yet")
        return self.k_ref, self.v_ref

    def clear(self) -> None:
        self.k_ref = None
        self.v_ref = None


@dataclass(slots=True)
class Flux2KVCache:
    double_block_caches: tuple[Flux2KVLayerCache, ...]
    single_block_caches: tuple[Flux2KVLayerCache, ...]
    num_ref_tokens: int

    @classmethod
    def create(
        cls,
        *,
        num_double_layers: int,
        num_single_layers: int,
        num_ref_tokens: int,
    ) -> Flux2KVCache:
        return cls(
            double_block_caches=tuple(
                Flux2KVLayerCache() for _ in range(num_double_layers)
            ),
            single_block_caches=tuple(
                Flux2KVLayerCache() for _ in range(num_single_layers)
            ),
            num_ref_tokens=num_ref_tokens,
        )

    def arrays(self) -> tuple[mx.array, ...]:
        arrays: list[mx.array] = []
        for cache in self.double_block_caches + self.single_block_caches:
            if cache.k_ref is not None:
                arrays.append(cache.k_ref)
            if cache.v_ref is not None:
                arrays.append(cache.v_ref)
        return tuple(arrays)

    def clear(self) -> None:
        for cache in self.double_block_caches:
            cache.clear()
        for cache in self.single_block_caches:
            cache.clear()
        self.num_ref_tokens = 0


def _broadcast_modulation(param: mx.array, *, length: int) -> mx.array:
    if int(param.shape[1]) == length:
        return param
    hidden_size = int(param.shape[-1])
    return mx.broadcast_to(param, (int(param.shape[0]), length, hidden_size))


def _blend_double_modulation(
    image_modulation: tuple[mx.array, ...],
    reference_modulation: tuple[mx.array, ...],
    *,
    num_ref_tokens: int,
    total_length: int,
) -> tuple[mx.array, ...]:
    if num_ref_tokens <= 0:
        return image_modulation
    image_length = total_length - num_ref_tokens
    blended: list[mx.array] = []
    for image_param, reference_param in zip(
        image_modulation, reference_modulation, strict=True
    ):
        blended.append(
            mx.concatenate(
                [
                    _broadcast_modulation(reference_param, length=num_ref_tokens),
                    _broadcast_modulation(image_param, length=image_length),
                ],
                axis=1,
            )
        )
    return tuple(blended)


def _blend_single_modulation(
    modulation: tuple[mx.array, ...],
    reference_modulation: tuple[mx.array, ...],
    *,
    num_txt_tokens: int,
    num_ref_tokens: int,
    total_length: int,
) -> tuple[mx.array, ...]:
    if num_ref_tokens <= 0:
        return modulation
    image_length = total_length - num_txt_tokens - num_ref_tokens
    blended: list[mx.array] = []
    for param, reference_param in zip(modulation, reference_modulation, strict=True):
        broadcast = _broadcast_modulation(param, length=total_length)
        blended.append(
            mx.concatenate(
                [
                    broadcast[:, :num_txt_tokens, :],
                    _broadcast_modulation(reference_param, length=num_ref_tokens),
                    broadcast[:, num_txt_tokens + num_ref_tokens :, :]
                    if image_length > 0
                    else broadcast[:, 0:0, :],
                ],
                axis=1,
            )
        )
    return tuple(blended)


def _kv_causal_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache: Flux2KVLayerCache | None = None,
) -> mx.array:
    if num_ref_tokens == 0 and kv_cache is None:
        return _sdpa(q, k, v)

    if kv_cache is not None:
        k_ref, v_ref = kv_cache.get()
        k_all = mx.concatenate(
            [
                k[:, :, :num_txt_tokens, :],
                k_ref,
                k[:, :, num_txt_tokens:, :],
            ],
            axis=2,
        )
        v_all = mx.concatenate(
            [
                v[:, :, :num_txt_tokens, :],
                v_ref,
                v[:, :, num_txt_tokens:, :],
            ],
            axis=2,
        )
        return _sdpa(q, k_all, v_all)

    ref_start = num_txt_tokens
    ref_end = num_txt_tokens + num_ref_tokens

    q_txt = q[:, :, :ref_start, :]
    q_ref = q[:, :, ref_start:ref_end, :]
    q_img = q[:, :, ref_end:, :]

    k_txt = k[:, :, :ref_start, :]
    k_ref = k[:, :, ref_start:ref_end, :]
    k_img = k[:, :, ref_end:, :]

    v_txt = v[:, :, :ref_start, :]
    v_ref = v[:, :, ref_start:ref_end, :]
    v_img = v[:, :, ref_end:, :]

    q_txt_img = mx.concatenate([q_txt, q_img], axis=2)
    k_all = mx.concatenate([k_txt, k_ref, k_img], axis=2)
    v_all = mx.concatenate([v_txt, v_ref, v_img], axis=2)
    attended_txt_img = _sdpa(q_txt_img, k_all, v_all)
    attended_txt = attended_txt_img[:, :ref_start, :]
    attended_img = attended_txt_img[:, ref_start:, :]
    attended_ref = _sdpa(q_ref, k_ref, v_ref)
    return mx.concatenate([attended_txt, attended_ref, attended_img], axis=1)


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

    def _qkv(
        self,
        image_tokens: mx.array,
        context_tokens: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, int, int]:
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
        return q, k, v, context_length, image_length

    def __call__(
        self,
        image_tokens: mx.array,
        context_tokens: mx.array,
        rope: mx.array,
    ) -> tuple[mx.array, mx.array]:
        q, k, v, context_length, _ = self._qkv(image_tokens, context_tokens)
        attended = _attention(q, k, v, rope)
        context_attended = attended[:, :context_length, :]
        image_attended = attended[:, context_length:, :]
        return self.to_out(image_attended), self.to_add_out(context_attended)

    def forward_kv_extract(
        self,
        image_tokens: mx.array,
        context_tokens: mx.array,
        rope: mx.array,
        *,
        num_ref_tokens: int,
        kv_cache: Flux2KVLayerCache,
    ) -> tuple[mx.array, mx.array]:
        q, k, v, context_length, _ = self._qkv(image_tokens, context_tokens)
        q = _apply_rope(q, rope)
        k = _apply_rope(k, rope)
        ref_start = context_length
        ref_end = context_length + num_ref_tokens
        kv_cache.store(
            k_ref=mx.array(k[:, :, ref_start:ref_end, :]),
            v_ref=mx.array(v[:, :, ref_start:ref_end, :]),
        )
        attended = _kv_causal_attention(
            q,
            k,
            v,
            num_txt_tokens=context_length,
            num_ref_tokens=num_ref_tokens,
        )
        context_attended = attended[:, :context_length, :]
        image_attended = attended[:, context_length:, :]
        return self.to_out(image_attended), self.to_add_out(context_attended)

    def forward_kv_cached(
        self,
        image_tokens: mx.array,
        context_tokens: mx.array,
        rope: mx.array,
        *,
        kv_cache: Flux2KVLayerCache,
    ) -> tuple[mx.array, mx.array]:
        q, k, v, context_length, _ = self._qkv(image_tokens, context_tokens)
        q = _apply_rope(q, rope)
        k = _apply_rope(k, rope)
        attended = _kv_causal_attention(
            q,
            k,
            v,
            num_txt_tokens=context_length,
            num_ref_tokens=0,
            kv_cache=kv_cache,
        )
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

    def forward_kv_extract(
        self,
        image_tokens: mx.array,
        context_tokens: mx.array,
        rope: mx.array,
        image_modulation: tuple[mx.array, ...],
        context_modulation: tuple[mx.array, ...],
        *,
        num_ref_tokens: int,
        kv_cache: Flux2KVLayerCache,
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

        image_attended, context_attended = self.attn.forward_kv_extract(
            (1.0 + image_scale_1) * self.image_norm1(image_tokens) + image_shift_1,
            (1.0 + context_scale_1) * self.context_norm1(context_tokens)
            + context_shift_1,
            rope,
            num_ref_tokens=num_ref_tokens,
            kv_cache=kv_cache,
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

    def forward_kv_cached(
        self,
        image_tokens: mx.array,
        context_tokens: mx.array,
        rope: mx.array,
        image_modulation: tuple[mx.array, ...],
        context_modulation: tuple[mx.array, ...],
        *,
        kv_cache: Flux2KVLayerCache,
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

        image_attended, context_attended = self.attn.forward_kv_cached(
            (1.0 + image_scale_1) * self.image_norm1(image_tokens) + image_shift_1,
            (1.0 + context_scale_1) * self.context_norm1(context_tokens)
            + context_shift_1,
            rope,
            kv_cache=kv_cache,
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

    def _qkv_mlp(
        self, tokens: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, int]:
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
        mlp_gate, mlp_value = mx.split(mlp, 2, axis=-1)
        return q, k, v, mx.concatenate([mlp_gate, mlp_value], axis=-1), length

    def _out(self, attended: mx.array, mlp: mx.array) -> mx.array:
        mlp_gate, mlp_value = mx.split(mlp, 2, axis=-1)
        return self.to_out(
            mx.concatenate([attended, _silu(mlp_gate) * mlp_value], axis=-1)
        )

    def __call__(self, tokens: mx.array, rope: mx.array) -> mx.array:
        q, k, v, mlp, _ = self._qkv_mlp(tokens)
        attended = _attention(q, k, v, rope)
        return self._out(attended, mlp)

    def forward_kv_extract(
        self,
        tokens: mx.array,
        rope: mx.array,
        *,
        num_txt_tokens: int,
        num_ref_tokens: int,
        kv_cache: Flux2KVLayerCache,
    ) -> mx.array:
        q, k, v, mlp, _ = self._qkv_mlp(tokens)
        q = _apply_rope(q, rope)
        k = _apply_rope(k, rope)
        ref_start = num_txt_tokens
        ref_end = num_txt_tokens + num_ref_tokens
        kv_cache.store(
            k_ref=mx.array(k[:, :, ref_start:ref_end, :]),
            v_ref=mx.array(v[:, :, ref_start:ref_end, :]),
        )
        attended = _kv_causal_attention(
            q,
            k,
            v,
            num_txt_tokens=num_txt_tokens,
            num_ref_tokens=num_ref_tokens,
        )
        return self._out(attended, mlp)

    def forward_kv_cached(
        self,
        tokens: mx.array,
        rope: mx.array,
        *,
        num_txt_tokens: int,
        kv_cache: Flux2KVLayerCache,
    ) -> mx.array:
        q, k, v, mlp, _ = self._qkv_mlp(tokens)
        q = _apply_rope(q, rope)
        k = _apply_rope(k, rope)
        attended = _kv_causal_attention(
            q,
            k,
            v,
            num_txt_tokens=num_txt_tokens,
            num_ref_tokens=0,
            kv_cache=kv_cache,
        )
        return self._out(attended, mlp)


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

    def forward_kv_extract(
        self,
        tokens: mx.array,
        rope: mx.array,
        modulation: tuple[mx.array, ...],
        *,
        num_txt_tokens: int,
        num_ref_tokens: int,
        kv_cache: Flux2KVLayerCache,
    ) -> mx.array:
        shift, scale, gate = modulation
        hidden = (1.0 + scale) * self.pre_norm(tokens) + shift
        return tokens + gate * self.attn.forward_kv_extract(
            hidden,
            rope,
            num_txt_tokens=num_txt_tokens,
            num_ref_tokens=num_ref_tokens,
            kv_cache=kv_cache,
        )

    def forward_kv_cached(
        self,
        tokens: mx.array,
        rope: mx.array,
        modulation: tuple[mx.array, ...],
        *,
        num_txt_tokens: int,
        kv_cache: Flux2KVLayerCache,
    ) -> mx.array:
        shift, scale, gate = modulation
        hidden = (1.0 + scale) * self.pre_norm(tokens) + shift
        return tokens + gate * self.attn.forward_kv_cached(
            hidden,
            rope,
            num_txt_tokens=num_txt_tokens,
            kv_cache=kv_cache,
        )


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

    def forward_kv_extract(
        self,
        *,
        x: mx.array,
        x_ids: mx.array,
        x_ref: mx.array,
        x_ref_ids: mx.array,
        timesteps: mx.array,
        ctx: mx.array,
        ctx_ids: mx.array,
        guidance: mx.array | None,
        ref_fixed_timestep: float = 0.0,
    ) -> tuple[mx.array, Flux2KVCache]:
        num_ref_tokens = int(x_ref.shape[1])
        image_tokens = mx.concatenate(
            [self.x_embedder(x_ref), self.x_embedder(x)], axis=1
        )
        context_tokens = self.context_embedder(ctx)
        combined_ids = mx.concatenate([x_ref_ids, x_ids], axis=1)
        timestep = timesteps.astype(image_tokens.dtype) * 1000.0
        guidance_input = (
            guidance.astype(image_tokens.dtype) * 1000.0
            if guidance is not None
            else None
        )
        vec = self.time_guidance_embed(timestep, guidance_input)
        ref_vec = self.time_guidance_embed(
            mx.full(
                timesteps.shape,
                ref_fixed_timestep,
                dtype=image_tokens.dtype,
            )
            * 1000.0,
            guidance_input,
        )
        rope = self.position_embedder(mx.concatenate([ctx_ids, combined_ids], axis=1))
        image_modulation = _blend_double_modulation(
            self.double_stream_modulation_img(vec),
            self.double_stream_modulation_img(ref_vec),
            num_ref_tokens=num_ref_tokens,
            total_length=int(image_tokens.shape[1]),
        )
        context_modulation = self.double_stream_modulation_txt(vec)
        single_modulation = self.single_stream_modulation(vec)
        kv_cache = Flux2KVCache.create(
            num_double_layers=len(self.transformer_blocks),
            num_single_layers=len(self.single_transformer_blocks),
            num_ref_tokens=num_ref_tokens,
        )
        for index, double_block in enumerate(self.transformer_blocks):
            image_tokens, context_tokens = double_block.forward_kv_extract(
                image_tokens,
                context_tokens,
                rope,
                image_modulation,
                context_modulation,
                num_ref_tokens=num_ref_tokens,
                kv_cache=kv_cache.double_block_caches[index],
            )
        combined = mx.concatenate([context_tokens, image_tokens], axis=1)
        combined_modulation = _blend_single_modulation(
            single_modulation,
            self.single_stream_modulation(ref_vec),
            num_txt_tokens=int(context_tokens.shape[1]),
            num_ref_tokens=num_ref_tokens,
            total_length=int(combined.shape[1]),
        )
        for index, single_block in enumerate(self.single_transformer_blocks):
            combined = single_block.forward_kv_extract(
                combined,
                rope,
                combined_modulation,
                num_txt_tokens=int(context_tokens.shape[1]),
                num_ref_tokens=num_ref_tokens,
                kv_cache=kv_cache.single_block_caches[index],
            )
        image_tokens = combined[:, int(context_tokens.shape[1]) + num_ref_tokens :, :]
        scale, shift = self.norm_out(vec)
        image_tokens = (1.0 + scale[:, None, :]) * self.final_norm(
            image_tokens
        ) + shift[:, None, :]
        output = self.proj_out(image_tokens)
        cache_arrays = kv_cache.arrays()
        if cache_arrays:
            mx.eval(output, *cache_arrays)
        return output, kv_cache

    def forward_kv_cached(
        self,
        *,
        x: mx.array,
        x_ids: mx.array,
        timesteps: mx.array,
        ctx: mx.array,
        ctx_ids: mx.array,
        guidance: mx.array | None,
        kv_cache: Flux2KVCache,
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
        for index, double_block in enumerate(self.transformer_blocks):
            image_tokens, context_tokens = double_block.forward_kv_cached(
                image_tokens,
                context_tokens,
                rope,
                image_modulation,
                context_modulation,
                kv_cache=kv_cache.double_block_caches[index],
            )
        combined = mx.concatenate([context_tokens, image_tokens], axis=1)
        for index, single_block in enumerate(self.single_transformer_blocks):
            combined = single_block.forward_kv_cached(
                combined,
                rope,
                single_modulation,
                num_txt_tokens=int(context_tokens.shape[1]),
                kv_cache=kv_cache.single_block_caches[index],
            )
        image_tokens = combined[:, int(context_tokens.shape[1]) :, :]
        scale, shift = self.norm_out(vec)
        image_tokens = (1.0 + scale[:, None, :]) * self.final_norm(
            image_tokens
        ) + shift[:, None, :]
        return self.proj_out(image_tokens)

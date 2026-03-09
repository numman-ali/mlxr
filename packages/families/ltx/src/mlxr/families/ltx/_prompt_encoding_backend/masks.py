from __future__ import annotations

from functools import lru_cache

from .._generation_backend.rope_ops import precompute_freqs_cis
from .compat import TextConfig
from .runtime import (
    _RUNTIME_IMPORT_ERROR,
    create_attention_mask,
    create_causal_mask,
    mx,
    nn,
    np,
)

if _RUNTIME_IMPORT_ERROR is None:

    def _required_quantization_int(
        quantization: dict[str, object],
        key: str,
    ) -> int:
        value = quantization.get(key)
        if not isinstance(value, int):
            raise ValueError(f"LTX quantization '{key}' must be an integer")
        return value

    def _apply_quantization(
        model: object,
        weights: set[str],
        quantization: dict[str, object],
    ) -> None:
        def get_class_predicate(path: str, module: object) -> object:
            if path in quantization:
                return quantization[path]
            if not hasattr(module, "to_quantized"):
                return False
            if hasattr(module, "weight") and module.weight.shape[0] % 64 != 0:
                return False
            return f"{path}.scales" in weights

        quantize = getattr(nn, "quantize")
        quantize(
            model,
            group_size=_required_quantization_int(quantization, "group_size"),
            bits=_required_quantization_int(quantization, "bits"),
            mode=str(quantization.get("mode", "affine")),
            class_predicate=get_class_predicate,
        )

    def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
        return mx.fast.rms_norm(x, mx.ones((x.shape[-1],), dtype=x.dtype), eps)

    def _left_padding(attention_mask: mx.array) -> mx.array:
        sequence_length = int(attention_mask.shape[-1])
        valid_tokens = mx.sum(attention_mask.astype(mx.int32), axis=-1)
        return (
            mx.full(valid_tokens.shape, sequence_length, dtype=mx.int32) - valid_tokens
        )

    def _gemma_attention_masks(
        *,
        hidden: mx.array,
        attention_mask: mx.array | None,
        cache: list[object | None],
        config: TextConfig,
    ) -> tuple[mx.array | str | None, mx.array | str | None]:
        if attention_mask is None:
            global_mask = create_attention_mask(
                hidden, cache[config.sliding_window_pattern - 1]
            )
            if config.sliding_window_pattern > 1:
                sliding_window_mask = create_attention_mask(
                    hidden,
                    cache[0],
                    window_size=config.sliding_window,
                )
            else:
                sliding_window_mask = None
            return global_mask, sliding_window_mask

        if any(entry is not None for entry in cache):
            raise NotImplementedError(
                "Padded LTX MLX Gemma prompt encoding does not support KV cache"
            )

        left_padding = _left_padding(attention_mask)
        global_mask = create_causal_mask(
            int(hidden.shape[1]),
            left_padding=left_padding,
        )
        if config.sliding_window_pattern > 1:
            sliding_window_mask = create_causal_mask(
                int(hidden.shape[1]),
                left_padding=left_padding,
                window_size=config.sliding_window,
            )
        else:
            sliding_window_mask = None
        return global_mask, sliding_window_mask

    @lru_cache(maxsize=32)
    def _connector_precomputed_freqs(
        seq_len: int,
        dim: int,
        num_heads: int,
        theta: float,
        max_pos: tuple[int, ...],
        rope_type: str,
        double_precision: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        position_axis = np.arange(seq_len, dtype=np.float32)[None, None, :]
        if len(max_pos) > 1:
            position_axis = np.broadcast_to(position_axis, (1, len(max_pos), seq_len))
        cos_freqs, sin_freqs = precompute_freqs_cis(
            mx.array(position_axis),
            dim=dim,
            out_dtype=mx.float32,
            theta=theta,
            max_pos=[int(position) for position in max_pos],
            use_middle_indices_grid=False,
            num_attention_heads=num_heads,
            rope_type=rope_type,
            double_precision=double_precision,
        )
        return (
            np.asarray(cos_freqs, dtype=np.float32),
            np.asarray(sin_freqs, dtype=np.float32),
        )

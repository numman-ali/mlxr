from __future__ import annotations

import functools

from .runtime import (
    _RUNTIME_IMPORT_ERROR,
    TextConfig,
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
    ) -> tuple[mx.array | None, mx.array | None]:
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

    @functools.lru_cache(maxsize=16)
    def _connector_precomputed_freqs(
        seq_len: int,
        dim: int,
        num_heads: int,
        theta: float,
        max_pos: tuple[int, ...],
        rope_type: str,
        double_precision: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        np_dtype = np.float64 if double_precision else np.float32
        n_elem = 2 * len(max_pos)
        indices = np.power(
            theta,
            np.linspace(
                np.log(1.0) / np.log(theta),
                np.log(theta) / np.log(theta),
                dim // n_elem,
                dtype=np_dtype,
            ),
        )
        indices = indices * (np.pi / 2)

        fractional_positions = np.stack(
            [
                np.arange(seq_len, dtype=np_dtype)
                / np.asarray(position_max, dtype=np_dtype)
                for position_max in max_pos
            ],
            axis=-1,
        )
        freqs = ((fractional_positions[:, :, None] * 2) - 1) * indices[None, None, :]
        freqs = freqs.reshape(1, seq_len, -1)

        if rope_type == "split":
            expected_freqs = dim // 2
            pad_size = expected_freqs - freqs.shape[-1]
            cos_freq = np.cos(freqs)
            sin_freq = np.sin(freqs)
            if pad_size > 0:
                cos_padding = np.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = np.zeros_like(sin_freq[:, :, :pad_size])
                cos_freq = np.concatenate([cos_padding, cos_freq], axis=-1)
                sin_freq = np.concatenate([sin_padding, sin_freq], axis=-1)
            cos_freq = np.swapaxes(cos_freq.reshape(1, seq_len, num_heads, -1), 1, 2)
            sin_freq = np.swapaxes(sin_freq.reshape(1, seq_len, num_heads, -1), 1, 2)
        else:
            pad_size = dim % n_elem
            cos_freq = np.repeat(np.cos(freqs), 2, axis=-1)
            sin_freq = np.repeat(np.sin(freqs), 2, axis=-1)
            if pad_size > 0:
                cos_padding = np.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = np.zeros_like(sin_freq[:, :, :pad_size])
                cos_freq = np.concatenate([cos_padding, cos_freq], axis=-1)
                sin_freq = np.concatenate([sin_padding, sin_freq], axis=-1)
        return cos_freq, sin_freq

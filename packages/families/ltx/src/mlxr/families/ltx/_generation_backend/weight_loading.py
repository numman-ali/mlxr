from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import mlx.core as mx

from .types import MLXArray, _GeneratorModule


def _dominant_weight_dtype(
    weights: Mapping[str, MLXArray], *, context: str
) -> mx.Dtype:
    dtype_sizes: Counter[mx.Dtype] = Counter()
    for value in weights.values():
        dtype_sizes[value.dtype] += int(value.size)
    if not dtype_sizes:
        raise RuntimeError(f"{context} did not include any weights to infer dtype")
    return max(dtype_sizes.items(), key=lambda item: item[1])[0]


def align_module_dtype_to_weights(
    module: _GeneratorModule,
    weights: Mapping[str, MLXArray],
    *,
    context: str,
) -> mx.Dtype:
    target_dtype = _dominant_weight_dtype(weights, context=context)
    module.set_dtype(target_dtype)
    return target_dtype

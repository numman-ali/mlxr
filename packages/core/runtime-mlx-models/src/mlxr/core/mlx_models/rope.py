from __future__ import annotations

import math

from ._nn_compat import ModuleBase, RopeLike, build_rope, mx


def _require_float(value: object, *, context: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(context)


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(context)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(context)


def _require_float_list(
    value: object,
    *,
    context: str,
) -> list[float] | float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and all(
        isinstance(item, (int, float)) for item in value
    ):
        return [float(item) for item in value]
    raise ValueError(context)


class SuScaledRoPE(ModuleBase):
    def __init__(
        self,
        dims: int,
        *,
        base: float = 10000.0,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        short_factor: list[float] | float = 1.0,
        long_factor: list[float] | float = 1.0,
        short_mscale: float | None = None,
        long_mscale: float | None = None,
    ) -> None:
        super().__init__()
        del short_factor, short_mscale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.dim = dims
        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        self._freqs = mx.array(long_factor, dtype=mx.float32) * freqs

        def default_scale(factor: float) -> float:
            return math.sqrt(
                1 + math.log(factor) / math.log(original_max_position_embeddings)
            )

        factor = max_position_embeddings / original_max_position_embeddings
        self._scale = long_mscale or (1.0 if factor <= 1.0 else default_scale(factor))

    def __call__(self, x: mx.array, *, offset: int | mx.array = 0) -> mx.array:
        x[..., : self.dim] = self._scale * x[..., : self.dim]
        return mx.fast.rope(
            x,
            self.dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Llama3RoPE(ModuleBase):
    def __init__(
        self,
        dims: int,
        *,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000.0,
        scaling_config: dict[str, object],
    ) -> None:
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional

        factor = _require_float(
            scaling_config["factor"], context="Llama3 rope factor must be numeric"
        )
        low_freq_factor = _require_float(
            scaling_config.get("low_freq_factor", 1.0),
            context="Llama3 rope low_freq_factor must be numeric",
        )
        high_freq_factor = _require_float(
            scaling_config.get("high_freq_factor", 4.0),
            context="Llama3 rope high_freq_factor must be numeric",
        )
        old_context_len = _require_int(
            scaling_config.get("original_max_position_embeddings", 8192),
            context="Llama3 rope original_max_position_embeddings must be an integer",
        )

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        wavelens = 2 * mx.pi * freqs
        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium, smooth_freqs, freqs)

    def __call__(self, x: mx.array, *, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class YarnRoPE(ModuleBase):
    def __init__(
        self,
        dims: int,
        *,
        traditional: bool = False,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ) -> None:
        super().__init__()
        del max_position_embeddings

        def yarn_find_correction_dim(num_rotations: float) -> float:
            return (
                dims
                * math.log(
                    original_max_position_embeddings / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        def yarn_find_correction_range() -> tuple[int, int]:
            low = math.floor(yarn_find_correction_dim(beta_fast))
            high = math.ceil(yarn_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale: float, scale_base: float) -> float:
            if scale <= 1:
                return 1.0
            return 0.1 * scale_base * math.log(scale) + 1.0

        def linear_ramp_mask(minimum: int, maximum: int, dim: int) -> mx.array:
            if minimum == maximum:
                maximum += 1
            linear = (mx.arange(dim, dtype=mx.float32) - minimum) / (maximum - minimum)
            return mx.clip(linear, 0, 1)

        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * freq_extra
        low, high = yarn_find_correction_range()
        freq_mask = 1.0 - linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x: mx.array, *, offset: int = 0) -> mx.array:
        if self.mscale != 1.0:
            x[..., : self.dims] = self.mscale * x[..., : self.dims]
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def initialize_rope(
    *,
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: dict[str, object] | None = None,
    max_position_embeddings: int | None = None,
) -> RopeLike:
    if scaling_config is None:
        rope_type = "default"
    else:
        rope_type = str(
            scaling_config.get("type") or scaling_config.get("rope_type", "default")
        )
    if rope_type in {"default", "linear"}:
        scale = 1.0
        if rope_type == "linear" and scaling_config is not None:
            scale = 1 / _require_float(
                scaling_config["factor"],
                context="Linear rope factor must be numeric",
            )
        return build_rope(dims, traditional=traditional, base=base, scale=scale)
    if rope_type == "llama3":
        if max_position_embeddings is None or scaling_config is None:
            raise ValueError(
                "llama3 rope scaling requires max_position_embeddings and scaling_config"
            )
        return Llama3RoPE(
            dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )
    if rope_type in {"yarn", "deepseek_yarn", "telechat3-yarn"}:
        if scaling_config is None:
            raise ValueError("yarn rope scaling requires scaling_config")
        return YarnRoPE(
            dims,
            max_position_embeddings=max_position_embeddings or 2048,
            traditional=traditional,
            scaling_factor=_require_float(
                scaling_config["factor"],
                context="Yarn rope factor must be numeric",
            ),
            base=base,
            original_max_position_embeddings=_require_int(
                scaling_config.get("original_max_position_embeddings", 4096),
                context="Yarn rope original_max_position_embeddings must be an integer",
            ),
            beta_fast=_require_float(
                scaling_config.get("beta_fast", 32.0),
                context="Yarn rope beta_fast must be numeric",
            ),
            beta_slow=_require_float(
                scaling_config.get("beta_slow", 1.0),
                context="Yarn rope beta_slow must be numeric",
            ),
            mscale=_require_float(
                scaling_config.get("mscale", 1.0),
                context="Yarn rope mscale must be numeric",
            ),
            mscale_all_dim=_require_float(
                scaling_config.get("mscale_all_dim", 0.0),
                context="Yarn rope mscale_all_dim must be numeric",
            ),
        )
    if rope_type == "longrope":
        if max_position_embeddings is None or scaling_config is None:
            raise ValueError(
                "longrope scaling requires max_position_embeddings and scaling_config"
            )
        return SuScaledRoPE(
            dims,
            base=base,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=_require_int(
                scaling_config["original_max_position_embeddings"],
                context="Longrope original_max_position_embeddings must be an integer",
            ),
            short_factor=_require_float_list(
                scaling_config["short_factor"],
                context="Longrope short_factor must be numeric or a numeric list",
            ),
            long_factor=_require_float_list(
                scaling_config["long_factor"],
                context="Longrope long_factor must be numeric or a numeric list",
            ),
            short_mscale=(
                _require_float(
                    scaling_config["short_mscale"],
                    context="Longrope short_mscale must be numeric",
                )
                if "short_mscale" in scaling_config
                else None
            ),
            long_mscale=(
                _require_float(
                    scaling_config["long_mscale"],
                    context="Longrope long_mscale must be numeric",
                )
                if "long_mscale" in scaling_config
                else None
            ),
        )
    raise ValueError(f"Unsupported rope scaling type '{rope_type}'")

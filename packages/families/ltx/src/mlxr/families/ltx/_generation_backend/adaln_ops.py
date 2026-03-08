from __future__ import annotations

import mlx.core as mx

from .. import _nn_compat as nn
from .timestep_embedding import Timesteps
from .types import MLXArray

ADALN_NUM_BASE_PARAMS = 6
ADALN_NUM_CROSS_ATTN_PARAMS = 3


def adaln_embedding_coefficient(cross_attention_adaln: bool) -> int:
    return ADALN_NUM_BASE_PARAMS + (
        ADALN_NUM_CROSS_ATTN_PARAMS if cross_attention_adaln else 0
    )


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int | None = None,
        sample_proj_bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(
            time_embed_dim,
            out_dim if out_dim is not None else time_embed_dim,
            bias=sample_proj_bias,
        )

    def __call__(self, sample: MLXArray) -> MLXArray:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, size_emb_dim: int) -> None:
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def __call__(self, timestep: MLXArray, *, hidden_dtype: mx.Dtype) -> MLXArray:
        projected = self.time_proj(timestep)
        return self.timestep_embedder(projected.astype(hidden_dtype))


class AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6) -> None:
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            bias=True,
        )

    def __call__(
        self,
        timestep: MLXArray,
        *,
        hidden_dtype: mx.Dtype | None = None,
    ) -> tuple[MLXArray, MLXArray]:
        dtype = hidden_dtype if hidden_dtype is not None else timestep.dtype
        embedded_timestep = self.emb(timestep, hidden_dtype=dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep

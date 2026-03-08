from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .rope_ops import precompute_freqs_cis
from .types import MLXArray, _PatchedModality, _PatchedTransformerArgs


class _ProjectionLike:
    def __call__(self, x: MLXArray) -> MLXArray:
        raise NotImplementedError


class _AdalnLike:
    def __call__(
        self,
        timestep: MLXArray,
        *,
        hidden_dtype: mx.Dtype | None = None,
    ) -> tuple[MLXArray, MLXArray]:
        raise NotImplementedError


def _resolve_rope_type_value(rope_type: object) -> str:
    if isinstance(rope_type, str):
        return rope_type
    rope_type_value = getattr(rope_type, "value", None)
    if isinstance(rope_type_value, str):
        return rope_type_value
    raise ValueError(f"Unsupported rope type value {rope_type!r}")


@dataclass(frozen=True, slots=True)
class _TransformerPreprocessorConfig:
    patchify_proj: _ProjectionLike
    adaln: _AdalnLike
    inner_dim: int
    max_pos: list[int]
    num_attention_heads: int
    use_middle_indices_grid: bool
    timestep_scale_multiplier: int
    positional_embedding_theta: float
    rope_type: object
    caption_projection: _ProjectionLike | None = None
    prompt_adaln: _AdalnLike | None = None
    double_precision_rope: bool = False


class TransformerArgsPreprocessor:
    def __init__(
        self,
        *,
        patchify_proj: _ProjectionLike,
        adaln: _AdalnLike,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        positional_embedding_theta: float,
        rope_type: object,
        caption_projection: _ProjectionLike | None = None,
        prompt_adaln: _AdalnLike | None = None,
        double_precision_rope: bool = False,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln
        self.double_precision_rope = double_precision_rope

    def _prepare_timestep(
        self,
        timestep: MLXArray,
        batch_size: int,
        *,
        hidden_dtype: mx.Dtype,
    ) -> tuple[MLXArray, MLXArray]:
        timestep_scaled = timestep * self.timestep_scale_multiplier
        values, embedded = self.adaln(
            mx.reshape(timestep_scaled, (-1,)),
            hidden_dtype=hidden_dtype,
        )
        values = mx.reshape(values, (batch_size, -1, values.shape[-1]))
        embedded = mx.reshape(embedded, (batch_size, -1, embedded.shape[-1]))
        return values, embedded

    def _prepare_context(
        self,
        context: MLXArray,
        x: MLXArray,
        attention_mask: MLXArray | None = None,
    ) -> tuple[MLXArray, MLXArray | None]:
        if self.caption_projection is None:
            if context.ndim != 3 or int(context.shape[-1]) != int(x.shape[-1]):
                raise ValueError(
                    "LTX prompt context must already be post-connector and transformer-width "
                    f"when caption projection is disabled; got {tuple(int(size) for size in context.shape)} "
                    f"for transformer width {int(x.shape[-1])}"
                )
            return context, attention_mask
        projected = self.caption_projection(context)
        batch_size = int(x.shape[0])
        projected = mx.reshape(projected, (batch_size, -1, x.shape[-1]))
        return projected, attention_mask

    def _prepare_attention_mask(
        self,
        attention_mask: MLXArray | None,
        hidden_dtype: mx.Dtype,
    ) -> MLXArray | None:
        if attention_mask is None or mx.issubdtype(attention_mask.dtype, mx.floating):
            return attention_mask
        return (attention_mask - 1).astype(hidden_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * mx.finfo(hidden_dtype).max

    def _prepare_positional_embeddings(
        self,
        *,
        positions: MLXArray,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
    ) -> tuple[MLXArray, MLXArray]:
        return precompute_freqs_cis(
            positions,
            dim=inner_dim,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=_resolve_rope_type_value(self.rope_type),
            double_precision=self.double_precision_rope,
        )

    def prepare(self, modality: _PatchedModality) -> _PatchedTransformerArgs:
        x = self.patchify_proj(modality.latent)
        timesteps, embedded_timestep = self._prepare_timestep(
            modality.timesteps, int(x.shape[0]), hidden_dtype=x.dtype
        )
        prompt_timestep = None
        if self.prompt_adaln is not None:
            sigma_scaled = modality.sigma * self.timestep_scale_multiplier
            prompt_values, _ = self.prompt_adaln(
                mx.reshape(sigma_scaled, (-1,)),
                hidden_dtype=modality.latent.dtype,
            )
            prompt_timestep = mx.reshape(
                prompt_values, (int(x.shape[0]), -1, prompt_values.shape[-1])
            )
        context, attention_mask = self._prepare_context(
            modality.context,
            x,
            modality.context_mask,
        )
        attention_mask = self._prepare_attention_mask(
            attention_mask, modality.latent.dtype
        )
        positional_embeddings = (
            modality.positional_embeddings
            if modality.positional_embeddings is not None
            else self._prepare_positional_embeddings(
                positions=modality.positions,
                inner_dim=self.inner_dim,
                max_pos=self.max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                num_attention_heads=self.num_attention_heads,
            )
        )
        return _PatchedTransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timesteps,
            embedded_timestep=embedded_timestep,
            positional_embeddings=positional_embeddings,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(
        self,
        *,
        patchify_proj: _ProjectionLike,
        adaln: _AdalnLike,
        cross_scale_shift_adaln: _AdalnLike,
        cross_gate_adaln: _AdalnLike,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        positional_embedding_theta: float,
        rope_type: object,
        av_ca_timestep_scale_multiplier: int,
        caption_projection: _ProjectionLike | None = None,
        prompt_adaln: _AdalnLike | None = None,
        double_precision_rope: bool = False,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            prompt_adaln=prompt_adaln,
            double_precision_rope=double_precision_rope,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def _prepare_cross_attention_timestep(
        self,
        *,
        timestep: MLXArray,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: mx.Dtype,
    ) -> tuple[MLXArray, MLXArray]:
        timestep_scaled = timestep * timestep_scale_multiplier
        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier
        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            mx.reshape(timestep_scaled, (-1,)),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = mx.reshape(
            scale_shift_timestep,
            (batch_size, -1, scale_shift_timestep.shape[-1]),
        )
        gate_timestep, _ = self.cross_gate_adaln(
            mx.reshape(timestep_scaled * av_ca_factor, (-1,)),
            hidden_dtype=hidden_dtype,
        )
        gate_timestep = mx.reshape(
            gate_timestep, (batch_size, -1, gate_timestep.shape[-1])
        )
        return scale_shift_timestep, gate_timestep

    def prepare(
        self,
        modality: _PatchedModality,
        cross_modality: _PatchedModality | None = None,
    ) -> _PatchedTransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        if cross_modality is None:
            return transformer_args
        if cross_modality.timesteps.shape[0] != modality.timesteps.shape[0]:
            raise ValueError(
                "Cross modality timesteps must have the same batch size as the modality"
            )
        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
        )
        cross_scale_shift_timestep, cross_gate_timestep = (
            self._prepare_cross_attention_timestep(
                timestep=modality.timesteps,
                timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
                batch_size=int(transformer_args.x.shape[0]),
                hidden_dtype=transformer_args.x.dtype,
            )
        )
        return _PatchedTransformerArgs(
            x=transformer_args.x,
            context=transformer_args.context,
            context_mask=transformer_args.context_mask,
            timesteps=transformer_args.timesteps,
            embedded_timestep=transformer_args.embedded_timestep,
            positional_embeddings=transformer_args.positional_embeddings,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
            enabled=transformer_args.enabled,
            prompt_timestep=transformer_args.prompt_timestep,
        )

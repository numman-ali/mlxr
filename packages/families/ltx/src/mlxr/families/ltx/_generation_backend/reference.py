from __future__ import annotations

from dataclasses import replace
from typing import Callable, Protocol, TypeGuard

import mlx.core as mx

from .. import _nn_compat as nn
from .types import (
    MLXArray,
    _PatchedModality,
    _PatchedTransformerArgs,
    _ReferenceImports,
    _TransformerConfigLike,
)


class _PatchableAttention(Protocol):
    heads: int
    rope_type: object
    to_q: Callable[[MLXArray], MLXArray]
    to_k: Callable[[MLXArray], MLXArray]
    to_v: Callable[[MLXArray], MLXArray]
    q_norm: Callable[[MLXArray], MLXArray]
    k_norm: Callable[[MLXArray], MLXArray]
    to_out: Callable[[MLXArray], MLXArray]
    to_gate_logits: nn.Linear | None

    def __call__(
        self,
        x: MLXArray,
        *,
        context: MLXArray | None = None,
        mask: MLXArray | None = None,
        pe: tuple[MLXArray, MLXArray] | None = None,
        k_pe: tuple[MLXArray, MLXArray] | None = None,
    ) -> MLXArray: ...


class _PatchablePreprocessor(Protocol):
    patchify_proj: Callable[[MLXArray], MLXArray]
    prompt_adaln: object | None
    timestep_scale_multiplier: int
    inner_dim: int
    max_pos: list[int]
    use_middle_indices_grid: bool
    num_attention_heads: int

    def _prepare_timestep(
        self, timestep: MLXArray, batch_size: int, *, hidden_dtype: mx.Dtype
    ) -> tuple[MLXArray, MLXArray]: ...

    def _prepare_context(
        self,
        context: MLXArray,
        x: MLXArray,
        attention_mask: MLXArray | None = None,
    ) -> tuple[MLXArray, MLXArray | None]: ...

    def _prepare_attention_mask(
        self,
        attention_mask: MLXArray | None,
        hidden_dtype: mx.Dtype,
    ) -> MLXArray | None: ...

    def _prepare_positional_embeddings(
        self,
        *,
        positions: MLXArray,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
    ) -> tuple[MLXArray, MLXArray]: ...


class _PatchableSimplePreprocessor(_PatchablePreprocessor, Protocol):
    def prepare(self, modality: _PatchedModality) -> _PatchedTransformerArgs: ...


class _PatchableMultiPreprocessor(Protocol):
    simple_preprocessor: _PatchableSimplePreprocessor
    audio_cross_attention_dim: int
    cross_pe_max_pos: int

    def prepare(
        self,
        modality: _PatchedModality,
        cross_modality: _PatchedModality | None = None,
    ) -> _PatchedTransformerArgs: ...

    def _prepare_cross_attention_timestep(
        self,
        *,
        timestep: MLXArray,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: mx.Dtype,
    ) -> tuple[MLXArray, MLXArray]: ...


class _PatchableModelType(Protocol):
    def is_audio_enabled(self) -> bool: ...

    def is_video_enabled(self) -> bool: ...


class _PatchableRuntimeConfig(Protocol):
    double_precision_rope: bool


class _PatchableModalityConfig(Protocol):
    dim: int
    heads: int
    apply_gated_attention: bool
    cross_attention_adaln: bool


class _PatchableModel(Protocol):
    _mlxr_22b_patch: bool
    model_type: _PatchableModelType
    inner_dim: int
    audio_inner_dim: int
    num_attention_heads: int
    audio_num_attention_heads: int
    positional_embedding_max_pos: list[int]
    audio_positional_embedding_max_pos: list[int]
    use_middle_indices_grid: bool
    audio_cross_attention_dim: int
    timestep_scale_multiplier: int
    av_ca_timestep_scale_multiplier: int
    positional_embedding_theta: float
    rope_type: object
    config: _PatchableRuntimeConfig
    patchify_proj: object
    audio_patchify_proj: object
    caption_projection: object | None
    audio_caption_projection: object | None
    adaln_single: object
    audio_adaln_single: object
    prompt_adaln_single: object
    audio_prompt_adaln_single: object
    av_ca_video_scale_shift_adaln_single: object
    av_ca_a2v_gate_adaln_single: object
    av_ca_audio_scale_shift_adaln_single: object
    av_ca_v2a_gate_adaln_single: object
    scale_shift_table: object
    norm_out: object
    proj_out: object
    audio_scale_shift_table: object
    audio_norm_out: object
    audio_proj_out: object
    transformer_blocks: object
    video_args_preprocessor: _PatchablePreprocessor | _PatchableMultiPreprocessor
    audio_args_preprocessor: _PatchableMultiPreprocessor

    def _process_transformer_blocks(
        self,
        *,
        video: _PatchedTransformerArgs | None,
        audio: _PatchedTransformerArgs | None,
    ) -> tuple[_PatchedTransformerArgs | None, _PatchedTransformerArgs | None]: ...

    def _process_output(
        self,
        scale_shift_table: object,
        norm_out: object,
        proj_out: object,
        x: MLXArray,
        embedded_timestep: MLXArray,
    ) -> MLXArray: ...


class _PatchableBlock(Protocol):
    norm_eps: float
    scale_shift_table: MLXArray
    prompt_scale_shift_table: MLXArray
    audio_scale_shift_table: MLXArray
    audio_prompt_scale_shift_table: MLXArray
    scale_shift_table_a2v_ca_audio: object
    scale_shift_table_a2v_ca_video: object
    attn1: _PatchableAttention
    attn2: _PatchableAttention
    audio_attn1: _PatchableAttention
    audio_attn2: _PatchableAttention
    audio_to_video_attn: _PatchableAttention
    video_to_audio_attn: _PatchableAttention
    ff: Callable[[MLXArray], MLXArray]
    audio_ff: Callable[[MLXArray], MLXArray]

    def get_ada_values(
        self,
        scale_shift_table: MLXArray,
        batch_size: int,
        timestep: MLXArray,
        block_slice: slice,
    ) -> tuple[MLXArray, MLXArray, MLXArray]: ...

    def get_av_ca_ada_values(
        self,
        scale_shift_table: object,
        batch_size: int,
        cross_scale_shift_timestep: MLXArray,
        cross_gate_timestep: MLXArray,
    ) -> tuple[MLXArray, MLXArray, MLXArray, MLXArray, MLXArray]: ...


def _is_multi_preprocessor(
    value: object,
) -> TypeGuard[_PatchableMultiPreprocessor]:
    return hasattr(value, "simple_preprocessor") and hasattr(value, "prepare")


def _patch_reference_modules(imports: _ReferenceImports) -> None:
    if getattr(imports.model_class, "_mlxr_22b_patch", False):
        return
    patch_attention = not getattr(imports.attention_class, "__module__", "").startswith(
        "mlxr."
    )
    patch_blocks = not getattr(
        imports.BasicAVTransformerBlock, "__module__", ""
    ).startswith("mlxr.")
    patch_config = not getattr(imports.model_config_class, "__module__", "").startswith(
        "mlxr."
    )
    patch_preprocessors = not getattr(
        imports.preprocessor_class, "__module__", ""
    ).startswith("mlxr.")

    original_get_video_config: Callable[[object], object | None] | None = (
        getattr(imports.model_config_class, "get_video_config")
        if patch_config
        else None
    )
    original_get_audio_config: Callable[[object], object | None] | None = (
        getattr(imports.model_config_class, "get_audio_config")
        if patch_config
        else None
    )
    original_attention_init: Callable[..., None] | None = (
        getattr(imports.attention_class, "__init__") if patch_attention else None
    )
    original_preprocessor_init: Callable[..., None] | None = (
        getattr(imports.preprocessor_class, "__init__") if patch_preprocessors else None
    )
    original_multi_preprocessor_init: Callable[..., None] | None = (
        getattr(imports.multi_preprocessor_class, "__init__")
        if patch_preprocessors
        else None
    )
    original_prepare_context: Callable[..., tuple[MLXArray, MLXArray | None]] | None = (
        getattr(imports.preprocessor_class, "_prepare_context")
        if patch_preprocessors
        else None
    )
    original_model_init_video: Callable[..., None] = getattr(
        imports.model_class, "_init_video"
    )
    original_model_init_audio: Callable[..., None] = getattr(
        imports.model_class, "_init_audio"
    )
    original_model_init_transformer_blocks: Callable[..., None] = getattr(
        imports.model_class, "_init_transformer_blocks"
    )
    original_block_init: Callable[..., None] | None = (
        getattr(imports.BasicAVTransformerBlock, "__init__") if patch_blocks else None
    )
    original_model_call: Callable[..., tuple[MLXArray | None, MLXArray | None]] = (
        getattr(imports.model_class, "__call__")
    )

    def patched_get_video_config(config_self: _TransformerConfigLike) -> object | None:
        if original_get_video_config is None:
            raise RuntimeError(
                "Expected donor video config getter when patching config"
            )
        video_config = original_get_video_config(config_self)
        if video_config is None:
            return None
        setattr(
            video_config,
            "apply_gated_attention",
            bool(getattr(config_self, "apply_gated_attention", False)),
        )
        setattr(
            video_config,
            "cross_attention_adaln",
            bool(getattr(config_self, "cross_attention_adaln", False)),
        )
        return video_config

    def patched_get_audio_config(config_self: _TransformerConfigLike) -> object | None:
        if original_get_audio_config is None:
            raise RuntimeError(
                "Expected donor audio config getter when patching config"
            )
        audio_config = original_get_audio_config(config_self)
        if audio_config is None:
            return None
        setattr(
            audio_config,
            "apply_gated_attention",
            bool(getattr(config_self, "apply_gated_attention", False)),
        )
        setattr(
            audio_config,
            "cross_attention_adaln",
            bool(getattr(config_self, "cross_attention_adaln", False)),
        )
        return audio_config

    def patched_attention_init(
        attention_self: _PatchableAttention,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: object = None,
        apply_gated_attention: bool = False,
    ) -> None:
        if original_attention_init is None:
            raise RuntimeError("Expected donor attention init when patching attention")
        if rope_type is None:
            rope_type = getattr(imports.rope_type_enum, "INTERLEAVED")
        original_attention_init(
            attention_self,
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
            rope_type=rope_type,
        )
        setattr(attention_self, "dim_head", dim_head)
        if apply_gated_attention:
            attention_self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)

    def patched_attention_call(
        attention_self: _PatchableAttention,
        x: MLXArray,
        context: MLXArray | None = None,
        mask: MLXArray | None = None,
        pe: tuple[MLXArray, MLXArray] | None = None,
        k_pe: tuple[MLXArray, MLXArray] | None = None,
    ) -> MLXArray:
        query = attention_self.to_q(x)
        context = x if context is None else context
        key = attention_self.to_k(context)
        value = attention_self.to_v(context)
        query = attention_self.q_norm(query)
        key = attention_self.k_norm(key)
        if pe is not None:
            query = imports.apply_rotary_emb(query, pe, attention_self.rope_type)
            key = imports.apply_rotary_emb(
                key,
                pe if k_pe is None else k_pe,
                attention_self.rope_type,
            )
        out = imports.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_self.heads,
            mask,
        )
        to_gate_logits = getattr(attention_self, "to_gate_logits", None)
        if to_gate_logits is not None:
            gates = 2.0 * mx.sigmoid(to_gate_logits(x))
            batch_size, seq_len, _ = out.shape
            heads = int(attention_self.heads)
            dim_head = int(getattr(attention_self, "dim_head"))
            reshaped = mx.reshape(out, (batch_size, seq_len, heads, dim_head))
            out = mx.reshape(
                reshaped * mx.expand_dims(gates, axis=-1),
                (batch_size, seq_len, heads * dim_head),
            )
        return attention_self.to_out(out)

    def patched_preprocessor_init(
        preprocessor_self: _PatchablePreprocessor,
        patchify_proj: object,
        adaln: object,
        caption_projection: object | None,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        positional_embedding_theta: float,
        rope_type: object,
        double_precision_rope: bool = False,
        prompt_adaln: object | None = None,
    ) -> None:
        if original_preprocessor_init is None:
            raise RuntimeError(
                "Expected original preprocessor init when patching donor preprocessors"
            )
        prompt_adaln_callable = prompt_adaln if callable(prompt_adaln) else None
        original_preprocessor_init(
            preprocessor_self,
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
            double_precision_rope=double_precision_rope,
        )
        preprocessor_self.prompt_adaln = prompt_adaln_callable

    def patched_multi_preprocessor_init(
        preprocessor_self: _PatchableMultiPreprocessor,
        patchify_proj: object,
        adaln: object,
        caption_projection: object | None,
        cross_scale_shift_adaln: object,
        cross_gate_adaln: object,
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
        double_precision_rope: bool = False,
        prompt_adaln: object | None = None,
    ) -> None:
        if original_multi_preprocessor_init is None:
            raise RuntimeError(
                "Expected original multimodal preprocessor init when patching donor preprocessors"
            )
        prompt_adaln_callable = prompt_adaln if callable(prompt_adaln) else None
        original_multi_preprocessor_init(
            preprocessor_self,
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            cross_scale_shift_adaln=cross_scale_shift_adaln,
            cross_gate_adaln=cross_gate_adaln,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            cross_pe_max_pos=cross_pe_max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            audio_cross_attention_dim=audio_cross_attention_dim,
            timestep_scale_multiplier=timestep_scale_multiplier,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
        )
        preprocessor_self.simple_preprocessor.prompt_adaln = prompt_adaln_callable

    def patched_multi_preprocessor_prepare(
        preprocessor_self: _PatchableMultiPreprocessor,
        modality: _PatchedModality,
        cross_modality: _PatchedModality | None = None,
    ) -> _PatchedTransformerArgs:
        simple_preprocessor = preprocessor_self.simple_preprocessor
        transformer_args = simple_preprocessor.prepare(modality)
        if cross_modality is None:
            return transformer_args

        if cross_modality.sigma.size > 1:
            if cross_modality.sigma.shape[0] != modality.timesteps.shape[0]:
                raise ValueError(
                    "Cross modality sigma must have the same batch size as the modality"
                )
            if cross_modality.sigma.ndim != 1:
                raise ValueError("Cross modality sigma must be a 1D tensor")
        cross_timestep = mx.reshape(
            cross_modality.sigma,
            (
                modality.timesteps.shape[0],
                1,
                *([1] * len(modality.timesteps.shape[2:])),
            ),
        )

        cross_pe = simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=preprocessor_self.audio_cross_attention_dim,
            max_pos=[preprocessor_self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=simple_preprocessor.num_attention_heads,
        )
        cross_scale_shift_timestep, cross_gate_timestep = (
            preprocessor_self._prepare_cross_attention_timestep(
                timestep=cross_timestep,
                timestep_scale_multiplier=simple_preprocessor.timestep_scale_multiplier,
                batch_size=transformer_args.x.shape[0],
                hidden_dtype=transformer_args.x.dtype,
            )
        )

        return _PatchedTransformerArgs(
            x=transformer_args.x,
            context=transformer_args.context,
            context_mask=transformer_args.context_mask,
            self_attention_mask=transformer_args.self_attention_mask,
            timesteps=transformer_args.timesteps,
            embedded_timestep=transformer_args.embedded_timestep,
            positional_embeddings=transformer_args.positional_embeddings,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
            enabled=transformer_args.enabled,
            prompt_timestep=transformer_args.prompt_timestep,
        )

    def patched_prepare_context(
        preprocessor_self: _PatchablePreprocessor,
        context: MLXArray,
        x: MLXArray,
        attention_mask: MLXArray | None = None,
    ) -> tuple[MLXArray, MLXArray | None]:
        if original_prepare_context is None:
            raise RuntimeError(
                "Expected original preprocessor context helper when patching donor preprocessors"
            )
        caption_projection = getattr(preprocessor_self, "caption_projection", None)
        if caption_projection is None:
            if context.ndim != 3 or int(context.shape[-1]) != int(x.shape[-1]):
                raise ValueError(
                    "LTX prompt context must already be post-connector and transformer-width "
                    f"when caption projection is disabled; got {tuple(int(size) for size in context.shape)} "
                    f"for transformer width {int(x.shape[-1])}"
                )
            return context, attention_mask
        return original_prepare_context(
            preprocessor_self,
            context=context,
            x=x,
            attention_mask=attention_mask,
        )

    def patched_prepare(
        preprocessor_self: _PatchablePreprocessor, modality: _PatchedModality
    ) -> _PatchedTransformerArgs:
        x = preprocessor_self.patchify_proj(modality.latent)
        timesteps, embedded_timestep = preprocessor_self._prepare_timestep(
            modality.timesteps,
            x.shape[0],
            hidden_dtype=x.dtype,
        )
        prompt_timestep = None
        prompt_adaln = getattr(preprocessor_self, "prompt_adaln", None)
        if callable(prompt_adaln):
            sigma_scaled = modality.sigma * preprocessor_self.timestep_scale_multiplier
            prompt_values, _ = prompt_adaln(
                mx.reshape(sigma_scaled, (-1,)),
                hidden_dtype=modality.latent.dtype,
            )
            prompt_timestep = mx.reshape(
                prompt_values, (x.shape[0], -1, prompt_values.shape[-1])
            )
        context, attention_mask = preprocessor_self._prepare_context(
            modality.context,
            x,
            modality.context_mask,
        )
        attention_mask = preprocessor_self._prepare_attention_mask(
            attention_mask,
            modality.latent.dtype,
        )
        positional_embeddings = (
            modality.positional_embeddings
            if modality.positional_embeddings is not None
            else preprocessor_self._prepare_positional_embeddings(
                positions=modality.positions,
                inner_dim=preprocessor_self.inner_dim,
                max_pos=preprocessor_self.max_pos,
                use_middle_indices_grid=preprocessor_self.use_middle_indices_grid,
                num_attention_heads=preprocessor_self.num_attention_heads,
            )
        )
        return _PatchedTransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            self_attention_mask=None,
            timesteps=timesteps,
            embedded_timestep=embedded_timestep,
            positional_embeddings=positional_embeddings,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
        )

    def patched_model_init_video(
        model_self: _PatchableModel, config: _TransformerConfigLike
    ) -> None:
        original_model_init_video(model_self, config)
        adaln_factory: Callable[..., object] = imports.adaln_class
        adaln_coefficient = (
            9 if bool(getattr(config, "cross_attention_adaln", False)) else 6
        )
        model_self.adaln_single = adaln_factory(
            model_self.inner_dim,
            embedding_coefficient=adaln_coefficient,
        )
        if bool(getattr(config, "caption_proj_before_connector", False)):
            if hasattr(model_self, "caption_projection"):
                delattr(model_self, "caption_projection")
        if bool(getattr(config, "cross_attention_adaln", False)):
            model_self.prompt_adaln_single = adaln_factory(
                model_self.inner_dim,
                embedding_coefficient=2,
            )

    def patched_model_init_audio(
        model_self: _PatchableModel, config: _TransformerConfigLike
    ) -> None:
        original_model_init_audio(model_self, config)
        adaln_factory: Callable[..., object] = imports.adaln_class
        adaln_coefficient = (
            9 if bool(getattr(config, "cross_attention_adaln", False)) else 6
        )
        model_self.audio_adaln_single = adaln_factory(
            model_self.audio_inner_dim,
            embedding_coefficient=adaln_coefficient,
        )
        if bool(getattr(config, "caption_proj_before_connector", False)):
            if hasattr(model_self, "audio_caption_projection"):
                delattr(model_self, "audio_caption_projection")
        if bool(getattr(config, "cross_attention_adaln", False)):
            model_self.audio_prompt_adaln_single = adaln_factory(
                model_self.audio_inner_dim,
                embedding_coefficient=2,
            )

    def patched_model_init_preprocessors(
        model_self: _PatchableModel,
        config: _TransformerConfigLike,
        cross_pe_max_pos: object = None,
    ) -> None:
        multi_preprocessor_factory = getattr(imports, "multi_preprocessor_class")
        preprocessor_factory = getattr(imports, "preprocessor_class")
        if model_self.model_type.is_audio_enabled():
            effective_cross_pe_max_pos = cross_pe_max_pos
            if effective_cross_pe_max_pos is None:
                effective_cross_pe_max_pos = max(
                    model_self.positional_embedding_max_pos[0],
                    model_self.audio_positional_embedding_max_pos[0],
                )
            model_self.video_args_preprocessor = multi_preprocessor_factory(
                patchify_proj=model_self.patchify_proj,
                adaln=model_self.adaln_single,
                caption_projection=getattr(model_self, "caption_projection", None),
                cross_scale_shift_adaln=model_self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=model_self.av_ca_a2v_gate_adaln_single,
                inner_dim=model_self.inner_dim,
                max_pos=model_self.positional_embedding_max_pos,
                num_attention_heads=model_self.num_attention_heads,
                cross_pe_max_pos=effective_cross_pe_max_pos,
                use_middle_indices_grid=model_self.use_middle_indices_grid,
                audio_cross_attention_dim=model_self.audio_cross_attention_dim,
                timestep_scale_multiplier=model_self.timestep_scale_multiplier,
                positional_embedding_theta=model_self.positional_embedding_theta,
                rope_type=model_self.rope_type,
                av_ca_timestep_scale_multiplier=model_self.av_ca_timestep_scale_multiplier,
                double_precision_rope=model_self.config.double_precision_rope,
                prompt_adaln=getattr(model_self, "prompt_adaln_single", None),
            )
            model_self.audio_args_preprocessor = multi_preprocessor_factory(
                patchify_proj=model_self.audio_patchify_proj,
                adaln=model_self.audio_adaln_single,
                caption_projection=getattr(
                    model_self, "audio_caption_projection", None
                ),
                cross_scale_shift_adaln=model_self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=model_self.av_ca_v2a_gate_adaln_single,
                inner_dim=model_self.audio_inner_dim,
                max_pos=model_self.audio_positional_embedding_max_pos,
                num_attention_heads=model_self.audio_num_attention_heads,
                cross_pe_max_pos=effective_cross_pe_max_pos,
                use_middle_indices_grid=model_self.use_middle_indices_grid,
                audio_cross_attention_dim=model_self.audio_cross_attention_dim,
                timestep_scale_multiplier=model_self.timestep_scale_multiplier,
                positional_embedding_theta=model_self.positional_embedding_theta,
                rope_type=model_self.rope_type,
                av_ca_timestep_scale_multiplier=model_self.av_ca_timestep_scale_multiplier,
                double_precision_rope=model_self.config.double_precision_rope,
                prompt_adaln=getattr(model_self, "audio_prompt_adaln_single", None),
            )
            return
        model_self.video_args_preprocessor = preprocessor_factory(
            patchify_proj=model_self.patchify_proj,
            adaln=model_self.adaln_single,
            caption_projection=getattr(model_self, "caption_projection", None),
            inner_dim=model_self.inner_dim,
            max_pos=model_self.positional_embedding_max_pos,
            num_attention_heads=model_self.num_attention_heads,
            use_middle_indices_grid=model_self.use_middle_indices_grid,
            timestep_scale_multiplier=model_self.timestep_scale_multiplier,
            positional_embedding_theta=model_self.positional_embedding_theta,
            rope_type=model_self.rope_type,
            double_precision_rope=model_self.config.double_precision_rope,
            prompt_adaln=getattr(model_self, "prompt_adaln_single", None),
        )

    def patched_model_init_transformer_blocks(
        model_self: _PatchableModel, config: _TransformerConfigLike
    ) -> None:
        if model_self.model_type.is_audio_enabled():
            original_model_init_transformer_blocks(model_self, config)
            return
        video_config = config.get_video_config()
        block_factory = getattr(imports, "BasicAVTransformerBlock")
        model_self.transformer_blocks = {
            index: block_factory(
                idx=index,
                video=video_config,
                rope_type=config.rope_type,
                norm_eps=config.norm_eps,
            )
            for index in range(config.num_layers)
        }

    def patched_block_init(
        block_self: _PatchableBlock,
        idx: int,
        video: _PatchableModalityConfig | None = None,
        audio: _PatchableModalityConfig | None = None,
        rope_type: object = None,
        norm_eps: float = 1e-6,
    ) -> None:
        if original_block_init is None:
            raise RuntimeError("Expected donor block init when patching block")
        if rope_type is None:
            rope_type = getattr(imports.rope_type_enum, "INTERLEAVED")
        original_block_init(
            block_self,
            idx=idx,
            video=video,
            audio=audio,
            rope_type=rope_type,
            norm_eps=norm_eps,
        )
        if video is not None and bool(getattr(video, "apply_gated_attention", False)):
            block_self.attn1.to_gate_logits = nn.Linear(
                video.dim, video.heads, bias=True
            )
            block_self.attn2.to_gate_logits = nn.Linear(
                video.dim, video.heads, bias=True
            )
            if hasattr(block_self, "audio_to_video_attn"):
                block_self.audio_to_video_attn.to_gate_logits = nn.Linear(
                    video.dim,
                    audio.heads if audio is not None else video.heads,
                    bias=True,
                )
        if video is not None and bool(getattr(video, "cross_attention_adaln", False)):
            block_self.scale_shift_table = mx.zeros((9, video.dim))
            block_self.prompt_scale_shift_table = mx.zeros((2, video.dim))
        if audio is not None and bool(getattr(audio, "apply_gated_attention", False)):
            block_self.audio_attn1.to_gate_logits = nn.Linear(
                audio.dim, audio.heads, bias=True
            )
            block_self.audio_attn2.to_gate_logits = nn.Linear(
                audio.dim, audio.heads, bias=True
            )
            if hasattr(block_self, "video_to_audio_attn"):
                block_self.video_to_audio_attn.to_gate_logits = nn.Linear(
                    audio.dim, audio.heads, bias=True
                )
        if audio is not None and bool(getattr(audio, "cross_attention_adaln", False)):
            block_self.audio_scale_shift_table = mx.zeros((9, audio.dim))
            block_self.audio_prompt_scale_shift_table = mx.zeros((2, audio.dim))

    def apply_cross_attention_adaln(
        *,
        block: _PatchableBlock,
        x: MLXArray,
        context: MLXArray,
        attn: _PatchableAttention,
        scale_shift_table: MLXArray,
        prompt_scale_shift_table: MLXArray,
        timestep: MLXArray,
        prompt_timestep: MLXArray | None,
        context_mask: MLXArray | None,
        norm_eps: float,
    ) -> MLXArray:
        if prompt_timestep is None:
            raise ValueError(
                "LTX prompt timestep is required for cross-attention AdaLN"
            )
        q_shift, q_scale, q_gate = block.get_ada_values(
            scale_shift_table, x.shape[0], timestep, slice(6, 9)
        )
        batch_size = x.shape[0]
        prompt_values = prompt_scale_shift_table[None, None] + mx.reshape(
            prompt_timestep,
            (batch_size, prompt_timestep.shape[1], 2, -1),
        )
        shift_kv = prompt_values[:, :, 0, :]
        scale_kv = prompt_values[:, :, 1, :]
        attn_input = imports.rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
        return (
            attn(
                attn_input,
                context=encoder_hidden_states,
                mask=context_mask,
            )
            * q_gate
        )

    def patched_block_call(
        block_self: _PatchableBlock,
        video: _PatchedTransformerArgs | None = None,
        audio: _PatchedTransformerArgs | None = None,
    ) -> tuple[_PatchedTransformerArgs | None, _PatchedTransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None
        run_vx = video is not None and vx is not None and video.enabled and vx.size > 0
        run_ax = audio is not None and ax is not None and audio.enabled and ax.size > 0
        run_a2v = run_vx and run_ax
        run_v2a = run_ax and run_vx

        if run_vx and video is not None and vx is not None:
            video_args = video
            video_x = vx
            vshift_msa, vscale_msa, vgate_msa = block_self.get_ada_values(
                block_self.scale_shift_table,
                video_x.shape[0],
                video_args.timesteps,
                slice(0, 3),
            )
            norm_vx = (
                imports.rms_norm(video_x, eps=block_self.norm_eps) * (1 + vscale_msa)
                + vshift_msa
            )
            vx = (
                video_x
                + block_self.attn1(norm_vx, pe=video_args.positional_embeddings)
                * vgate_msa
            )
            if hasattr(block_self, "prompt_scale_shift_table"):
                vx = vx + apply_cross_attention_adaln(
                    block=block_self,
                    x=vx,
                    context=video_args.context,
                    attn=block_self.attn2,
                    scale_shift_table=block_self.scale_shift_table,
                    prompt_scale_shift_table=block_self.prompt_scale_shift_table,
                    timestep=video_args.timesteps,
                    prompt_timestep=video_args.prompt_timestep,
                    context_mask=video_args.context_mask,
                    norm_eps=block_self.norm_eps,
                )
            else:
                vx = vx + block_self.attn2(
                    imports.rms_norm(vx, eps=block_self.norm_eps),
                    context=video_args.context,
                    mask=video_args.context_mask,
                )

        if run_ax and audio is not None and ax is not None:
            audio_args = audio
            audio_x = ax
            ashift_msa, ascale_msa, agate_msa = block_self.get_ada_values(
                block_self.audio_scale_shift_table,
                audio_x.shape[0],
                audio_args.timesteps,
                slice(0, 3),
            )
            norm_ax = (
                imports.rms_norm(audio_x, eps=block_self.norm_eps) * (1 + ascale_msa)
                + ashift_msa
            )
            ax = (
                audio_x
                + block_self.audio_attn1(norm_ax, pe=audio_args.positional_embeddings)
                * agate_msa
            )
            if hasattr(block_self, "audio_prompt_scale_shift_table"):
                ax = ax + apply_cross_attention_adaln(
                    block=block_self,
                    x=ax,
                    context=audio_args.context,
                    attn=block_self.audio_attn2,
                    scale_shift_table=block_self.audio_scale_shift_table,
                    prompt_scale_shift_table=block_self.audio_prompt_scale_shift_table,
                    timestep=audio_args.timesteps,
                    prompt_timestep=audio_args.prompt_timestep,
                    context_mask=audio_args.context_mask,
                    norm_eps=block_self.norm_eps,
                )
            else:
                ax = ax + block_self.audio_attn2(
                    imports.rms_norm(ax, eps=block_self.norm_eps),
                    context=audio_args.context,
                    mask=audio_args.context_mask,
                )

        if (run_a2v or run_v2a) and video is not None and audio is not None:
            missing_cross_inputs = (
                vx is None
                or ax is None
                or video.cross_scale_shift_timestep is None
                or video.cross_gate_timestep is None
                or audio.cross_scale_shift_timestep is None
                or audio.cross_gate_timestep is None
                or video.cross_positional_embeddings is None
                or audio.cross_positional_embeddings is None
            )
            if missing_cross_inputs:
                raise RuntimeError(
                    "LTX AV cross-attention requested but cross-modality preparation is incomplete"
                )

        if (
            (run_a2v or run_v2a)
            and video is not None
            and audio is not None
            and vx is not None
            and ax is not None
            and video.cross_scale_shift_timestep is not None
            and video.cross_gate_timestep is not None
            and audio.cross_scale_shift_timestep is not None
            and audio.cross_gate_timestep is not None
            and video.cross_positional_embeddings is not None
            and audio.cross_positional_embeddings is not None
        ):
            video_args = video
            audio_args = audio
            video_cross_scale_shift_timestep = video_args.cross_scale_shift_timestep
            video_cross_gate_timestep = video_args.cross_gate_timestep
            audio_cross_scale_shift_timestep = audio_args.cross_scale_shift_timestep
            audio_cross_gate_timestep = audio_args.cross_gate_timestep
            assert video_cross_scale_shift_timestep is not None
            assert video_cross_gate_timestep is not None
            assert audio_cross_scale_shift_timestep is not None
            assert audio_cross_gate_timestep is not None
            vx_norm3 = imports.rms_norm(vx, eps=block_self.norm_eps)
            ax_norm3 = imports.rms_norm(ax, eps=block_self.norm_eps)
            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_out_v2a,
            ) = block_self.get_av_ca_ada_values(
                block_self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio_cross_scale_shift_timestep,
                audio_cross_gate_timestep,
            )
            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_out_a2v,
            ) = block_self.get_av_ca_ada_values(
                block_self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video_cross_scale_shift_timestep,
                video_cross_gate_timestep,
            )
            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                vx = vx + (
                    block_self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video_args.cross_positional_embeddings,
                        k_pe=audio_args.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                )
            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                ax = ax + (
                    block_self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio_args.cross_positional_embeddings,
                        k_pe=video_args.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                )

        if run_vx and video is not None and vx is not None:
            vshift_mlp, vscale_mlp, vgate_mlp = block_self.get_ada_values(
                block_self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = (
                imports.rms_norm(vx, eps=block_self.norm_eps) * (1 + vscale_mlp)
                + vshift_mlp
            )
            vx = vx + block_self.ff(vx_scaled) * vgate_mlp

        if run_ax and audio is not None and ax is not None:
            ashift_mlp, ascale_mlp, agate_mlp = block_self.get_ada_values(
                block_self.audio_scale_shift_table,
                ax.shape[0],
                audio.timesteps,
                slice(3, 6),
            )
            ax_scaled = (
                imports.rms_norm(ax, eps=block_self.norm_eps) * (1 + ascale_mlp)
                + ashift_mlp
            )
            ax = ax + block_self.audio_ff(ax_scaled) * agate_mlp

        return (
            replace(video, x=vx) if video is not None and vx is not None else None,
            replace(audio, x=ax) if audio is not None and ax is not None else None,
        )

    def patched_model_call(
        model_self: _PatchableModel,
        video: _PatchedModality | None = None,
        audio: _PatchedModality | None = None,
    ) -> tuple[MLXArray | None, MLXArray | None]:
        if not model_self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not model_self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")
        if (
            not model_self.model_type.is_audio_enabled()
            or not model_self.model_type.is_video_enabled()
        ):
            return original_model_call(model_self, video=video, audio=audio)
        if not _is_multi_preprocessor(model_self.video_args_preprocessor):
            raise RuntimeError("Expected multi-modal video preprocessor")
        if not _is_multi_preprocessor(model_self.audio_args_preprocessor):
            raise RuntimeError("Expected multi-modal audio preprocessor")
        video_preprocessor = model_self.video_args_preprocessor
        audio_preprocessor = model_self.audio_args_preprocessor
        video_args = (
            video_preprocessor.prepare(video, audio) if video is not None else None
        )
        audio_args = (
            audio_preprocessor.prepare(audio, video) if audio is not None else None
        )
        video_out, audio_out = model_self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
        )
        vx = (
            model_self._process_output(
                model_self.scale_shift_table,
                model_self.norm_out,
                model_self.proj_out,
                video_out.x,
                video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )
        ax = (
            model_self._process_output(
                model_self.audio_scale_shift_table,
                model_self.audio_norm_out,
                model_self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax

    if patch_config:
        setattr(
            imports.model_config_class, "get_video_config", patched_get_video_config
        )
        setattr(
            imports.model_config_class, "get_audio_config", patched_get_audio_config
        )
    if patch_attention:
        setattr(imports.attention_class, "__init__", patched_attention_init)
        setattr(imports.attention_class, "__call__", patched_attention_call)
    if patch_preprocessors:
        setattr(imports.preprocessor_class, "__init__", patched_preprocessor_init)
        setattr(
            imports.multi_preprocessor_class,
            "__init__",
            patched_multi_preprocessor_init,
        )
        setattr(
            imports.multi_preprocessor_class,
            "prepare",
            patched_multi_preprocessor_prepare,
        )
        setattr(imports.preprocessor_class, "_prepare_context", patched_prepare_context)
        setattr(imports.preprocessor_class, "prepare", patched_prepare)
    setattr(imports.model_class, "_init_video", patched_model_init_video)
    setattr(imports.model_class, "_init_audio", patched_model_init_audio)
    setattr(
        imports.model_class,
        "_init_preprocessors",
        patched_model_init_preprocessors,
    )
    setattr(
        imports.model_class,
        "_init_transformer_blocks",
        patched_model_init_transformer_blocks,
    )
    setattr(imports.model_class, "__call__", patched_model_call)
    if patch_blocks:
        setattr(imports.BasicAVTransformerBlock, "__init__", patched_block_init)
        setattr(imports.BasicAVTransformerBlock, "__call__", patched_block_call)
    setattr(imports.model_class, "_mlxr_22b_patch", True)

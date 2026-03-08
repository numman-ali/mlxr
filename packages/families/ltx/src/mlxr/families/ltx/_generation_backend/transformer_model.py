from __future__ import annotations

from pathlib import Path
from typing import Mapping

import mlx.core as mx
from mlx.utils import tree_flatten

from .. import _nn_compat as nn
from .adaln_ops import AdaLayerNormSingle, adaln_embedding_coefficient
from .model_config import LTXModelConfig
from .primitives import to_denoised
from .rope_ops import LTXRopeType
from .text_projection import TextProjection
from .transformer_blocks import BasicAVTransformerBlock
from .transformer_preprocessors import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgsPreprocessor,
)
from .types import MLXArray, _PatchedModality, _PatchedTransformerArgs


class LTXModel(nn.Module):
    _mlxr_22b_patch = True

    def __init__(self, config: LTXModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.use_middle_indices_grid = config.use_middle_indices_grid
        self.rope_type = config.rope_type
        self.double_precision_rope = config.double_precision_rope
        self.timestep_scale_multiplier = config.timestep_scale_multiplier
        self.positional_embedding_theta = config.positional_embedding_theta
        self.video_args_preprocessor: (
            TransformerArgsPreprocessor | MultiModalTransformerArgsPreprocessor
        )
        self.audio_args_preprocessor: (
            TransformerArgsPreprocessor | MultiModalTransformerArgsPreprocessor
        )

        video_max_pos = config.positional_embedding_max_pos
        audio_max_pos = config.audio_positional_embedding_max_pos
        if video_max_pos is None or audio_max_pos is None:
            raise RuntimeError("Owned LTX model config must resolve positional limits")

        cross_pe_max_pos: int | None = None
        if config.model_type.is_video_enabled():
            self.positional_embedding_max_pos = video_max_pos
            self.num_attention_heads = config.num_attention_heads
            self.inner_dim = config.inner_dim
            self._init_video(config)

        if config.model_type.is_audio_enabled():
            self.audio_positional_embedding_max_pos = audio_max_pos
            self.audio_num_attention_heads = config.audio_num_attention_heads
            self.audio_inner_dim = config.audio_inner_dim
            self._init_audio(config)

        if (
            config.model_type.is_video_enabled()
            and config.model_type.is_audio_enabled()
        ):
            cross_pe_max_pos = max(video_max_pos[0], audio_max_pos[0])
            self.av_ca_timestep_scale_multiplier = (
                config.av_ca_timestep_scale_multiplier
            )
            self.audio_cross_attention_dim = config.audio_cross_attention_dim
            self._init_audio_video()

        self._init_preprocessors(cross_pe_max_pos)
        self._init_transformer_blocks(config)

    def _init_video(self, config: LTXModelConfig) -> None:
        self.patchify_proj = nn.Linear(config.in_channels, self.inner_dim, bias=True)
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=adaln_embedding_coefficient(
                config.cross_attention_adaln
            ),
        )
        if config.cross_attention_adaln:
            self.prompt_adaln_single = AdaLayerNormSingle(
                self.inner_dim,
                embedding_coefficient=2,
            )
        if not config.caption_proj_before_connector:
            self.caption_projection = TextProjection(
                config.caption_channels,
                self.inner_dim,
            )
        self.scale_shift_table = mx.zeros((2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=config.norm_eps, affine=False)
        self.proj_out = nn.Linear(self.inner_dim, config.out_channels)

    def _init_audio(self, config: LTXModelConfig) -> None:
        self.audio_patchify_proj = nn.Linear(
            config.audio_in_channels, self.audio_inner_dim, bias=True
        )
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=adaln_embedding_coefficient(
                config.cross_attention_adaln
            ),
        )
        if config.cross_attention_adaln:
            self.audio_prompt_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim,
                embedding_coefficient=2,
            )
        if not config.caption_proj_before_connector:
            self.audio_caption_projection = TextProjection(
                config.audio_caption_channels,
                self.audio_inner_dim,
            )
        self.audio_scale_shift_table = mx.zeros((2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(
            self.audio_inner_dim,
            eps=config.norm_eps,
            affine=False,
        )
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, config.audio_out_channels)

    def _init_audio_video(self) -> None:
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=4,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=4,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(self, cross_pe_max_pos: int | None) -> None:
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            assert cross_pe_max_pos is not None
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=getattr(self, "caption_projection", None),
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                double_precision_rope=self.config.double_precision_rope,
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=getattr(self, "audio_caption_projection", None),
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                double_precision_rope=self.config.double_precision_rope,
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
            return

        if self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=getattr(self, "caption_projection", None),
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                double_precision_rope=self.config.double_precision_rope,
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )

        if self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=getattr(self, "audio_caption_projection", None),
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                double_precision_rope=self.config.double_precision_rope,
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(self, config: LTXModelConfig) -> None:
        video_config = config.get_video_config()
        audio_config = config.get_audio_config()
        self._transformer_blocks = {
            idx: BasicAVTransformerBlock(
                idx=idx,
                video=video_config,
                audio=audio_config,
                rope_type=LTXRopeType.from_value(config.rope_type),
                norm_eps=config.norm_eps,
            )
            for idx in range(config.num_layers)
        }

    @property
    def transformer_blocks(self) -> Mapping[int, object]:
        return self._transformer_blocks

    def _process_transformer_blocks(
        self,
        *,
        video: _PatchedTransformerArgs | None,
        audio: _PatchedTransformerArgs | None,
    ) -> tuple[_PatchedTransformerArgs | None, _PatchedTransformerArgs | None]:
        for block in self._transformer_blocks.values():
            video, audio = block(video=video, audio=audio)
        return video, audio

    def _process_output(
        self,
        scale_shift_table: MLXArray,
        norm_out: nn.LayerNorm,
        proj_out: nn.Linear,
        x: MLXArray,
        embedded_timestep: MLXArray,
    ) -> MLXArray:
        scale_shift_values = (
            scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        output = norm_out(x)
        output = output * (1 + scale) + shift
        return proj_out(output)

    def __call__(
        self,
        *,
        video: _PatchedModality | None = None,
        audio: _PatchedModality | None = None,
    ) -> tuple[MLXArray | None, MLXArray | None]:
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            video_preprocessor = self.video_args_preprocessor
            audio_preprocessor = self.audio_args_preprocessor
            if not isinstance(
                video_preprocessor, MultiModalTransformerArgsPreprocessor
            ):
                raise RuntimeError("Expected multimodal video preprocessor")
            if not isinstance(
                audio_preprocessor, MultiModalTransformerArgsPreprocessor
            ):
                raise RuntimeError("Expected multimodal audio preprocessor")
            video_args = (
                video_preprocessor.prepare(video, audio) if video is not None else None
            )
            audio_args = (
                audio_preprocessor.prepare(audio, video) if audio is not None else None
            )
        else:
            video_preprocessor = self.video_args_preprocessor
            audio_preprocessor = self.audio_args_preprocessor
            if video is not None and not isinstance(
                video_preprocessor, TransformerArgsPreprocessor
            ):
                raise RuntimeError("Expected single-modality video preprocessor")
            if audio is not None and not isinstance(
                audio_preprocessor, TransformerArgsPreprocessor
            ):
                raise RuntimeError("Expected single-modality audio preprocessor")
            video_args = (
                video_preprocessor.prepare(video) if video is not None else None
            )
            audio_args = (
                audio_preprocessor.prepare(audio) if audio is not None else None
            )

        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
        )
        video_latents = (
            self._process_output(
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
                video_out.x,
                video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )
        audio_latents = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return video_latents, audio_latents

    @staticmethod
    def _sanitize_key(key: str) -> str | None:
        if not key.startswith("model.diffusion_model."):
            return None
        if "audio_embeddings_connector" in key or "video_embeddings_connector" in key:
            return None
        sanitized = key.replace("model.diffusion_model.", "")
        replacements = (
            (".to_out.0.", ".to_out."),
            (".ff.net.0.proj.", ".ff.proj_in."),
            (".ff.net.2.", ".ff.proj_out."),
            (".audio_ff.net.0.proj.", ".audio_ff.proj_in."),
            (".audio_ff.net.2.", ".audio_ff.proj_out."),
            (".linear_1.", ".linear1."),
            (".linear_2.", ".linear2."),
        )
        for old, new in replacements:
            sanitized = sanitized.replace(old, new)
        return sanitized

    @classmethod
    def sanitize(cls, weights: Mapping[str, MLXArray]) -> dict[str, MLXArray]:
        sanitized: dict[str, MLXArray] = {}
        for key, value in weights.items():
            sanitized_key = cls._sanitize_key(key)
            if sanitized_key is not None:
                sanitized[sanitized_key] = value
        return sanitized

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Path,
        *,
        config: LTXModelConfig,
        strict: bool = True,
        weights_override: Mapping[str, MLXArray] | None = None,
    ) -> "LTXModel":
        weights = (
            dict(weights_override)
            if weights_override is not None
            else mx.load(str(checkpoint_path))
        )
        if not isinstance(weights, dict):
            raise RuntimeError(
                f"LTX checkpoint '{checkpoint_path}' did not load into a weight mapping"
            )
        sanitized = cls.sanitize(weights)
        if any(key.endswith(".scales") or key.endswith(".biases") for key in sanitized):
            raise RuntimeError(
                "Owned LTX transformer loader does not yet support pre-quantized transformer weights"
            )
        model = cls(config)
        if strict:
            parameter_tree = tree_flatten(model.parameters(), destination={})
            if not isinstance(parameter_tree, dict):
                raise RuntimeError(
                    "Owned LTX transformer loader expected a parameter mapping"
                )
            expected_keys = set(parameter_tree.keys())
            filtered_weights = {
                key: value for key, value in sanitized.items() if key in expected_keys
            }
            missing = expected_keys.difference(filtered_weights)
            if missing:
                sample = ", ".join(sorted(missing)[:5])
                raise RuntimeError(
                    "Owned LTX transformer loader is missing required weights: "
                    + sample
                )
            model.load_weights(list(filtered_weights.items()), strict=False)
            return model

        model.load_weights(list(sanitized.items()), strict=False)
        return model


class X0Model(nn.Module):
    def __init__(self, velocity_model: LTXModel) -> None:
        super().__init__()
        self.velocity_model = velocity_model

    def __call__(
        self,
        *,
        video: _PatchedModality | None = None,
        audio: _PatchedModality | None = None,
    ) -> tuple[MLXArray | None, MLXArray | None]:
        vx, ax = self.velocity_model(video=video, audio=audio)
        denoised_video = (
            to_denoised(video.latent, vx, video.timesteps)
            if vx is not None and video is not None
            else None
        )
        denoised_audio = (
            to_denoised(audio.latent, ax, audio.timesteps)
            if ax is not None and audio is not None
            else None
        )
        return denoised_video, denoised_audio

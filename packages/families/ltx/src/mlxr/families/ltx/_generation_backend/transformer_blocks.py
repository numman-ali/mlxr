from __future__ import annotations

from dataclasses import replace

import mlx.core as mx

from .. import _nn_compat as nn
from .attention_ops import Attention
from .feed_forward import FeedForward
from .model_config import TransformerConfig
from .primitives import rms_norm
from .rope_ops import LTXRopeType
from .types import MLXArray, _PatchedTransformerArgs


class BasicAVTransformerBlock(nn.Module):
    def __init__(
        self,
        idx: int,
        *,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType | str = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps
        resolved_rope_type = LTXRopeType.from_value(rope_type)

        if video is not None:
            self.attn1 = Attention(
                video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=resolved_rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=resolved_rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = mx.zeros(
                (9 if video.cross_attention_adaln else 6, video.dim)
            )
            if video.cross_attention_adaln:
                self.prompt_scale_shift_table = mx.zeros((2, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=resolved_rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=resolved_rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = mx.zeros(
                (9 if audio.cross_attention_adaln else 6, audio.dim)
            )
            if audio.cross_attention_adaln:
                self.audio_prompt_scale_shift_table = mx.zeros((2, audio.dim))

        if video is not None and audio is not None:
            self.audio_to_video_attn = Attention(
                video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=resolved_rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.video_to_audio_attn = Attention(
                audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=resolved_rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.scale_shift_table_a2v_ca_audio = mx.zeros((5, audio.dim))
            self.scale_shift_table_a2v_ca_video = mx.zeros((5, video.dim))

        self.cross_attention_adaln = (
            video is not None and video.cross_attention_adaln
        ) or (audio is not None and audio.cross_attention_adaln)

    def get_ada_values(
        self,
        scale_shift_table: MLXArray,
        batch_size: int,
        timestep: MLXArray,
        indices: slice,
    ) -> tuple[MLXArray, ...]:
        num_ada_params = int(scale_shift_table.shape[0])
        values = (
            scale_shift_table[indices][None, None]
            + mx.reshape(timestep, (batch_size, timestep.shape[1], num_ada_params, -1))[
                :, :, indices, :
            ]
        )
        return tuple(values[:, :, index, :] for index in range(values.shape[2]))

    def get_av_ca_ada_values(
        self,
        scale_shift_table: MLXArray,
        batch_size: int,
        scale_shift_timestep: MLXArray,
        gate_timestep: MLXArray,
    ) -> tuple[MLXArray, MLXArray, MLXArray, MLXArray, MLXArray]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:4, :],
            batch_size,
            scale_shift_timestep,
            slice(None, None),
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[4:, :],
            batch_size,
            gate_timestep,
            slice(None, None),
        )
        scale_shift_squeezed = tuple(
            mx.squeeze(value, axis=1) if value.shape[1] == 1 else value
            for value in scale_shift_ada_values
        )
        gate_squeezed = tuple(
            mx.squeeze(value, axis=1) if value.shape[1] == 1 else value
            for value in gate_ada_values
        )
        scale_1, shift_1, scale_2, shift_2 = scale_shift_squeezed
        (gate,) = gate_squeezed
        return scale_1, shift_1, scale_2, shift_2, gate

    def _apply_text_cross_attention(
        self,
        *,
        x: MLXArray,
        context: MLXArray,
        attn: Attention,
        scale_shift_table: MLXArray,
        prompt_scale_shift_table: MLXArray | None,
        timestep: MLXArray,
        prompt_timestep: MLXArray | None,
        context_mask: MLXArray | None,
    ) -> MLXArray:
        if prompt_scale_shift_table is not None:
            if prompt_timestep is None:
                raise ValueError(
                    "Prompt timestep is required for cross-attention AdaLN"
                )
            q_shift, q_scale, q_gate = self.get_ada_values(
                scale_shift_table,
                x.shape[0],
                timestep,
                slice(6, 9),
            )
            prompt_values = prompt_scale_shift_table[None, None] + mx.reshape(
                prompt_timestep,
                (x.shape[0], prompt_timestep.shape[1], 2, -1),
            )
            shift_kv = prompt_values[:, :, 0, :]
            scale_kv = prompt_values[:, :, 1, :]
            attn_input = rms_norm(x, eps=self.norm_eps) * (1 + q_scale) + q_shift
            encoder_hidden_states = context * (1 + scale_kv) + shift_kv
            return (
                attn(
                    attn_input,
                    context=encoder_hidden_states,
                    mask=context_mask,
                )
                * q_gate
            )

        return attn(
            rms_norm(x, eps=self.norm_eps),
            context=context,
            mask=context_mask,
        )

    def __call__(
        self,
        *,
        video: _PatchedTransformerArgs | None = None,
        audio: _PatchedTransformerArgs | None = None,
    ) -> tuple[_PatchedTransformerArgs | None, _PatchedTransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None
        run_vx = video is not None and vx is not None and video.enabled and vx.size > 0
        run_ax = audio is not None and ax is not None and audio.enabled and ax.size > 0

        if run_vx and video is not None and vx is not None:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table,
                vx.shape[0],
                video.timesteps,
                slice(0, 3),
            )
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            vx = (
                vx
                + self.attn1(
                    norm_vx,
                    pe=video.positional_embeddings,
                    mask=video.self_attention_mask,
                )
                * vgate_msa
            )
            vx = vx + self._apply_text_cross_attention(
                x=vx,
                context=video.context,
                attn=self.attn2,
                scale_shift_table=self.scale_shift_table,
                prompt_scale_shift_table=getattr(
                    self, "prompt_scale_shift_table", None
                ),
                timestep=video.timesteps,
                prompt_timestep=video.prompt_timestep,
                context_mask=video.context_mask,
            )

        if run_ax and audio is not None and ax is not None:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table,
                ax.shape[0],
                audio.timesteps,
                slice(0, 3),
            )
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            ax = (
                ax
                + self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    mask=audio.self_attention_mask,
                )
                * agate_msa
            )
            ax = ax + self._apply_text_cross_attention(
                x=ax,
                context=audio.context,
                attn=self.audio_attn2,
                scale_shift_table=self.audio_scale_shift_table,
                prompt_scale_shift_table=getattr(
                    self, "audio_prompt_scale_shift_table", None
                ),
                timestep=audio.timesteps,
                prompt_timestep=audio.prompt_timestep,
                context_mask=audio.context_mask,
            )

        if (
            run_vx
            and run_ax
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
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )
            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
            ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
            vx = vx + (
                self.audio_to_video_attn(
                    vx_scaled,
                    context=ax_scaled,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                )
                * gate_out_a2v
            )

            ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
            vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
            ax = ax + (
                self.video_to_audio_attn(
                    ax_scaled,
                    context=vx_scaled,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                )
                * gate_out_v2a
            )

        if run_vx and video is not None and vx is not None:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table,
                vx.shape[0],
                video.timesteps,
                slice(3, 6),
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

        if run_ax and audio is not None and ax is not None:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table,
                ax.shape[0],
                audio.timesteps,
                slice(3, 6),
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        return (
            replace(video, x=vx) if video is not None and vx is not None else None,
            replace(audio, x=ax) if audio is not None and ax is not None else None,
        )

from __future__ import annotations

from contextlib import nullcontext

import mlx.core as mx
from mlxr.core.runtime import TraceRecorder, mlx_memory_snapshot

from ..prompt_encoding import PromptEncodingResult
from .conditioning import _attention_mask, _context_width
from .debug import _debug_progress_enabled
from .types import (
    MLXArray,
    _AudioVideoTransformer,
    _LatentStateLike,
    _PatchedModality,
    _ReferenceImports,
    _RuntimeModelConfig,
)

_VIDEO_CFG_SCALE = 3.0
_AUDIO_CFG_SCALE = 7.0
_GUIDANCE_RESCALE_SCALE = 0.7


def _sync_optional_arrays(*arrays: MLXArray | None) -> None:
    realized = tuple(array for array in arrays if array is not None)
    if realized:
        mx.eval(*realized)


def _optional_context_width(value: object) -> int | None:
    if value is None:
        return None
    return _context_width(value)


def _guided_prediction(
    conditioned: MLXArray,
    unconditioned: MLXArray | None,
    *,
    scale: float,
    rescale_scale: float,
) -> MLXArray:
    if unconditioned is None or scale == 1.0:
        return conditioned
    predicted = conditioned + (scale - 1.0) * (conditioned - unconditioned)
    if rescale_scale == 0.0:
        return predicted
    conditioned_std = mx.std(conditioned)
    predicted_std = mx.std(predicted)
    scale_factor = conditioned_std / mx.maximum(
        predicted_std,
        mx.array(1e-6, dtype=predicted_std.dtype),
    )
    rescaled_factor = rescale_scale * scale_factor + (1.0 - rescale_scale)
    return predicted * rescaled_factor


def _assert_prompt_runtime_contract(
    prompt_context: PromptEncodingResult,
    runtime_config: _RuntimeModelConfig,
) -> None:
    if prompt_context.context_representation != "post_connector":
        raise ValueError(
            "LTX real generation requires post-connector prompt context for the "
            "current 22B bridge"
        )
    context_width = _context_width(prompt_context.video_context)
    if prompt_context.transformer_context_dim is not None:
        if prompt_context.transformer_context_dim != context_width:
            raise ValueError(
                "LTX prompt contract mismatch: prompt context width "
                f"{context_width} does not match declared transformer_context_dim "
                f"{prompt_context.transformer_context_dim}"
            )
        if prompt_context.transformer_context_dim != runtime_config.cross_attention_dim:
            raise ValueError(
                "LTX prompt/generation config mismatch: prompt transformer_context_dim "
                f"{prompt_context.transformer_context_dim} != runtime cross_attention_dim "
                f"{runtime_config.cross_attention_dim}"
            )
    if prompt_context.audio_context is None:
        raise ValueError(
            "LTX real generation requires audio_context from prompt_encode for the "
            "current 22B audio-video bridge"
        )
    if _attention_mask(prompt_context) is not None:
        raise ValueError(
            "LTX real generation requires an all-valid post-connector prompt mask "
            "for the current 22B audio-video bridge"
        )
    audio_context_width = _context_width(prompt_context.audio_context)
    if audio_context_width != runtime_config.audio_cross_attention_dim:
        raise ValueError(
            "LTX prompt/generation config mismatch: audio context width "
            f"{audio_context_width} != runtime audio_cross_attention_dim "
            f"{runtime_config.audio_cross_attention_dim}"
        )
    if (
        prompt_context.caption_proj_before_connector
        != runtime_config.caption_proj_before_connector
    ):
        raise ValueError(
            "LTX prompt/generation config mismatch for caption_proj_before_connector"
        )
    if prompt_context.rope_type != runtime_config.rope_type:
        raise ValueError("LTX prompt/generation config mismatch for rope_type")
    if prompt_context.double_precision_rope != runtime_config.double_precision_rope:
        raise ValueError(
            "LTX prompt/generation config mismatch for double_precision_rope"
        )
    if (
        prompt_context.transformer_apply_gated_attention is not None
        and prompt_context.transformer_apply_gated_attention
        != runtime_config.apply_gated_attention
    ):
        raise ValueError(
            "LTX prompt/generation config mismatch for apply_gated_attention"
        )
    if (
        prompt_context.transformer_cross_attention_adaln is not None
        and prompt_context.transformer_cross_attention_adaln
        != runtime_config.cross_attention_adaln
    ):
        raise ValueError(
            "LTX prompt/generation config mismatch for cross_attention_adaln"
        )
    negative_video_context_width = _optional_context_width(
        prompt_context.negative_video_context
    )
    negative_audio_context_width = _optional_context_width(
        prompt_context.negative_audio_context
    )
    if prompt_context.negative_prompt_text is None:
        if (
            negative_video_context_width is not None
            or negative_audio_context_width is not None
        ):
            raise ValueError(
                "LTX prompt contract mismatch: negative prompt contexts were populated "
                "without negative_prompt_text"
            )
        return
    if negative_video_context_width is None:
        raise ValueError(
            "LTX guided generation requires negative_video_context when negative_prompt_text is present"
        )
    if negative_audio_context_width is None:
        raise ValueError(
            "LTX guided generation requires negative_audio_context when negative_prompt_text is present"
        )
    if negative_video_context_width != runtime_config.cross_attention_dim:
        raise ValueError(
            "LTX negative prompt/generation config mismatch: negative video context width "
            f"{negative_video_context_width} != runtime cross_attention_dim "
            f"{runtime_config.cross_attention_dim}"
        )
    if negative_audio_context_width != runtime_config.audio_cross_attention_dim:
        raise ValueError(
            "LTX negative prompt/generation config mismatch: negative audio context width "
            f"{negative_audio_context_width} != runtime audio_cross_attention_dim "
            f"{runtime_config.audio_cross_attention_dim}"
        )


def _denoise_distilled_audio_video(
    *,
    imports: _ReferenceImports,
    transformer: _AudioVideoTransformer,
    latents: MLXArray,
    positions: MLXArray,
    text_embeddings: MLXArray,
    audio_latents: MLXArray,
    audio_positions: MLXArray,
    audio_embeddings: MLXArray,
    negative_text_embeddings: MLXArray | None,
    negative_audio_embeddings: MLXArray | None,
    sigmas: tuple[float, ...],
    state: _LatentStateLike | None,
    runtime_config: _RuntimeModelConfig,
    freeze_audio: bool = False,
    trace_recorder: TraceRecorder | None = None,
    trace_sync: bool = False,
) -> tuple[MLXArray, MLXArray]:
    latents_dtype = latents.dtype
    cfg_enabled = (
        negative_text_embeddings is not None and negative_audio_embeddings is not None
    )
    batch_size, channels, frames, latent_h, latent_w = latents.shape
    num_tokens = int(frames * latent_h * latent_w)
    audio_batch, audio_channels, audio_frames, audio_bins = audio_latents.shape
    precomputed_rope = imports.precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=runtime_config.double_precision_rope,
    )
    precomputed_audio_rope = imports.precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=runtime_config.double_precision_rope,
    )
    if state is not None:
        denoise_mask = mx.reshape(state.denoise_mask, (batch_size, 1, frames, 1, 1))
        denoise_mask = mx.broadcast_to(
            denoise_mask, (batch_size, 1, frames, latent_h, latent_w)
        )
        video_timesteps_mask = mx.reshape(
            denoise_mask, (batch_size, num_tokens)
        ).astype(latents_dtype)
    else:
        video_timesteps_mask = mx.ones((batch_size, num_tokens), dtype=latents_dtype)
    audio_timesteps_mask = (
        mx.zeros((audio_batch, audio_frames), dtype=latents_dtype)
        if freeze_audio
        else mx.ones((audio_batch, audio_frames), dtype=latents_dtype)
    )
    total_steps = max(len(sigmas) - 1, 0)
    for step_index, (sigma_value, sigma_next_value) in enumerate(
        zip(sigmas[:-1], sigmas[1:]),
        start=1,
    ):
        step_context = (
            trace_recorder.span(
                "ltx.denoise.step",
                attributes={
                    "step_index": step_index,
                    "total_steps": total_steps,
                    "sigma": round(float(sigma_value), 6),
                    "sigma_next": round(float(sigma_next_value), 6),
                    "audio_frames": audio_frames,
                    "cfg_enabled": cfg_enabled,
                    "freeze_audio": freeze_audio,
                },
                snapshot=mlx_memory_snapshot,
            )
            if trace_recorder is not None
            else nullcontext()
        )
        with step_context:
            velocity: MLXArray | None = None
            audio_velocity: MLXArray | None = None
            negative_velocity: MLXArray | None = None
            negative_audio_velocity: MLXArray | None = None
            if _debug_progress_enabled():
                print(
                    "[ltx] denoise "
                    f"step={step_index}/{total_steps} "
                    f"sigma={float(sigma_value):.6f} "
                    f"audio_frames={audio_frames}",
                    flush=True,
                )
            sigma = mx.array(float(sigma_value), dtype=latents_dtype)
            sigma_next = mx.array(float(sigma_next_value), dtype=latents_dtype)
            flat_latents = mx.transpose(
                mx.reshape(latents, (batch_size, channels, -1)), (0, 2, 1)
            )
            modality = _PatchedModality(
                latent=flat_latents,
                sigma=mx.full((batch_size,), float(sigma_value), dtype=latents_dtype),
                timesteps=sigma * video_timesteps_mask,
                positions=positions,
                context=text_embeddings,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_rope,
            )
            audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
            audio_flat = mx.reshape(
                audio_flat, (audio_batch, audio_frames, audio_channels * audio_bins)
            )
            audio_modality = _PatchedModality(
                latent=audio_flat,
                sigma=mx.full((audio_batch,), float(sigma_value), dtype=latents_dtype),
                timesteps=sigma * audio_timesteps_mask,
                positions=audio_positions,
                context=audio_embeddings,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
            )
            conditioned_context = (
                trace_recorder.span(
                    "ltx.denoise.forward.conditioned",
                    attributes={"step_index": step_index},
                    sync=(lambda: _sync_optional_arrays(velocity, audio_velocity))
                    if trace_sync
                    else None,
                )
                if trace_recorder is not None
                else nullcontext()
            )
            with conditioned_context:
                velocity, audio_velocity = transformer(
                    video=modality, audio=audio_modality
                )
            if velocity is None or audio_velocity is None:
                raise RuntimeError(
                    "LTX transformer returned empty video/audio velocities for an enabled AV step"
                )
            velocity = mx.reshape(
                mx.transpose(velocity, (0, 2, 1)),
                (batch_size, channels, frames, latent_h, latent_w),
            )
            denoised = imports.to_denoised(latents, velocity, sigma)
            audio_velocity = mx.reshape(
                audio_velocity, (audio_batch, audio_frames, audio_channels, audio_bins)
            )
            audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
            audio_denoised = (
                audio_latents
                if freeze_audio
                else imports.to_denoised(audio_latents, audio_velocity, sigma)
            )
            negative_denoised: MLXArray | None = None
            negative_audio_denoised: MLXArray | None = None
            if (
                negative_text_embeddings is not None
                and negative_audio_embeddings is not None
            ):
                negative_modality = _PatchedModality(
                    latent=flat_latents,
                    sigma=mx.full(
                        (batch_size,), float(sigma_value), dtype=latents_dtype
                    ),
                    timesteps=sigma * video_timesteps_mask,
                    positions=positions,
                    context=negative_text_embeddings,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_rope,
                )
                negative_audio_modality = _PatchedModality(
                    latent=audio_flat,
                    sigma=mx.full(
                        (audio_batch,), float(sigma_value), dtype=latents_dtype
                    ),
                    timesteps=sigma * audio_timesteps_mask,
                    positions=audio_positions,
                    context=negative_audio_embeddings,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_audio_rope,
                )
                negative_context = (
                    trace_recorder.span(
                        "ltx.denoise.forward.negative",
                        attributes={"step_index": step_index},
                        sync=(
                            lambda: _sync_optional_arrays(
                                negative_velocity,
                                negative_audio_velocity,
                            )
                        )
                        if trace_sync
                        else None,
                    )
                    if trace_recorder is not None
                    else nullcontext()
                )
                with negative_context:
                    negative_velocity, negative_audio_velocity = transformer(
                        video=negative_modality,
                        audio=negative_audio_modality,
                    )
                if negative_velocity is None or negative_audio_velocity is None:
                    raise RuntimeError(
                        "LTX transformer returned empty negative video/audio velocities for an enabled AV step"
                    )
                negative_velocity = mx.reshape(
                    mx.transpose(negative_velocity, (0, 2, 1)),
                    (batch_size, channels, frames, latent_h, latent_w),
                )
                negative_denoised = imports.to_denoised(
                    latents, negative_velocity, sigma
                )
                if not freeze_audio:
                    negative_audio_velocity = mx.reshape(
                        negative_audio_velocity,
                        (audio_batch, audio_frames, audio_channels, audio_bins),
                    )
                    negative_audio_velocity = mx.transpose(
                        negative_audio_velocity, (0, 2, 1, 3)
                    )
                    negative_audio_denoised = imports.to_denoised(
                        audio_latents,
                        negative_audio_velocity,
                        sigma,
                    )
            denoised = _guided_prediction(
                denoised,
                negative_denoised,
                scale=_VIDEO_CFG_SCALE if cfg_enabled else 1.0,
                rescale_scale=_GUIDANCE_RESCALE_SCALE if cfg_enabled else 0.0,
            )
            if not freeze_audio:
                audio_denoised = _guided_prediction(
                    audio_denoised,
                    negative_audio_denoised,
                    scale=_AUDIO_CFG_SCALE if cfg_enabled else 1.0,
                    rescale_scale=_GUIDANCE_RESCALE_SCALE if cfg_enabled else 0.0,
                )
            if state is not None:
                denoised = imports.apply_denoise_mask(
                    denoised, state.clean_latent, state.denoise_mask
                )
            if float(sigma_next_value) == 0.0:
                latents = denoised.astype(latents_dtype)
                audio_latents = audio_denoised.astype(latents_dtype)
            else:
                latents = (
                    denoised.astype(mx.float32)
                    + sigma_next.astype(mx.float32)
                    * (latents.astype(mx.float32) - denoised.astype(mx.float32))
                    / sigma.astype(mx.float32)
                ).astype(latents_dtype)
                if freeze_audio:
                    audio_latents = audio_denoised.astype(latents_dtype)
                else:
                    audio_latents = (
                        audio_denoised.astype(mx.float32)
                        + sigma_next.astype(mx.float32)
                        * (
                            audio_latents.astype(mx.float32)
                            - audio_denoised.astype(mx.float32)
                        )
                        / sigma.astype(mx.float32)
                    ).astype(latents_dtype)
            mx.eval(latents, audio_latents)
    return latents, audio_latents

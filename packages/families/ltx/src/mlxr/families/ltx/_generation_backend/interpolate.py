from __future__ import annotations

from ..generation import AudioConditioningInput, ConditioningInput, GeneratedVideo
from ..prompt_encoding import PromptEncodingResult
from .two_stage import _TwoStageHost, generate_two_stage


def generate_interpolation(
    host: _TwoStageHost,
    *,
    prompt_context: PromptEncodingResult,
    conditioning_inputs: tuple[ConditioningInput, ...],
    audio_conditioning: AudioConditioningInput | None,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int | None,
    num_inference_steps: int | None,
    guidance_scale: float | None,
) -> GeneratedVideo:
    if len(conditioning_inputs) < 2:
        raise ValueError(
            "LTX video.interpolate requires at least two keyed image references"
        )

    generated = generate_two_stage(
        host,
        prompt_context=prompt_context,
        conditioning_inputs=conditioning_inputs,
        audio_conditioning=audio_conditioning,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        replace_first_frame_latent=False,
    )
    generated.backend = "mlxr_ltx_keyframe_interpolation"
    generated.metadata["pipeline_kind"] = "keyframe_interpolation"
    generated.metadata["backend"] = "mlxr_ltx_keyframe_interpolation"
    generated.metadata["conditioning_mode"] = "guiding_keyframes"
    generated.metadata["interpolation_keyframe_count"] = len(conditioning_inputs)
    return generated

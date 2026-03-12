from __future__ import annotations

import math

import mlx.core as mx

from .config import AutoencoderConfig, SchedulerConfig


def calculate_shift(image_seq_len: int, config: SchedulerConfig) -> float:
    if image_seq_len <= config.base_image_seq_len:
        return config.base_shift
    if image_seq_len >= config.max_image_seq_len:
        return config.max_shift
    slope = (config.max_shift - config.base_shift) / (
        config.max_image_seq_len - config.base_image_seq_len
    )
    intercept = config.base_shift - slope * config.base_image_seq_len
    return float(image_seq_len * slope + intercept)


def pack_latents(latents: mx.array) -> mx.array:
    batch_size, frames, channels, height, width = (int(size) for size in latents.shape)
    if frames != 1:
        raise ValueError(
            "Qwen-Image prompt-only generation currently expects one latent frame"
        )
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Qwen-Image latent height and width must be divisible by 2")
    packed = latents.reshape(
        batch_size,
        frames,
        channels,
        height // 2,
        2,
        width // 2,
        2,
    )
    packed = packed.transpose(0, 3, 5, 2, 4, 6, 1)
    return packed.reshape(batch_size, (height // 2) * (width // 2), channels * 4)


def unpack_latents(
    packed: mx.array,
    *,
    height: int,
    width: int,
    vae_scale_factor: int,
) -> mx.array:
    batch_size, _, channels = (int(size) for size in packed.shape)
    adjusted_height = 2 * (int(height) // (vae_scale_factor * 2))
    adjusted_width = 2 * (int(width) // (vae_scale_factor * 2))
    unpacked = packed.reshape(
        batch_size,
        adjusted_height // 2,
        adjusted_width // 2,
        channels // 4,
        2,
        2,
    )
    unpacked = unpacked.transpose(0, 3, 1, 4, 2, 5)
    return unpacked.reshape(
        batch_size,
        channels // 4,
        1,
        adjusted_height,
        adjusted_width,
    )


def normalize_latents(latents: mx.array, config: AutoencoderConfig) -> mx.array:
    mean = mx.array(config.latents_mean, dtype=latents.dtype).reshape(1, -1, 1, 1, 1)
    std = mx.array(config.latents_std, dtype=latents.dtype).reshape(1, -1, 1, 1, 1)
    return (latents - mean) / std


def denormalize_latents(latents: mx.array, config: AutoencoderConfig) -> mx.array:
    mean = mx.array(config.latents_mean, dtype=latents.dtype).reshape(1, -1, 1, 1, 1)
    std = mx.array(config.latents_std, dtype=latents.dtype).reshape(1, -1, 1, 1, 1)
    return latents * std + mean


def shifted_sigmas(
    *,
    num_inference_steps: int,
    image_seq_len: int,
    config: SchedulerConfig,
) -> list[float]:
    sigmas = mx.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    if config.use_dynamic_shifting:
        mu = calculate_shift(image_seq_len, config)
        if config.time_shift_type != "exponential":
            raise ValueError(
                "Owned Qwen-Image scheduler currently supports exponential shifting only"
            )
        sigmas = _time_shift_exponential(mu, sigmas)
    if config.shift_terminal is not None:
        sigmas = _stretch_shift_to_terminal(sigmas, config.shift_terminal)
    values = [float(value.item()) for value in sigmas]
    if config.invert_sigmas:
        values = [1.0 - value for value in values]
    values.append(1.0 if config.invert_sigmas else 0.0)
    return values


def _time_shift_exponential(mu: float, t: mx.array) -> mx.array:
    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + (1.0 / t - 1.0))


def _stretch_shift_to_terminal(t: mx.array, shift_terminal: float) -> mx.array:
    one_minus_z = 1.0 - t
    scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
    return 1.0 - (one_minus_z / scale_factor)

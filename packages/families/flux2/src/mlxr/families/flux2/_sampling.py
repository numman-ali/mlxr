from __future__ import annotations

import math

import mlx.core as mx

_BASE_SHIFT = 0.5
_MAX_SHIFT = 1.15


def prepare_latent_images(latents: mx.array) -> tuple[mx.array, mx.array]:
    batch_size, height, width, channels = (int(size) for size in latents.shape)
    flattened = latents.reshape(batch_size, height * width, channels)
    time_ids = mx.zeros((height, width), dtype=mx.int32)
    row_ids, col_ids = mx.meshgrid(
        mx.arange(height, dtype=mx.int32),
        mx.arange(width, dtype=mx.int32),
        indexing="ij",
    )
    layer_ids = mx.zeros((height, width), dtype=mx.int32)
    image_ids = mx.stack([time_ids, row_ids, col_ids, layer_ids], axis=-1)
    image_ids = mx.repeat(
        image_ids.reshape(1, height * width, 4),
        batch_size,
        axis=0,
    )
    return flattened, image_ids


def unpack_latent_images(
    flattened: mx.array,
    *,
    latent_height: int,
    latent_width: int,
    channels: int,
) -> mx.array:
    batch_size = int(flattened.shape[0])
    return flattened.reshape(batch_size, latent_height, latent_width, channels)


def prepare_text_ids(batch_size: int, sequence_length: int) -> mx.array:
    text_ids = mx.zeros((sequence_length, 4), dtype=mx.int32)
    text_ids[:, 3] = mx.arange(sequence_length, dtype=mx.int32)
    return mx.repeat(text_ids[None, :, :], batch_size, axis=0)


def scalar_schedule(
    *,
    num_steps: int,
    image_sequence_length: int,
    start: float = 1.0,
    stop: float = 0.0,
) -> list[float]:
    timesteps = mx.linspace(start, stop, num_steps + 1)
    shifted = _time_shift(
        image_sequence_length=image_sequence_length,
        num_steps=num_steps,
        timesteps=timesteps,
    )
    return [float(value.item()) for value in shifted]


def _time_shift(
    *, image_sequence_length: int, num_steps: int, timesteps: mx.array
) -> mx.array:
    mu = compute_empirical_mu(image_sequence_length, num_steps)
    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + (1.0 / timesteps - 1.0))


def compute_empirical_mu(image_sequence_length: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_sequence_length > 4_300:
        return float(a2 * image_sequence_length + b2)

    mu_200 = a2 * image_sequence_length + b2
    mu_10 = a1 * image_sequence_length + b1
    slope = (mu_200 - mu_10) / 190.0
    intercept = mu_200 - 200.0 * slope
    return float(slope * num_steps + intercept)

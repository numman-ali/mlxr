from __future__ import annotations

import mlx.core as mx

from .config import SchedulerConfig
from .sampling import shifted_sigmas


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.timesteps: list[float] = []
        self.sigmas: list[float] = []

    def set_timesteps(
        self,
        *,
        num_inference_steps: int,
        image_sequence_length: int,
    ) -> None:
        self.sigmas = shifted_sigmas(
            num_inference_steps=num_inference_steps,
            image_seq_len=image_sequence_length,
            config=self.config,
        )
        self.timesteps = [
            sigma * self.config.num_train_timesteps for sigma in self.sigmas[:-1]
        ]

    def step(
        self,
        *,
        sample: mx.array,
        model_output: mx.array,
        current_sigma: float,
        next_sigma: float,
    ) -> mx.array:
        sample_f32 = sample.astype(mx.float32)
        model_output_f32 = model_output.astype(mx.float32)
        prev_sample = sample_f32 + (next_sigma - current_sigma) * model_output_f32
        return prev_sample.astype(model_output.dtype)

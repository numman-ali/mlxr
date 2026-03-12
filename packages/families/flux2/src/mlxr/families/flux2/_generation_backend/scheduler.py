from __future__ import annotations

import mlx.core as mx

from .._sampling import scalar_schedule
from .config import SchedulerConfig


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config

    def timesteps(
        self,
        *,
        num_inference_steps: int,
        image_sequence_length: int,
    ) -> list[float]:
        return scalar_schedule(
            num_steps=num_inference_steps,
            image_sequence_length=image_sequence_length,
        )

    def step(
        self,
        *,
        sample: mx.array,
        model_output: mx.array,
        timestep: float,
        next_timestep: float,
    ) -> mx.array:
        return sample + (next_timestep - timestep) * model_output

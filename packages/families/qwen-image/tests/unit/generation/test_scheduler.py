from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlxr.families.qwen_image._generation_backend.config import SchedulerConfig
from mlxr.families.qwen_image._generation_backend.scheduler import (
    FlowMatchEulerDiscreteScheduler,
)


class QwenImageGenerationSchedulerTests(unittest.TestCase):
    def test_set_timesteps_builds_descending_schedule(self) -> None:
        scheduler = FlowMatchEulerDiscreteScheduler(
            SchedulerConfig(
                base_image_seq_len=256,
                base_shift=0.5,
                invert_sigmas=False,
                max_image_seq_len=8192,
                max_shift=0.9,
                num_train_timesteps=1000,
                shift_terminal=0.02,
                time_shift_type="exponential",
                use_dynamic_shifting=True,
            )
        )

        scheduler.set_timesteps(num_inference_steps=4, image_sequence_length=4096)

        self.assertEqual(len(scheduler.timesteps), 4)
        self.assertEqual(len(scheduler.sigmas), 5)
        self.assertGreater(scheduler.timesteps[0], scheduler.timesteps[-1])

    def test_step_uses_sigma_space_euler_update(self) -> None:
        scheduler = FlowMatchEulerDiscreteScheduler(
            SchedulerConfig(
                base_image_seq_len=256,
                base_shift=0.5,
                invert_sigmas=False,
                max_image_seq_len=8192,
                max_shift=0.9,
                num_train_timesteps=1000,
                shift_terminal=0.02,
                time_shift_type="exponential",
                use_dynamic_shifting=True,
            )
        )
        sample = mx.zeros((1, 16, 8), dtype=mx.float32)
        model_output = mx.ones_like(sample)

        updated = scheduler.step(
            sample=sample,
            model_output=model_output,
            current_sigma=1.0,
            next_sigma=0.5,
        )

        self.assertEqual(updated.shape, sample.shape)
        self.assertTrue(
            np.allclose(
                np.asarray(updated), np.full(sample.shape, -0.5, dtype=np.float32)
            )
        )

    def test_step_upcasts_sample_math_and_returns_model_output_dtype(self) -> None:
        scheduler = FlowMatchEulerDiscreteScheduler(
            SchedulerConfig(
                base_image_seq_len=256,
                base_shift=0.5,
                invert_sigmas=False,
                max_image_seq_len=8192,
                max_shift=0.9,
                num_train_timesteps=1000,
                shift_terminal=0.02,
                time_shift_type="exponential",
                use_dynamic_shifting=True,
            )
        )
        sample = mx.array([[[1000.0]]], dtype=mx.bfloat16)
        model_output = mx.array([[[0.125]]], dtype=mx.float32)

        updated = scheduler.step(
            sample=sample,
            model_output=model_output,
            current_sigma=1.0,
            next_sigma=0.5,
        )

        self.assertEqual(updated.dtype, mx.float32)
        self.assertTrue(
            np.allclose(np.asarray(updated), np.array([[[999.9375]]], dtype=np.float32))
        )


if __name__ == "__main__":
    unittest.main()

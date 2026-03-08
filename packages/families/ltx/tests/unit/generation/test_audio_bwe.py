from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file


class LTXAudioBWETests(unittest.TestCase):
    def test_ltx_generation_backend_reads_nested_bwe_vocoder_contract(self) -> None:
        from mlxr.families.ltx import _generation_backend as backend

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ltx-checkpoint.safetensors"
            save_file(
                {"dummy": np.zeros((1,), dtype=np.float32)},
                str(checkpoint_path),
                metadata={
                    "config": json.dumps(
                        {
                            "vocoder": {
                                "vocoder": {
                                    "upsample_initial_channel": 1536,
                                    "resblock": "AMP1",
                                    "upsample_rates": [5, 2, 2, 2, 2, 2],
                                    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
                                    "resblock_kernel_sizes": [3, 7, 11],
                                    "resblock_dilation_sizes": [
                                        [1, 3, 5],
                                        [1, 3, 5],
                                        [1, 3, 5],
                                    ],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                },
                                "bwe": {
                                    "upsample_initial_channel": 512,
                                    "resblock": "AMP1",
                                    "upsample_rates": [6, 5, 2, 2, 2],
                                    "upsample_kernel_sizes": [12, 11, 4, 4, 4],
                                    "resblock_kernel_sizes": [3, 7, 11],
                                    "resblock_dilation_sizes": [
                                        [1, 3, 5],
                                        [1, 3, 5],
                                        [1, 3, 5],
                                    ],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                    "apply_final_activation": False,
                                    "input_sampling_rate": 16000,
                                    "output_sampling_rate": 48000,
                                    "hop_length": 80,
                                    "n_fft": 512,
                                    "win_size": 512,
                                    "num_mels": 64,
                                },
                            }
                        }
                    )
                },
            )

            config = backend._runtime_vocoder_config(checkpoint_path)

        self.assertEqual(config.vocoder.resblock, "AMP1")
        self.assertEqual(config.vocoder.activation, "snakebeta")
        self.assertFalse(config.vocoder.use_tanh_at_final)
        self.assertTrue(config.vocoder.apply_final_activation)
        self.assertFalse(config.vocoder.use_bias_at_final)
        self.assertEqual(config.vocoder.upsample_initial_channel, 1536)
        self.assertEqual(config.output_sample_rate, 16000)
        self.assertTrue(config.uses_bwe)
        self.assertEqual(config.bwe_output_sample_rate, 48000)
        self.assertIsNotNone(config.bwe)
        assert config.bwe is not None
        self.assertEqual(config.bwe.input_sample_rate, 16000)
        self.assertEqual(config.bwe.output_sample_rate, 48000)
        self.assertEqual(config.bwe.hop_length, 80)
        self.assertEqual(config.bwe.n_fft, 512)
        self.assertEqual(config.bwe.win_size, 512)
        self.assertEqual(config.bwe.num_mels, 64)

    def test_ltx_generation_backend_uses_bwe_vocoder_when_checkpoint_has_full_contract(
        self,
    ) -> None:
        from mlxr.families.ltx._generation_backend.video_stack import (
            _load_runtime_vocoder,
        )

        def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
            sanitized: dict[str, mx.array] = {}
            for key, value in weights.items():
                if key.endswith(".weight") and value.ndim == 3:
                    if key.startswith("ups."):
                        value = mx.transpose(value, (1, 2, 0))
                    else:
                        value = mx.transpose(value, (0, 2, 1))
                sanitized[key] = value
            return sanitized

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ltx-checkpoint.safetensors"
            save_file(
                {
                    "vocoder.vocoder.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.vocoder.conv_post.weight": np.zeros(
                        (2, 2, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_post.weight": np.zeros(
                        (2, 2, 7), dtype=np.float32
                    ),
                    "vocoder.mel_stft.mel_basis": np.zeros((64, 257), dtype=np.float32),
                    "vocoder.mel_stft.stft_fn.forward_basis": np.zeros(
                        (514, 1, 512), dtype=np.float32
                    ),
                    "vocoder.mel_stft.stft_fn.inverse_basis": np.zeros(
                        (514, 1, 512), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={
                    "config": json.dumps(
                        {
                            "vocoder": {
                                "vocoder": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                },
                                "bwe": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                    "apply_final_activation": False,
                                    "input_sampling_rate": 16000,
                                    "output_sampling_rate": 48000,
                                    "hop_length": 80,
                                    "n_fft": 512,
                                    "win_size": 512,
                                    "num_mels": 64,
                                },
                            }
                        }
                    )
                },
            )
            checkpoint_weights = mx.load(str(checkpoint_path))
            if not isinstance(checkpoint_weights, dict):
                self.fail("Expected checkpoint weights to load into a mapping")

            vocoder, sample_rate, backend_name = _load_runtime_vocoder(
                checkpoint_path=checkpoint_path,
                checkpoint_weights=checkpoint_weights,
                sanitize_vocoder_weights=sanitize,
            )

        self.assertEqual(sample_rate, 48000)
        self.assertEqual(backend_name, "mlxr_vocoder_with_bwe")
        self.assertTrue(callable(vocoder))
        self.assertTrue(hasattr(vocoder, "parameters"))

    def test_ltx_generation_backend_bwe_loader_fails_closed_on_missing_stft_buffers(
        self,
    ) -> None:
        from mlxr.families.ltx._generation_backend.video_stack import (
            _load_runtime_vocoder,
        )

        def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
            sanitized: dict[str, mx.array] = {}
            for key, value in weights.items():
                if key.endswith(".weight") and value.ndim == 3:
                    if key.startswith("ups."):
                        value = mx.transpose(value, (1, 2, 0))
                    else:
                        value = mx.transpose(value, (0, 2, 1))
                sanitized[key] = value
            return sanitized

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ltx-checkpoint.safetensors"
            save_file(
                {
                    "vocoder.vocoder.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={
                    "config": json.dumps(
                        {
                            "vocoder": {
                                "vocoder": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                },
                                "bwe": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "input_sampling_rate": 16000,
                                    "output_sampling_rate": 48000,
                                    "hop_length": 80,
                                    "n_fft": 512,
                                    "win_size": 512,
                                    "num_mels": 64,
                                },
                            }
                        }
                    )
                },
            )
            checkpoint_weights = mx.load(str(checkpoint_path))
            if not isinstance(checkpoint_weights, dict):
                self.fail("Expected checkpoint weights to load into a mapping")

            with self.assertRaisesRegex(RuntimeError, "missing mel STFT buffers"):
                _load_runtime_vocoder(
                    checkpoint_path=checkpoint_path,
                    checkpoint_weights=checkpoint_weights,
                    sanitize_vocoder_weights=sanitize,
                )

    def test_ltx_generation_backend_bwe_loader_fails_closed_on_wrong_stft_shapes(
        self,
    ) -> None:
        from mlxr.families.ltx._generation_backend.video_stack import (
            _load_runtime_vocoder,
        )

        def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
            sanitized: dict[str, mx.array] = {}
            for key, value in weights.items():
                if key.endswith(".weight") and value.ndim == 3:
                    if key.startswith("ups."):
                        value = mx.transpose(value, (1, 2, 0))
                    else:
                        value = mx.transpose(value, (0, 2, 1))
                sanitized[key] = value
            return sanitized

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ltx-checkpoint.safetensors"
            save_file(
                {
                    "vocoder.vocoder.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.vocoder.conv_post.weight": np.zeros(
                        (2, 2, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_post.weight": np.zeros(
                        (2, 2, 7), dtype=np.float32
                    ),
                    "vocoder.mel_stft.mel_basis": np.zeros((63, 257), dtype=np.float32),
                    "vocoder.mel_stft.stft_fn.forward_basis": np.zeros(
                        (514, 1, 512), dtype=np.float32
                    ),
                    "vocoder.mel_stft.stft_fn.inverse_basis": np.zeros(
                        (514, 1, 512), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={
                    "config": json.dumps(
                        {
                            "vocoder": {
                                "vocoder": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                },
                                "bwe": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                    "apply_final_activation": False,
                                    "input_sampling_rate": 16000,
                                    "output_sampling_rate": 48000,
                                    "hop_length": 80,
                                    "n_fft": 512,
                                    "win_size": 512,
                                    "num_mels": 64,
                                },
                            }
                        }
                    )
                },
            )
            checkpoint_weights = mx.load(str(checkpoint_path))
            if not isinstance(checkpoint_weights, dict):
                self.fail("Expected checkpoint weights to load into a mapping")

            with self.assertRaisesRegex(RuntimeError, "invalid mel basis shape"):
                _load_runtime_vocoder(
                    checkpoint_path=checkpoint_path,
                    checkpoint_weights=checkpoint_weights,
                    sanitize_vocoder_weights=sanitize,
                )

    def test_ltx_audio_bwe_wrapper_uses_batch_channel_time_waveform_contract(
        self,
    ) -> None:
        from mlxr.families.ltx._audio_bwe import AudioVocoderWithBWE

        class FakeBaseVocoder:
            def __call__(self, mel_spec: mx.array) -> mx.array:
                del mel_spec
                return mx.zeros((1, 2, 5), dtype=mx.float32)

        class FakeMelSTFT:
            def mel_spectrogram(
                self, waveform: mx.array
            ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
                batch_channels = int(waveform.shape[0])
                log_mel = mx.zeros((batch_channels, 64, 2), dtype=mx.float32)
                magnitude = mx.zeros((batch_channels, 257, 2), dtype=mx.float32)
                phase = mx.zeros((batch_channels, 257, 2), dtype=mx.float32)
                energy = mx.zeros((batch_channels, 2), dtype=mx.float32)
                return log_mel, magnitude, phase, energy

        class FakeBWEGenerator:
            def __init__(self) -> None:
                self.last_input_shape: tuple[int, ...] | None = None

            def __call__(self, mel_spec: mx.array) -> mx.array:
                self.last_input_shape = tuple(int(v) for v in mel_spec.shape)
                return mx.zeros((1, 2, 15), dtype=mx.float32)

        class FakeResampler:
            def __call__(self, waveform: mx.array) -> mx.array:
                self.last_waveform_shape = tuple(int(v) for v in waveform.shape)
                return mx.zeros((1, 2, 15), dtype=mx.float32)

        bwe_generator = FakeBWEGenerator()
        wrapper = AudioVocoderWithBWE(
            vocoder=FakeBaseVocoder(),
            bwe_generator=bwe_generator,
            mel_stft=FakeMelSTFT(),
            input_sample_rate=16000,
            output_sample_rate=48000,
            hop_length=80,
        )
        wrapper.resampler = FakeResampler()

        output = wrapper(mx.zeros((1, 2, 2, 64), dtype=mx.float32))

        self.assertEqual(output.shape, (1, 2, 15))
        self.assertEqual(bwe_generator.last_input_shape, (1, 2, 2, 64))

    def test_ltx_audio_bwe_hann_filter_has_expected_shape(self) -> None:
        from mlxr.families.ltx._audio_bwe import _hann_sinc_filter1d

        filter_ = _hann_sinc_filter1d(ratio=3)

        self.assertEqual(filter_.shape[0], 1)
        self.assertEqual(filter_.shape[2], 1)
        self.assertEqual(filter_.dtype, mx.float32)

    def test_ltx_audio_bwe_resampler_upsamples_batch_channel_time_waveform(
        self,
    ) -> None:
        from mlxr.families.ltx._audio_bwe import _WaveformResampler

        resampler = _WaveformResampler(
            input_sample_rate=16000,
            output_sample_rate=48000,
        )

        waveform = mx.ones((1, 2, 5), dtype=mx.float32)
        upsampled = resampler(waveform)

        self.assertEqual(upsampled.shape, (1, 2, 15))

    def test_ltx_audio_mel_stft_returns_expected_shapes(self) -> None:
        from mlxr.families.ltx._audio_bwe import AudioMelSTFT

        mel_stft = AudioMelSTFT(
            filter_length=4,
            hop_length=2,
            win_length=4,
            n_mel_channels=2,
        )
        mel_stft.stft_fn.forward_basis = mx.ones((6, 4, 1), dtype=mx.float32)
        mel_stft.mel_basis = mx.ones((2, 3), dtype=mx.float32)

        log_mel, magnitude, phase, energy = mel_stft.mel_spectrogram(
            mx.ones((1, 6), dtype=mx.float32)
        )

        self.assertEqual(log_mel.shape, (1, 2, 3))
        self.assertEqual(magnitude.shape, (1, 3, 3))
        self.assertEqual(phase.shape, (1, 3, 3))
        self.assertEqual(energy.shape, (1, 3))

    def test_ltx_audio_mel_stft_rejects_wrong_rank(self) -> None:
        from mlxr.families.ltx._audio_bwe import AudioMelSTFT

        mel_stft = AudioMelSTFT(
            filter_length=4,
            hop_length=2,
            win_length=4,
            n_mel_channels=2,
        )

        with self.assertRaisesRegex(ValueError, "expects \\[batch, time\\]"):
            mel_stft.mel_spectrogram(mx.ones((1, 2, 3), dtype=mx.float32))

    def test_ltx_audio_vocoder_rejects_unsupported_activation(self) -> None:
        from mlxr.families.ltx._audio_vocoder import AudioVocoder

        with self.assertRaisesRegex(ValueError, "Unsupported LTX vocoder activation"):
            AudioVocoder(
                resblock="AMP1",
                activation="gelu",
            )

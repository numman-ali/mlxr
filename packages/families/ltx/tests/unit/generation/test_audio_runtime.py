from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlxr.families.ltx._generation_backend.runtime_helpers import (
    _normalize_audio_mel_layout,
)
from safetensors.numpy import save_file


class LTXAudioRuntimeTests(unittest.TestCase):
    def test_ltx_generation_runtime_config_infers_current_22b_flags(self) -> None:
        from mlxr.families.ltx import _generation_backend as backend

        checkpoint_metadata = {
            "transformer": {
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "rope_type": "split",
                "frequencies_precision": "float64",
                "caption_proj_before_connector": True,
            },
            "vae": {
                "latent_channels": 128,
                "out_channels": 3,
                "patch_size": 4,
                "decoder_base_channels": 128,
                "decoder_blocks": [
                    ["res_x", {"num_layers": 4}],
                    ["compress_all", {"multiplier": 2, "residual": False}],
                    ["res_x", {"num_layers": 3}],
                    ["compress_all", {"multiplier": 2, "residual": False}],
                    ["res_x", {"num_layers": 3}],
                    ["compress_all", {"multiplier": 2, "residual": False}],
                    ["res_x", {"num_layers": 4}],
                ],
                "norm_layer": "pixel_norm",
                "decoder_spatial_padding_mode": "reflect",
                "timestep_conditioning": True,
                "causal_decoder": False,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.safetensors"
            save_file(
                {
                    "transformer_blocks.0.attn1.to_gate_logits.weight": np.zeros(
                        (32, 4096), dtype=np.float32
                    ),
                    "transformer_blocks.0.prompt_scale_shift_table": np.zeros(
                        (2, 4096), dtype=np.float32
                    ),
                    "audio_patchify_proj.weight": np.zeros(
                        (2048, 128), dtype=np.float32
                    ),
                    "audio_proj_out.weight": np.zeros((128, 2048), dtype=np.float32),
                    "audio_prompt_adaln_single.linear.weight": np.zeros(
                        (4096, 2048), dtype=np.float32
                    ),
                    "transformer_blocks.0.audio_prompt_scale_shift_table": np.zeros(
                        (2, 2048), dtype=np.float32
                    ),
                    "transformer_blocks.0.audio_attn1.to_q.weight": np.zeros(
                        (2048, 2048), dtype=np.float32
                    ),
                    "transformer_blocks.0.audio_attn2.to_q.weight": np.zeros(
                        (2048, 2048), dtype=np.float32
                    ),
                    "transformer_blocks.0.audio_to_video_attn.to_q.weight": np.zeros(
                        (2048, 4096), dtype=np.float32
                    ),
                    "transformer_blocks.0.video_to_audio_attn.to_q.weight": np.zeros(
                        (2048, 2048), dtype=np.float32
                    ),
                    "vae.per_channel_statistics.mean-of-means": np.zeros(
                        (128,), dtype=np.float32
                    ),
                    "vae.per_channel_statistics.std-of-means": np.ones(
                        (128,), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={"config": json.dumps(checkpoint_metadata)},
            )

            runtime_config = backend._runtime_model_config(checkpoint_path)

        self.assertTrue(runtime_config.apply_gated_attention)
        self.assertTrue(runtime_config.cross_attention_adaln)
        self.assertTrue(runtime_config.caption_proj_before_connector)
        self.assertEqual(runtime_config.rope_type, "split")
        self.assertTrue(runtime_config.double_precision_rope)
        self.assertTrue(runtime_config.audio_enabled)
        self.assertEqual(runtime_config.audio_num_attention_heads, 32)
        self.assertEqual(runtime_config.audio_attention_head_dim, 64)
        self.assertEqual(runtime_config.audio_in_channels, 128)
        self.assertEqual(runtime_config.audio_out_channels, 128)
        self.assertEqual(runtime_config.audio_latent_mel_bins, 16)
        self.assertEqual(runtime_config.audio_cross_attention_dim, 2048)
        self.assertEqual(runtime_config.audio_positional_embedding_max_pos, [20])
        self.assertEqual(runtime_config.av_ca_timestep_scale_multiplier, 1000)

    def test_ltx_runtime_imports_derive_audio_latent_mel_bins_from_transformer_contract(
        self,
    ) -> None:
        from mlxr.families.ltx._generation_backend.distilled import (
            LTXDistilledVideoGenerator,
        )

        checkpoint_metadata = {
            "transformer": {
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "rope_type": "split",
                "frequencies_precision": "float64",
                "caption_proj_before_connector": True,
                "audio_num_attention_heads": 32,
                "audio_attention_head_dim": 64,
                "audio_in_channels": 128,
                "audio_out_channels": 128,
                "audio_cross_attention_dim": 2048,
                "audio_positional_embedding_max_pos": [20],
            },
            "vae": {
                "latent_channels": 128,
                "out_channels": 3,
                "patch_size": 4,
                "decoder_base_channels": 128,
                "decoder_blocks": [["res_x", {"num_layers": 4}]],
                "norm_layer": "pixel_norm",
                "decoder_spatial_padding_mode": "reflect",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.safetensors"
            upsampler_path = Path(tmp_dir) / "upsampler.safetensors"
            save_file(
                {
                    "model.diffusion_model.audio_patchify_proj.weight": np.zeros(
                        (2048, 128), dtype=np.float32
                    ),
                    "model.diffusion_model.audio_attn1.to_q.weight": np.zeros(
                        (2048, 2048), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={"config": json.dumps(checkpoint_metadata)},
            )
            upsampler_path.write_text("unused", encoding="utf-8")

            generator = LTXDistilledVideoGenerator(
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=upsampler_path,
            )
            imports = generator._imports()

        self.assertEqual(imports.audio_latent_channels, 8)
        self.assertEqual(imports.audio_mel_bins, 16)

    def test_ltx_runtime_model_config_rejects_inconsistent_audio_output_geometry(
        self,
    ) -> None:
        from mlxr.families.ltx._generation_backend.config import _runtime_model_config

        checkpoint_metadata = {
            "transformer": {
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "rope_type": "split",
                "frequencies_precision": "float64",
                "caption_proj_before_connector": True,
                "audio_num_attention_heads": 32,
                "audio_attention_head_dim": 64,
                "audio_in_channels": 128,
                "audio_out_channels": 96,
                "audio_cross_attention_dim": 2048,
                "audio_positional_embedding_max_pos": [20],
            },
            "vae": {
                "latent_channels": 128,
                "out_channels": 3,
                "patch_size": 4,
                "decoder_base_channels": 128,
                "decoder_blocks": [["res_x", {"num_layers": 4}]],
                "norm_layer": "pixel_norm",
                "decoder_spatial_padding_mode": "reflect",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.safetensors"
            save_file(
                {
                    "model.diffusion_model.audio_patchify_proj.weight": np.zeros(
                        (2048, 128), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={"config": json.dumps(checkpoint_metadata)},
            )

            with self.assertRaisesRegex(
                RuntimeError, "inconsistent audio latent geometry"
            ):
                _runtime_model_config(checkpoint_path)

    def test_ltx_audio_conditioning_normalizes_mel_layout_for_encoder(self) -> None:
        mel_bins_first = np.zeros((1, 64, 2, 65), dtype=np.float32)
        normalized = _normalize_audio_mel_layout(mel_bins_first, input_channels=2)
        self.assertEqual(normalized.shape, (1, 2, 65, 64))

        channels_first = np.zeros((1, 2, 65, 64), dtype=np.float32)
        preserved = _normalize_audio_mel_layout(channels_first, input_channels=2)
        self.assertEqual(preserved.shape, (1, 2, 65, 64))

    def test_ltx_audio_processor_waveform_to_mel_returns_canonical_layout(
        self,
    ) -> None:
        from mlxr.families.ltx._generation_backend.audio_processor import AudioProcessor

        processor = AudioProcessor(
            sample_rate=16000,
            mel_bins=64,
            mel_hop_length=160,
            n_fft=1024,
        )
        samples = np.linspace(0.0, 1.0, 16000, endpoint=False, dtype=np.float32)
        left = np.sin(2.0 * np.pi * 220.0 * samples).astype(np.float32)
        right = np.sin(2.0 * np.pi * 440.0 * samples).astype(np.float32)
        waveform = np.stack([left, right], axis=0)

        mel = processor.waveform_to_mel(waveform, sample_rate=16000)

        self.assertEqual(mel.ndim, 4)
        self.assertEqual(mel.shape[0], 1)
        self.assertEqual(mel.shape[1], 2)
        self.assertEqual(mel.shape[3], 64)
        self.assertEqual(mel.dtype, np.float32)

    def test_ltx_audio_processor_rejects_wrong_sample_rate(self) -> None:
        from mlxr.families.ltx._generation_backend.audio_processor import AudioProcessor

        processor = AudioProcessor(
            sample_rate=16000,
            mel_bins=64,
            mel_hop_length=160,
            n_fft=1024,
        )
        waveform = np.zeros((2, 16000), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "expected sample_rate=16000"):
            processor.waveform_to_mel(waveform, sample_rate=24000)

    def test_ltx_owned_decode_audio_composes_decoder_and_vocoder(self) -> None:
        from mlxr.families.ltx._generation_backend.audio_render import decode_audio
        from mlxr.families.ltx._generation_backend.types import MLXArray

        class _Stats:
            def __init__(self) -> None:
                self._mean_of_means: MLXArray = mx.zeros((1,), dtype=mx.float32)
                self._std_of_means: MLXArray = mx.ones((1,), dtype=mx.float32)

            def normalize(self, x: MLXArray) -> MLXArray:
                return x

        class _FakeDecoder:
            def __init__(self) -> None:
                self.per_channel_statistics = _Stats()

            def parameters(self) -> tuple[object, ...]:
                return ()

            def load_weights(
                self, weights: list[tuple[str, mx.array]], *, strict: bool = False
            ) -> None:
                del weights, strict

            def __call__(self, audio_latents: mx.array) -> mx.array:
                return audio_latents + 1.0

        class _FakeVocoder:
            def parameters(self) -> tuple[object, ...]:
                return ()

            def load_weights(
                self, weights: list[tuple[str, mx.array]], *, strict: bool = False
            ) -> None:
                del weights, strict

            def __call__(self, decoded_audio: mx.array) -> mx.array:
                batch = int(decoded_audio.shape[0])
                return mx.ones((batch, 2, 32), dtype=decoded_audio.dtype)

        latents = mx.zeros((1, 8, 4, 16), dtype=mx.float32)
        waveform = decode_audio(latents, _FakeDecoder(), _FakeVocoder())

        self.assertEqual(tuple(int(size) for size in waveform.shape), (2, 32))

    def test_ltx_audio_stack_passes_raw_audio_vae_weights_to_decoder_loader(
        self,
    ) -> None:
        from mlxr.families.ltx._generation_backend import runtime_helpers as helpers
        from mlxr.families.ltx._generation_backend.distilled import (
            LTXDistilledVideoGenerator,
        )

        class _FakeDecoder:
            def parameters(self) -> tuple[object, ...]:
                return ()

        class _FakeVocoder:
            def parameters(self) -> tuple[object, ...]:
                return ()

        captured: dict[str, object] = {}

        def _fake_load_audio_decoder(
            checkpoint_root: Path, *, unified_weights: dict[str, mx.array]
        ) -> _FakeDecoder:
            captured["checkpoint_root"] = checkpoint_root
            captured["weight_keys"] = tuple(sorted(unified_weights))
            return _FakeDecoder()

        checkpoint_metadata = {
            "transformer": {
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "rope_type": "split",
                "frequencies_precision": "float64",
                "caption_proj_before_connector": True,
                "audio_num_attention_heads": 32,
                "audio_attention_head_dim": 64,
                "audio_in_channels": 128,
                "audio_out_channels": 128,
                "audio_cross_attention_dim": 2048,
                "audio_positional_embedding_max_pos": [20],
            },
            "vae": {
                "latent_channels": 128,
                "out_channels": 3,
                "patch_size": 4,
                "decoder_base_channels": 128,
                "decoder_blocks": [["res_x", {"num_layers": 4}]],
                "norm_layer": "pixel_norm",
                "decoder_spatial_padding_mode": "reflect",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.safetensors"
            upsampler_path = Path(tmp_dir) / "upsampler.safetensors"
            save_file(
                {
                    "model.diffusion_model.audio_patchify_proj.weight": np.zeros(
                        (2048, 128), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={"config": json.dumps(checkpoint_metadata)},
            )
            upsampler_path.write_text("unused", encoding="utf-8")
            host = LTXDistilledVideoGenerator(
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=upsampler_path,
            )
            imports = host._imports()
            object.__setattr__(imports, "load_audio_decoder", _fake_load_audio_decoder)

            raw_weights = {
                "audio_vae.decoder.conv_in.conv.weight": mx.zeros((512, 8, 3, 3)),
                "vocoder.vocoder.conv_pre.weight": mx.zeros((1536, 128, 7)),
            }

            with (
                patch.object(
                    helpers,
                    "_load_checkpoint_prefixed_weights",
                    return_value=raw_weights,
                ),
                patch.object(
                    helpers,
                    "_load_runtime_vocoder",
                    return_value=(_FakeVocoder(), 24000, "test_vocoder"),
                ),
            ):
                decoder, _, sample_rate, backend_label = helpers._ensure_audio_stack(
                    host,
                    imports,
                )

        self.assertIsInstance(decoder, _FakeDecoder)
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(backend_label, "test_vocoder")
        self.assertEqual(
            captured["weight_keys"],
            (
                "audio_vae.decoder.conv_in.conv.weight",
                "vocoder.vocoder.conv_pre.weight",
            ),
        )

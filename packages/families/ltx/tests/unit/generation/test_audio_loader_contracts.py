from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten
from mlxr.families.ltx._generation_backend.audio_autoencoder import (
    AudioCausalityAxis,
    AudioDecoderModel,
    AudioEncoderModel,
    AudioNormKind,
)
from mlxr.families.ltx._generation_backend.config import _runtime_audio_encoder_config
from mlxr.families.ltx._generation_backend.primitives import (
    sanitize_audio_vae_weights,
)
from mlxr.families.ltx._generation_backend.runtime_helpers import _ensure_audio_encoder
from mlxr.families.ltx._generation_backend.types import _RuntimeAudioEncoderConfig
from mlxr.families.ltx._generation_backend.video_stack import (
    _load_runtime_audio_decoder,
)


def _audio_runtime_config() -> _RuntimeAudioEncoderConfig:
    return _RuntimeAudioEncoderConfig(
        base_channels=32,
        ch_mult=(1, 2, 4),
        num_res_blocks=1,
        attn_resolutions=set(),
        resolution=16,
        latent_channels=2,
        dropout=0.0,
        in_channels=2,
        norm_type="group",
        causality_axis="none",
        mid_block_add_attention=False,
        sample_rate=16000,
        mel_hop_length=160,
        mel_bins=16,
        n_fft=1024,
        is_causal=False,
        double_z=False,
    )


def _checkpoint_style_audio_weights(
    module: AudioEncoderModel | AudioDecoderModel,
    *,
    prefix: str,
) -> dict[str, mx.array]:
    checkpoint_weights: dict[str, mx.array] = {}
    for key, value in tree_flatten(module.trainable_parameters()):
        if not isinstance(key, str):
            continue
        checkpoint_value = value
        if key.endswith("conv.weight") and value.ndim == 4:
            checkpoint_value = mx.transpose(value, (0, 3, 1, 2))
        checkpoint_weights[f"{prefix}{key}"] = checkpoint_value
    checkpoint_weights["audio_vae.per_channel_statistics.mean-of-means"] = (
        module.per_channel_statistics._mean_of_means
    )
    checkpoint_weights["audio_vae.per_channel_statistics.std-of-means"] = (
        module.per_channel_statistics._std_of_means
    )
    return checkpoint_weights


@dataclass
class _FakeAudioProcessor:
    sample_rate: int
    mel_bins: int
    mel_hop_length: int
    n_fft: int


@dataclass
class _FakeReferenceImports:
    audio_processor_class: object


@dataclass
class _FakeRuntimeHelperHost:
    checkpoint_path: Path
    spatial_upsampler_path: Path
    _reference_imports: object | None = None
    _transformer: object | None = None
    _vae_decoder: object | None = None
    _vae_encoder: object | None = None
    _upsampler: object | None = None
    _audio_encoder: object | None = None
    _audio_decoder: object | None = None
    _audio_processor: object | None = None
    _vocoder: object | None = None
    _audio_output_sample_rate: int | None = None
    _audio_backend: str | None = None


class AudioLoaderContractTests(unittest.TestCase):
    def test_audio_runtime_config_falls_back_to_checkpoint_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            checkpoint_root = Path(tmpdir)
            checkpoint_path = checkpoint_root / "checkpoint.safetensors"
            checkpoint_path.write_bytes(b"placeholder")

            metadata = {
                "audio_vae": {
                    "model": {
                        "params": {
                            "ddconfig": {
                                "ch_mult": [1, 2, 4],
                                "num_res_blocks": 2,
                                "attn_resolutions": [],
                                "resolution": 256,
                                "z_channels": 8,
                                "dropout": 0.0,
                                "in_channels": 2,
                                "norm_type": "pixel",
                                "causality_axis": "height",
                                "mid_block_add_attention": False,
                                "mel_bins": 64,
                                "double_z": True,
                            }
                        }
                    }
                }
            }

            with (
                patch(
                    "mlxr.families.ltx._generation_backend.config._checkpoint_file_for_root",
                    return_value=checkpoint_path,
                ),
                patch(
                    "mlxr.families.ltx._generation_backend.config._checkpoint_metadata",
                    return_value=metadata,
                ),
            ):
                config = _runtime_audio_encoder_config(checkpoint_root)

        self.assertFalse(config.mid_block_add_attention)

    def test_audio_encoder_loader_accepts_complete_checkpoint_contract(self) -> None:
        config = _audio_runtime_config()
        reference_encoder = AudioEncoderModel(
            base_channels=config.base_channels,
            channel_multipliers=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            in_channels=config.in_channels,
            resolution=config.resolution,
            latent_channels=config.latent_channels,
            double_z=config.double_z,
            norm_kind=AudioNormKind.GROUP,
            causality_axis=AudioCausalityAxis.NONE,
            mid_block_add_attention=config.mid_block_add_attention,
            sample_rate=config.sample_rate,
            mel_hop_length=config.mel_hop_length,
            n_fft=config.n_fft,
            mel_bins=config.mel_bins,
            is_causal=config.is_causal,
        )
        checkpoint_weights = _checkpoint_style_audio_weights(
            reference_encoder,
            prefix="audio_vae.encoder.",
        )
        host = _FakeRuntimeHelperHost(
            checkpoint_path=Path("/tmp/checkpoint.safetensors"),
            spatial_upsampler_path=Path("/tmp/upsampler.safetensors"),
        )
        imports = _FakeReferenceImports(audio_processor_class=_FakeAudioProcessor)

        with (
            patch(
                "mlxr.families.ltx._generation_backend.runtime_helpers._load_checkpoint_prefixed_weights",
                return_value=checkpoint_weights,
            ),
            patch(
                "mlxr.families.ltx._generation_backend.runtime_helpers._runtime_audio_encoder_config",
                return_value=config,
            ),
        ):
            encoder, processor = _ensure_audio_encoder(host, imports)

        self.assertEqual(
            tuple(
                int(size)
                for size in encoder.per_channel_statistics._mean_of_means.shape
            ),
            (8,),
        )
        self.assertIsInstance(processor, _FakeAudioProcessor)

    def test_audio_encoder_loader_rejects_missing_statistics(self) -> None:
        config = _audio_runtime_config()
        reference_encoder = AudioEncoderModel(
            base_channels=config.base_channels,
            channel_multipliers=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            in_channels=config.in_channels,
            resolution=config.resolution,
            latent_channels=config.latent_channels,
            double_z=config.double_z,
            norm_kind=AudioNormKind.GROUP,
            causality_axis=AudioCausalityAxis.NONE,
            mid_block_add_attention=config.mid_block_add_attention,
            sample_rate=config.sample_rate,
            mel_hop_length=config.mel_hop_length,
            n_fft=config.n_fft,
            mel_bins=config.mel_bins,
            is_causal=config.is_causal,
        )
        checkpoint_weights = _checkpoint_style_audio_weights(
            reference_encoder,
            prefix="audio_vae.encoder.",
        )
        checkpoint_weights.pop("audio_vae.per_channel_statistics.std-of-means")
        host = _FakeRuntimeHelperHost(
            checkpoint_path=Path("/tmp/checkpoint.safetensors"),
            spatial_upsampler_path=Path("/tmp/upsampler.safetensors"),
        )
        imports = _FakeReferenceImports(audio_processor_class=_FakeAudioProcessor)

        with (
            patch(
                "mlxr.families.ltx._generation_backend.runtime_helpers._load_checkpoint_prefixed_weights",
                return_value=checkpoint_weights,
            ),
            patch(
                "mlxr.families.ltx._generation_backend.runtime_helpers._runtime_audio_encoder_config",
                return_value=config,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "per-channel statistics"):
                _ensure_audio_encoder(host, imports)

    def test_audio_decoder_loader_accepts_complete_checkpoint_contract(self) -> None:
        config = _audio_runtime_config()
        reference_decoder = AudioDecoderModel(
            base_channels=config.base_channels,
            out_channels=config.in_channels,
            channel_multipliers=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            resolution=config.resolution,
            latent_channels=config.latent_channels,
            norm_kind=AudioNormKind.GROUP,
            causality_axis=AudioCausalityAxis.NONE,
            dropout=config.dropout,
            mid_block_add_attention=config.mid_block_add_attention,
            sample_rate=config.sample_rate,
            mel_hop_length=config.mel_hop_length,
            is_causal=config.is_causal,
            mel_bins=config.mel_bins,
        )
        checkpoint_weights = _checkpoint_style_audio_weights(
            reference_decoder,
            prefix="audio_vae.decoder.",
        )

        with patch(
            "mlxr.families.ltx._generation_backend.video_stack._runtime_audio_encoder_config",
            return_value=config,
        ):
            decoder = _load_runtime_audio_decoder(
                checkpoint_root=Path("/tmp/checkpoint"),
                sanitize_audio_vae_weights=sanitize_audio_vae_weights,
                unified_weights=checkpoint_weights,
            )

        self.assertEqual(
            tuple(
                int(size)
                for size in decoder.per_channel_statistics._mean_of_means.shape
            ),
            (8,),
        )

    def test_audio_decoder_loader_rejects_missing_required_weight(self) -> None:
        config = _audio_runtime_config()
        reference_decoder = AudioDecoderModel(
            base_channels=config.base_channels,
            out_channels=config.in_channels,
            channel_multipliers=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            resolution=config.resolution,
            latent_channels=config.latent_channels,
            norm_kind=AudioNormKind.GROUP,
            causality_axis=AudioCausalityAxis.NONE,
            dropout=config.dropout,
            mid_block_add_attention=config.mid_block_add_attention,
            sample_rate=config.sample_rate,
            mel_hop_length=config.mel_hop_length,
            is_causal=config.is_causal,
            mel_bins=config.mel_bins,
        )
        checkpoint_weights = _checkpoint_style_audio_weights(
            reference_decoder,
            prefix="audio_vae.decoder.",
        )
        checkpoint_weights.pop("audio_vae.decoder.conv_in.conv.weight")

        with patch(
            "mlxr.families.ltx._generation_backend.video_stack._runtime_audio_encoder_config",
            return_value=config,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing required weights"):
                _load_runtime_audio_decoder(
                    checkpoint_root=Path("/tmp/checkpoint"),
                    sanitize_audio_vae_weights=sanitize_audio_vae_weights,
                    unified_weights=checkpoint_weights,
                )


if __name__ == "__main__":
    unittest.main()

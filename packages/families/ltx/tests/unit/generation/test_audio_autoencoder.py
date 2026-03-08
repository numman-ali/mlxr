from __future__ import annotations

import unittest

import mlx.core as mx
from mlxr.families.ltx._generation_backend.audio_autoencoder import (
    AudioCausalityAxis,
    AudioDecoderModel,
    AudioEncoderModel,
    AudioLatentShape,
    AudioNormKind,
    AudioPatchifier,
    PerChannelStatistics,
)


class AudioAutoencoderContractTests(unittest.TestCase):
    def test_audio_patchifier_roundtrip_preserves_values(self) -> None:
        patchifier = AudioPatchifier(
            patch_size=1,
            sample_rate=16000,
            hop_length=160,
            audio_latent_downsample_factor=4,
            is_causal=True,
        )
        latents = mx.arange(1 * 2 * 3 * 4, dtype=mx.float32).reshape(1, 2, 3, 4)
        flattened = patchifier.patchify(latents)
        restored = patchifier.unpatchify(
            flattened,
            AudioLatentShape(batch=1, channels=2, frames=3, mel_bins=4),
        )

        self.assertTrue(mx.array_equal(latents, restored).item())

    def test_per_channel_statistics_roundtrip_restores_values(self) -> None:
        stats = PerChannelStatistics(width=8)
        values = mx.arange(16, dtype=mx.float32).reshape(2, 1, 8)

        normalized = stats.normalize(values)
        restored = stats.un_normalize(normalized)

        self.assertTrue(mx.allclose(values, restored).item())

    def test_audio_encoder_and_decoder_obey_owned_shape_contract(self) -> None:
        encoder = AudioEncoderModel(
            base_channels=32,
            channel_multipliers=(1, 2, 4),
            num_res_blocks=1,
            attn_resolutions=set(),
            dropout=0.0,
            in_channels=2,
            resolution=16,
            latent_channels=2,
            double_z=False,
            norm_kind=AudioNormKind.GROUP,
            causality_axis=AudioCausalityAxis.NONE,
            mid_block_add_attention=False,
            sample_rate=16000,
            mel_hop_length=160,
            n_fft=1024,
            mel_bins=16,
            is_causal=False,
        )
        decoder = AudioDecoderModel(
            base_channels=32,
            out_channels=2,
            channel_multipliers=(1, 2, 4),
            num_res_blocks=1,
            attn_resolutions=set(),
            resolution=16,
            latent_channels=2,
            norm_kind=AudioNormKind.GROUP,
            causality_axis=AudioCausalityAxis.NONE,
            dropout=0.0,
            mid_block_add_attention=False,
            sample_rate=16000,
            mel_hop_length=160,
            is_causal=False,
            mel_bins=16,
        )

        spectrogram = mx.zeros((1, 2, 16, 16), dtype=mx.float32)

        latents = encoder(spectrogram)
        decoded = decoder(latents)

        self.assertEqual(tuple(int(size) for size in latents.shape), (1, 2, 4, 4))
        self.assertEqual(tuple(int(size) for size in decoded.shape), (1, 2, 16, 16))


if __name__ == "__main__":
    unittest.main()

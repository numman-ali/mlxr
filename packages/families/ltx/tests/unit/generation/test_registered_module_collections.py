from __future__ import annotations

import unittest

from mlx.utils import tree_flatten
from mlxr.families.ltx._generation_backend.audio_autoencoder import (
    AudioCausalityAxis,
    AudioDecoderModel,
    AudioEncoderModel,
    AudioNormKind,
)
from mlxr.families.ltx._generation_backend.spatial_upsampler import LatentUpsampler
from mlxr.families.ltx._generation_backend.video_decoder_blocks import (
    PaddingModeType,
    ResBlockGroup,
)
from mlxr.families.ltx._generation_backend.video_encoder import (
    LatentLogVarianceType,
    VideoEncoder,
)
from mlxr.families.ltx._generation_backend.video_stack import _ConfiguredVideoDecoder


class RegisteredModuleCollectionsTests(unittest.TestCase):
    def test_res_block_group_registers_nested_blocks(self) -> None:
        group = ResBlockGroup(channels=32, num_layers=3)
        parameter_tree = tree_flatten(group.parameters(), destination={})

        self.assertIn("res_blocks.0.conv1.conv.conv.weight", parameter_tree)
        self.assertIn("res_blocks.1.conv2.conv.conv.bias", parameter_tree)
        self.assertIn("res_blocks.2.conv1.conv.conv.weight", parameter_tree)

    def test_video_encoder_registers_all_down_blocks(self) -> None:
        encoder = VideoEncoder(
            in_channels=3,
            out_channels=8,
            encoder_blocks=[
                ("res_x", {"num_layers": 2}),
                ("compress_all_res", {"multiplier": 2}),
            ],
            patch_size=1,
            latent_log_var=LatentLogVarianceType.NONE,
            encoder_spatial_padding_mode=PaddingModeType.REFLECT,
        )
        parameter_tree = tree_flatten(encoder.parameters(), destination={})

        self.assertIn(
            "down_blocks.0.res_blocks.0.conv1.conv.conv.weight", parameter_tree
        )
        self.assertIn("down_blocks.1.conv.conv.conv.weight", parameter_tree)

    def test_video_decoder_registers_all_up_blocks(self) -> None:
        decoder = _ConfiguredVideoDecoder(
            in_channels=8,
            out_channels=3,
            patch_size=1,
            decoder_blocks=(
                ("compress_all", {"multiplier": 2, "residual": False}),
                ("res_x", {"num_layers": 2}),
            ),
            base_channels=32,
            spatial_padding_mode=PaddingModeType.REFLECT,
            timestep_conditioning=False,
            causal_decoder=False,
        )
        parameter_tree = tree_flatten(decoder.parameters(), destination={})

        self.assertIn("up_blocks.0.res_blocks.0.conv1.conv.conv.weight", parameter_tree)
        self.assertIn("up_blocks.1.conv.conv.weight", parameter_tree)

    def test_spatial_upsampler_registers_residual_blocks(self) -> None:
        upsampler = LatentUpsampler(
            in_channels=8, mid_channels=32, num_blocks_per_stage=2
        )
        parameter_tree = tree_flatten(upsampler.parameters(), destination={})

        self.assertIn("res_blocks.0.conv1.weight", parameter_tree)
        self.assertIn("post_upsample_res_blocks.1.conv2.bias", parameter_tree)

    def test_audio_autoencoder_registers_stage_blocks(self) -> None:
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

        encoder_tree = tree_flatten(encoder.parameters(), destination={})
        decoder_tree = tree_flatten(decoder.parameters(), destination={})

        self.assertIn("down.0.block.0.conv1.conv.weight", encoder_tree)
        self.assertIn("down.1.block.0.conv1.conv.weight", encoder_tree)
        self.assertIn("up.0.block.0.conv1.conv.weight", decoder_tree)
        self.assertIn("up.1.block.1.conv2.conv.bias", decoder_tree)


if __name__ == "__main__":
    unittest.main()

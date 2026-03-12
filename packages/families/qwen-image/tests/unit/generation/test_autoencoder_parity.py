from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage as DiffusersAutoencoderKLQwenImage
from mlxr.families.qwen_image._generation_backend.autoencoder import (
    QwenImageAutoencoderDecoder,
)
from mlxr.families.qwen_image._generation_backend.config import AutoencoderConfig


class QwenImageAutoencoderParityTests(unittest.TestCase):
    def test_encode_matches_tiny_diffusers_forward(self) -> None:
        torch.manual_seed(0)
        diffusers_model = DiffusersAutoencoderKLQwenImage(
            base_dim=8,
            z_dim=4,
            dim_mult=[1, 2],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False],
            dropout=0.0,
            input_channels=3,
            latents_mean=[0.1, 0.2, 0.3, 0.4],
            latents_std=[1.0, 1.0, 1.0, 1.0],
        ).eval()
        mlx_model = QwenImageAutoencoderDecoder(
            AutoencoderConfig(
                attn_scales=(),
                base_dim=8,
                dim_mult=(1, 2),
                dropout=0.0,
                input_channels=3,
                latents_mean=(0.1, 0.2, 0.3, 0.4),
                latents_std=(1.0, 1.0, 1.0, 1.0),
                num_res_blocks=1,
                temperal_downsample=(False,),
                z_dim=4,
            )
        )
        mlx_model.load_weights(_autoencoder_weights(diffusers_model), strict=True)
        mx.eval(mlx_model.parameters())

        images = torch.randn(1, 3, 1, 16, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = diffusers_model.encode(images).latent_dist.mode()

        actual = mlx_model.encode(mx.array(images.numpy()))

        np.testing.assert_allclose(
            np.asarray(actual),
            expected.detach().cpu().numpy(),
            atol=1.0e-4,
            rtol=1.0e-4,
        )

    def test_decode_matches_tiny_diffusers_forward(self) -> None:
        torch.manual_seed(0)
        diffusers_model = DiffusersAutoencoderKLQwenImage(
            base_dim=8,
            z_dim=4,
            dim_mult=[1, 2],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False],
            dropout=0.0,
            input_channels=3,
            latents_mean=[0.1, 0.2, 0.3, 0.4],
            latents_std=[1.0, 1.0, 1.0, 1.0],
        ).eval()
        mlx_model = QwenImageAutoencoderDecoder(
            AutoencoderConfig(
                attn_scales=(),
                base_dim=8,
                dim_mult=(1, 2),
                dropout=0.0,
                input_channels=3,
                latents_mean=(0.1, 0.2, 0.3, 0.4),
                latents_std=(1.0, 1.0, 1.0, 1.0),
                num_res_blocks=1,
                temperal_downsample=(False,),
                z_dim=4,
            )
        )
        mlx_model.load_weights(_autoencoder_weights(diffusers_model), strict=True)
        mx.eval(mlx_model.parameters())

        latents = torch.randn(1, 4, 1, 8, 8, dtype=torch.float32)

        with torch.no_grad():
            expected = diffusers_model.decode(latents, return_dict=False)[0]

        actual = mlx_model.decode(mx.array(latents.numpy()))

        np.testing.assert_allclose(
            np.asarray(actual),
            expected.detach().cpu().numpy(),
            atol=1.0e-4,
            rtol=1.0e-4,
        )

    def test_tiled_encode_matches_tiny_diffusers_forward(self) -> None:
        torch.manual_seed(0)
        diffusers_model = DiffusersAutoencoderKLQwenImage(
            base_dim=8,
            z_dim=4,
            dim_mult=[1, 2],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False],
            dropout=0.0,
            input_channels=3,
            latents_mean=[0.1, 0.2, 0.3, 0.4],
            latents_std=[1.0, 1.0, 1.0, 1.0],
        ).eval()
        diffusers_model.enable_tiling(
            tile_sample_min_height=8,
            tile_sample_min_width=8,
            tile_sample_stride_height=6,
            tile_sample_stride_width=6,
        )
        mlx_model = QwenImageAutoencoderDecoder(
            AutoencoderConfig(
                attn_scales=(),
                base_dim=8,
                dim_mult=(1, 2),
                dropout=0.0,
                input_channels=3,
                latents_mean=(0.1, 0.2, 0.3, 0.4),
                latents_std=(1.0, 1.0, 1.0, 1.0),
                num_res_blocks=1,
                temperal_downsample=(False,),
                z_dim=4,
            )
        )
        mlx_model.load_weights(_autoencoder_weights(diffusers_model), strict=True)
        mlx_model.enable_tiling(
            tile_sample_min_height=8,
            tile_sample_min_width=8,
            tile_sample_stride_height=6,
            tile_sample_stride_width=6,
        )
        mx.eval(mlx_model.parameters())

        images = torch.randn(1, 3, 1, 32, 32, dtype=torch.float32)

        with torch.no_grad():
            expected = diffusers_model.encode(images).latent_dist.mode()

        actual = mlx_model.encode(mx.array(images.numpy()))

        np.testing.assert_allclose(
            np.asarray(actual),
            expected.detach().cpu().numpy(),
            atol=1.0e-4,
            rtol=1.0e-4,
        )

    def test_tiled_decode_matches_tiny_diffusers_forward(self) -> None:
        torch.manual_seed(0)
        diffusers_model = DiffusersAutoencoderKLQwenImage(
            base_dim=8,
            z_dim=4,
            dim_mult=[1, 2],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False],
            dropout=0.0,
            input_channels=3,
            latents_mean=[0.1, 0.2, 0.3, 0.4],
            latents_std=[1.0, 1.0, 1.0, 1.0],
        ).eval()
        diffusers_model.enable_tiling(
            tile_sample_min_height=8,
            tile_sample_min_width=8,
            tile_sample_stride_height=6,
            tile_sample_stride_width=6,
        )
        mlx_model = QwenImageAutoencoderDecoder(
            AutoencoderConfig(
                attn_scales=(),
                base_dim=8,
                dim_mult=(1, 2),
                dropout=0.0,
                input_channels=3,
                latents_mean=(0.1, 0.2, 0.3, 0.4),
                latents_std=(1.0, 1.0, 1.0, 1.0),
                num_res_blocks=1,
                temperal_downsample=(False,),
                z_dim=4,
            )
        )
        mlx_model.load_weights(_autoencoder_weights(diffusers_model), strict=True)
        mlx_model.enable_tiling(
            tile_sample_min_height=8,
            tile_sample_min_width=8,
            tile_sample_stride_height=6,
            tile_sample_stride_width=6,
        )
        mx.eval(mlx_model.parameters())

        latents = torch.randn(1, 4, 1, 16, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = diffusers_model.decode(latents, return_dict=False)[0]

        actual = mlx_model.decode(mx.array(latents.numpy()))

        np.testing.assert_allclose(
            np.asarray(actual),
            expected.detach().cpu().numpy(),
            atol=1.0e-4,
            rtol=1.0e-4,
        )


def _autoencoder_weights(
    model: DiffusersAutoencoderKLQwenImage,
) -> list[tuple[str, mx.array]]:
    weights: list[tuple[str, mx.array]] = []
    for name, tensor in model.state_dict().items():
        if not name.startswith(
            ("encoder.", "quant_conv.", "post_quant_conv.", "decoder.")
        ):
            continue
        aliased = name.replace(".resample.1.", ".resample_conv.")
        value = tensor.detach().cpu().numpy()
        squeezed = np.squeeze(value)
        if squeezed.ndim == 1:
            value = squeezed
        elif value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        elif value.ndim == 5:
            value = value.transpose(0, 2, 3, 4, 1)
        else:
            value = squeezed
        weights.append((aliased, mx.array(value)))
    return weights


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlxr.families.flux2._generation_backend.autoencoder import AutoencoderKLFlux2
from mlxr.families.flux2._generation_backend.config import (
    AutoencoderConfig,
    Flux2TransformerConfig,
    SchedulerConfig,
)
from mlxr.families.flux2._generation_backend.transformer import Flux2Transformer2DModel


def tiny_transformer_config(*, guidance_embeds: bool = True) -> Flux2TransformerConfig:
    return Flux2TransformerConfig(
        attention_head_dim=4,
        axes_dims_rope=(2, 2),
        eps=1.0e-6,
        guidance_embeds=guidance_embeds,
        in_channels=8,
        joint_attention_dim=12,
        mlp_ratio=2.0,
        num_attention_heads=2,
        num_layers=1,
        num_single_layers=1,
        rope_theta=10_000.0,
        timestep_guidance_channels=8,
    )


def tiny_autoencoder_config() -> AutoencoderConfig:
    return AutoencoderConfig(
        block_out_channels=(32, 32),
        force_upcast=True,
        in_channels=3,
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=32,
        out_channels=3,
        patch_size=(2, 2),
        use_post_quant_conv=True,
        use_quant_conv=True,
    )


def tiny_scheduler_config(*, use_dynamic_shifting: bool = False) -> SchedulerConfig:
    return SchedulerConfig(
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=4096,
        max_shift=1.15,
        num_train_timesteps=1000,
        use_dynamic_shifting=use_dynamic_shifting,
    )


def write_tiny_bundle(root: Path) -> None:
    transformer = Flux2Transformer2DModel(tiny_transformer_config())
    autoencoder = AutoencoderKLFlux2(tiny_autoencoder_config())

    transformer_dir = root / "transformer"
    autoencoder_dir = root / "vae"
    scheduler_dir = root / "scheduler"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    autoencoder_dir.mkdir(parents=True, exist_ok=True)
    scheduler_dir.mkdir(parents=True, exist_ok=True)

    (transformer_dir / "config.json").write_text(
        json.dumps(asdict(tiny_transformer_config())),
        encoding="utf-8",
    )
    (autoencoder_dir / "config.json").write_text(
        json.dumps(asdict(tiny_autoencoder_config())),
        encoding="utf-8",
    )
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps(asdict(tiny_scheduler_config())),
        encoding="utf-8",
    )

    mx.save_safetensors(
        str(transformer_dir / "diffusion_pytorch_model.safetensors"),
        dict(tree_flatten(transformer.parameters(), destination={})),
    )
    mx.save_safetensors(
        str(autoencoder_dir / "diffusion_pytorch_model.safetensors"),
        dict(tree_flatten(autoencoder.parameters(), destination={})),
    )

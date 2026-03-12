from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Flux2TransformerConfig:
    attention_head_dim: int
    axes_dims_rope: tuple[int, ...]
    eps: float
    guidance_embeds: bool
    in_channels: int
    joint_attention_dim: int
    mlp_ratio: float
    num_attention_heads: int
    num_layers: int
    num_single_layers: int
    rope_theta: float
    timestep_guidance_channels: int

    @classmethod
    def from_path(cls, config_path: Path) -> Flux2TransformerConfig:
        raw = json.loads(config_path.read_text("utf-8"))
        return cls(
            attention_head_dim=int(raw["attention_head_dim"]),
            axes_dims_rope=tuple(int(value) for value in raw["axes_dims_rope"]),
            eps=float(raw["eps"]),
            guidance_embeds=bool(raw.get("guidance_embeds", False)),
            in_channels=int(raw["in_channels"]),
            joint_attention_dim=int(raw["joint_attention_dim"]),
            mlp_ratio=float(raw["mlp_ratio"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_layers=int(raw["num_layers"]),
            num_single_layers=int(raw["num_single_layers"]),
            rope_theta=float(raw["rope_theta"]),
            timestep_guidance_channels=int(raw["timestep_guidance_channels"]),
        )

    @property
    def hidden_size(self) -> int:
        return self.attention_head_dim * self.num_attention_heads


@dataclass(frozen=True, slots=True)
class AutoencoderConfig:
    block_out_channels: tuple[int, ...]
    force_upcast: bool
    in_channels: int
    latent_channels: int
    layers_per_block: int
    norm_num_groups: int
    out_channels: int
    patch_size: tuple[int, int]
    use_post_quant_conv: bool
    use_quant_conv: bool

    @classmethod
    def from_path(cls, config_path: Path) -> AutoencoderConfig:
        raw = json.loads(config_path.read_text("utf-8"))
        raw_patch_size = tuple(int(value) for value in raw["patch_size"])
        if len(raw_patch_size) != 2:
            raise ValueError(
                f"FLUX.2 autoencoder patch_size must have length 2, got {raw_patch_size}"
            )
        return cls(
            block_out_channels=tuple(int(value) for value in raw["block_out_channels"]),
            force_upcast=bool(raw.get("force_upcast", True)),
            in_channels=int(raw["in_channels"]),
            latent_channels=int(raw["latent_channels"]),
            layers_per_block=int(raw["layers_per_block"]),
            norm_num_groups=int(raw["norm_num_groups"]),
            out_channels=int(raw["out_channels"]),
            patch_size=(raw_patch_size[0], raw_patch_size[1]),
            use_post_quant_conv=bool(raw.get("use_post_quant_conv", True)),
            use_quant_conv=bool(raw.get("use_quant_conv", True)),
        )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    base_image_seq_len: int
    base_shift: float
    max_image_seq_len: int
    max_shift: float
    num_train_timesteps: int
    use_dynamic_shifting: bool

    @classmethod
    def from_path(cls, config_path: Path) -> SchedulerConfig:
        raw = json.loads(config_path.read_text("utf-8"))
        return cls(
            base_image_seq_len=int(raw["base_image_seq_len"]),
            base_shift=float(raw["base_shift"]),
            max_image_seq_len=int(raw["max_image_seq_len"]),
            max_shift=float(raw["max_shift"]),
            num_train_timesteps=int(raw["num_train_timesteps"]),
            use_dynamic_shifting=bool(raw.get("use_dynamic_shifting", False)),
        )

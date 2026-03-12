from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class QwenImageTransformerConfig:
    attention_head_dim: int
    axes_dims_rope: tuple[int, int, int]
    guidance_embeds: bool
    in_channels: int
    joint_attention_dim: int
    num_attention_heads: int
    num_layers: int
    out_channels: int
    patch_size: int
    pooled_projection_dim: int
    use_additional_t_cond: bool
    use_layer3d_rope: bool
    zero_cond_t: bool

    @classmethod
    def from_path(cls, config_path: Path) -> QwenImageTransformerConfig:
        raw = json.loads(config_path.read_text("utf-8"))
        raw_axes_dims = raw.get("axes_dims_rope", (16, 56, 56))
        axes_dims_rope = tuple(int(value) for value in raw_axes_dims)
        if len(axes_dims_rope) != 3:
            raise ValueError("Qwen-Image axes_dims_rope must contain exactly 3 values")
        return cls(
            attention_head_dim=int(raw["attention_head_dim"]),
            axes_dims_rope=(
                axes_dims_rope[0],
                axes_dims_rope[1],
                axes_dims_rope[2],
            ),
            guidance_embeds=bool(raw.get("guidance_embeds", False)),
            in_channels=int(raw["in_channels"]),
            joint_attention_dim=int(raw["joint_attention_dim"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_layers=int(raw["num_layers"]),
            out_channels=int(raw["out_channels"]),
            patch_size=int(raw["patch_size"]),
            pooled_projection_dim=int(raw.get("pooled_projection_dim", 0)),
            use_additional_t_cond=bool(raw.get("use_additional_t_cond", False)),
            use_layer3d_rope=bool(raw.get("use_layer3d_rope", False)),
            zero_cond_t=bool(raw.get("zero_cond_t", False)),
        )

    @property
    def hidden_size(self) -> int:
        return self.attention_head_dim * self.num_attention_heads

    @property
    def latent_channels(self) -> int:
        return self.in_channels // 4


@dataclass(frozen=True, slots=True)
class AutoencoderConfig:
    attn_scales: tuple[float, ...]
    base_dim: int
    dim_mult: tuple[int, ...]
    dropout: float
    input_channels: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]
    num_res_blocks: int
    temperal_downsample: tuple[bool, ...]
    z_dim: int

    @classmethod
    def from_path(cls, config_path: Path) -> AutoencoderConfig:
        raw = json.loads(config_path.read_text("utf-8"))
        return cls(
            attn_scales=tuple(float(value) for value in raw.get("attn_scales", [])),
            base_dim=int(raw.get("base_dim", 96)),
            dim_mult=tuple(int(value) for value in raw.get("dim_mult", [1, 2, 4, 4])),
            dropout=float(raw.get("dropout", 0.0)),
            input_channels=int(raw.get("input_channels", 3)),
            latents_mean=tuple(float(value) for value in raw["latents_mean"]),
            latents_std=tuple(float(value) for value in raw["latents_std"]),
            num_res_blocks=int(raw.get("num_res_blocks", 2)),
            temperal_downsample=tuple(
                bool(value) for value in raw["temperal_downsample"]
            ),
            z_dim=int(raw["z_dim"]),
        )

    @property
    def scale_factor(self) -> int:
        return int(2 ** len(self.temperal_downsample))

    @property
    def pixel_multiple(self) -> int:
        # Qwen-Image packs 2x2 latent blocks after the 8x VAE compression.
        return self.scale_factor * 2

    @property
    def temporal_upsample(self) -> tuple[bool, ...]:
        return tuple(reversed(self.temperal_downsample))


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    base_image_seq_len: int
    base_shift: float
    invert_sigmas: bool
    max_image_seq_len: int
    max_shift: float
    num_train_timesteps: int
    shift_terminal: float | None
    time_shift_type: str
    use_dynamic_shifting: bool

    @classmethod
    def from_path(cls, config_path: Path) -> SchedulerConfig:
        raw = json.loads(config_path.read_text("utf-8"))
        shift_terminal = raw.get("shift_terminal")
        return cls(
            base_image_seq_len=int(raw["base_image_seq_len"]),
            base_shift=float(raw["base_shift"]),
            invert_sigmas=bool(raw.get("invert_sigmas", False)),
            max_image_seq_len=int(raw["max_image_seq_len"]),
            max_shift=float(raw["max_shift"]),
            num_train_timesteps=int(raw["num_train_timesteps"]),
            shift_terminal=(
                float(shift_terminal)
                if isinstance(shift_terminal, (int, float))
                else None
            ),
            time_shift_type=str(raw.get("time_shift_type", "exponential")),
            use_dynamic_shifting=bool(raw.get("use_dynamic_shifting", False)),
        )

    def with_overrides(
        self,
        *,
        base_shift: float | None = None,
        max_shift: float | None = None,
        shift_terminal: float | None = None,
        preserve_shift_terminal: bool = True,
    ) -> SchedulerConfig:
        return SchedulerConfig(
            base_image_seq_len=self.base_image_seq_len,
            base_shift=float(self.base_shift if base_shift is None else base_shift),
            invert_sigmas=self.invert_sigmas,
            max_image_seq_len=self.max_image_seq_len,
            max_shift=float(self.max_shift if max_shift is None else max_shift),
            num_train_timesteps=self.num_train_timesteps,
            shift_terminal=(
                self.shift_terminal if preserve_shift_terminal else shift_terminal
            ),
            time_shift_type=self.time_shift_type,
            use_dynamic_shifting=self.use_dynamic_shifting,
        )

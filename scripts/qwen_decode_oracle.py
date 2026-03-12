from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage
from mlxr.families.qwen_image._generation_backend.runtime import (
    GenerationTrace,
    _RuntimeImageGenerator,
    create_image_generator,
)
from mlxr.families.qwen_image.prompt_encoding import create_prompt_encoder
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "manual-runs"


@dataclass(frozen=True, slots=True)
class OracleArgs:
    model_root: Path
    variant: str
    task: str
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    scheduler_preset: str
    image_paths: tuple[Path, ...]
    lora_paths: tuple[Path, ...]
    lora_scales: tuple[float, ...]
    output_dir: Path


def _parse_args() -> OracleArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Run owned Qwen generation once, save final latents, and decode those "
            "same latents with owned MLX and official diffusers Qwen VAE."
        )
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--task", type=str, default="image.generate")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--scheduler-preset",
        type=str,
        default="default",
        choices=("default", "lightning", "turbo_wuli"),
    )
    parser.add_argument(
        "--image", dest="image_paths", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--lora", dest="lora_paths", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--lora-scale",
        dest="lora_scales",
        type=float,
        action="append",
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parsed = parser.parse_args()
    if len(parsed.lora_scales) not in {0, len(parsed.lora_paths)}:
        raise SystemExit(
            "--lora-scale must be omitted or provided once for every --lora"
        )
    output_dir = (
        parsed.output_dir
        if parsed.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / f"qwen-decode-oracle-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    scales = (
        tuple(float(value) for value in parsed.lora_scales)
        if parsed.lora_scales
        else tuple(1.0 for _ in parsed.lora_paths)
    )
    return OracleArgs(
        model_root=parsed.model_root,
        variant=parsed.variant,
        task=parsed.task,
        prompt=parsed.prompt,
        negative_prompt=parsed.negative_prompt,
        width=int(parsed.width),
        height=int(parsed.height),
        num_inference_steps=int(parsed.num_inference_steps),
        guidance_scale=float(parsed.guidance_scale),
        seed=int(parsed.seed),
        scheduler_preset=str(parsed.scheduler_preset),
        image_paths=tuple(Path(value) for value in parsed.image_paths),
        lora_paths=tuple(Path(value) for value in parsed.lora_paths),
        lora_scales=scales,
        output_dir=output_dir,
    )


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = create_prompt_encoder(
        text_encoder_path=args.model_root / "text_encoder",
        tokenizer_path=args.model_root / "tokenizer",
        processor_path=(
            args.model_root / "processor"
            if (args.model_root / "processor").exists()
            else None
        ),
        task=args.task,
    )
    generator = create_image_generator(
        variant=args.variant,
        model_root=args.model_root,
        task=args.task,
        quantize_bits=None,
        scheduler_preset=args.scheduler_preset,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )
    if not isinstance(generator, _RuntimeImageGenerator):
        raise RuntimeError(
            "Expected the owned Qwen backend to return _RuntimeImageGenerator"
        )

    prompt_context = encoder.encode(
        args.prompt,
        max_length=1024,
        negative_prompt=args.negative_prompt,
        image_paths=args.image_paths,
    )
    trace = generator.debug_generate(
        prompt_context=prompt_context,
        task=args.task,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        image_paths=args.image_paths,
    )
    _save_trace_arrays(trace, args.output_dir)
    _save_png(trace.decoded, args.output_dir / "owned_decode.png")
    diffusers_plain = _decode_with_diffusers(
        args.model_root / "vae",
        trace.denormalized_latents,
        enable_tiling=False,
    )
    diffusers_tiled = _decode_with_diffusers(
        args.model_root / "vae",
        trace.denormalized_latents,
        enable_tiling=True,
    )
    _save_png(diffusers_plain, args.output_dir / "diffusers_decode.png")
    _save_png(diffusers_tiled, args.output_dir / "diffusers_tiled_decode.png")
    manifest = {
        "args": _jsonable(asdict(args)),
        "owned_metadata": _jsonable(trace.metadata),
        "diffusers_plain_vs_owned": _decode_stats(trace.decoded, diffusers_plain),
        "diffusers_tiled_vs_owned": _decode_stats(trace.decoded, diffusers_tiled),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    encoder.close()
    generator.close()
    return 0


def _save_trace_arrays(trace: GenerationTrace, output_dir: Path) -> None:
    np.save(output_dir / "packed_sample.npy", _to_numpy(trace.sample))
    np.save(output_dir / "unpacked_latents.npy", _to_numpy(trace.unpacked_latents))
    np.save(
        output_dir / "denormalized_latents.npy",
        _to_numpy(trace.denormalized_latents),
    )


def _decode_with_diffusers(
    vae_path: Path,
    denormalized_latents: mx.array,
    *,
    enable_tiling: bool,
) -> np.ndarray:
    vae = AutoencoderKLQwenImage.from_pretrained(  # type: ignore[no-untyped-call]
        str(vae_path),
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    if enable_tiling:
        vae.enable_tiling()
    latents = torch.from_numpy(_to_numpy(denormalized_latents))
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    return cast(np.ndarray, decoded.detach().cpu().numpy())


def _save_png(decoded: mx.array | np.ndarray, output_path: Path) -> None:
    array = _to_numpy(decoded)
    frame = array[0, :, 0, :, :].transpose(1, 2, 0)
    pixels = np.clip((frame + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(output_path, format="PNG")


def _decode_stats(
    left: mx.array | np.ndarray, right: mx.array | np.ndarray
) -> dict[str, Any]:
    left_array = _to_numpy(left)
    right_array = _to_numpy(right)
    diff = left_array - right_array
    return {
        "mean_abs_diff": float(np.abs(diff).mean()),
        "max_abs_diff": float(np.abs(diff).max()),
        "root_mean_square_diff": float(np.sqrt(np.mean(np.square(diff)))),
    }


def _to_numpy(value: mx.array | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return np.array(value.astype(mx.float32), dtype=np.float32)


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())

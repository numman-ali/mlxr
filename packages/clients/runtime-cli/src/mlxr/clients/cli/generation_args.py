from __future__ import annotations

import argparse
import json


def _validate_generate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if bool(args.plan_only) and bool(args.wait):
        parser.error("--wait cannot be used with --plan-only")
    if args.export_path is not None and not bool(args.wait):
        parser.error("--export-path requires --wait")
    if bool(args.overwrite_export) and args.export_path is None:
        parser.error("--overwrite-export requires --export-path")
    if float(args.timeout_seconds) <= 0:
        parser.error("--timeout-seconds must be greater than 0")
    if float(args.poll_interval_seconds) < 0:
        parser.error("--poll-interval-seconds must be non-negative")
    if args.keyframe_image and (
        int(args.image_frame_index) != 0 or float(args.image_strength) != 1.0
    ):
        parser.error(
            "--keyframe-image cannot be mixed with non-default --image-frame-index "
            "or --image-strength values"
        )
    if int(args.image_frame_index) < 0:
        parser.error("--image-frame-index must be non-negative")
    if not (0.0 <= float(args.image_strength) <= 1.0):
        parser.error("--image-strength must be between 0.0 and 1.0")
    for keyframe in args.keyframe_image:
        if keyframe.frame_index < 0:
            parser.error("--keyframe-image frame index must be non-negative")
        if not (0.0 <= keyframe.strength <= 1.0):
            parser.error("--keyframe-image strength must be between 0.0 and 1.0")
    for lora in args.lora:
        if lora.strength <= 0.0:
            parser.error("--lora strength must be greater than 0.0")
    if float(args.audio_start_seconds) < 0.0:
        parser.error("--audio-start-seconds must be non-negative")
    if (
        args.audio_max_duration_seconds is not None
        and float(args.audio_max_duration_seconds) <= 0.0
    ):
        parser.error("--audio-max-duration-seconds must be greater than 0")
    if args.num_inference_steps is not None and int(args.num_inference_steps) <= 0:
        parser.error("--num-inference-steps must be greater than 0")
    if args.guidance_scale is not None and float(args.guidance_scale) < 0.0:
        parser.error("--guidance-scale must be non-negative")
    if (args.window_start_seconds is None) != (args.window_end_seconds is None):
        parser.error(
            "--window-start-seconds and --window-end-seconds must be provided together"
        )
    if args.window_start_seconds is not None and float(args.window_start_seconds) < 0.0:
        parser.error("--window-start-seconds must be non-negative")
    if args.window_end_seconds is not None and float(args.window_end_seconds) <= float(
        args.window_start_seconds
    ):
        parser.error("--window-end-seconds must be greater than --window-start-seconds")
    if args.extensions_json is not None:
        try:
            from .generation import _parse_extensions_json

            _parse_extensions_json(str(args.extensions_json))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))

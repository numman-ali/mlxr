#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

from ltx.prompting import PromptShapingOptions, shape_text_first_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "showcase-runs"
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT
    / "tmp"
    / "real-benchmark-output"
    / "20260306T230257Z"
    / "t2v"
    / "runtime-home"
    / "artifacts-portable"
    / "ltx"
    / "ltx-benchmark-huggingface-t2v"
    / "sha256_11927891e2f2f2c46e30f5af8cb529f7da9032526e7fc5e5330a1e9df4b901b6"
    / "payload"
)


ExpectedAudioType = Literal[
    "music",
    "ambience",
    "barking",
    "speech",
    "other",
    "unknown",
]


class GeminiReview(TypedDict):
    visual_summary: str
    primary_subject: str
    scene: str
    style: str
    camera: str
    audio_present: str
    audio_type: str
    audio_summary: str
    confidence: str
    verdict: str


class SmokeRunResult(TypedDict):
    video_path: str
    manifest_path: str


class SceneSummary(TypedDict):
    scene_id: str
    title: str
    prompt: str
    video_path: str
    manifest_path: str
    ffprobe_path: str
    gemini_review: GeminiReview
    review_ok: bool
    expected_subject: str
    expected_audio_types: list[ExpectedAudioType]
    gemini_review_error: str | None


@dataclass(frozen=True, slots=True)
class ShowcaseScene:
    scene_id: str
    title: str
    base_prompt: str
    shaping: PromptShapingOptions
    expected_subject_terms: tuple[str, ...]
    expected_scene_terms: tuple[str, ...]
    expected_audio_types: tuple[ExpectedAudioType, ...]
    notes: str


SHOWCASE_SCENES: tuple[ShowcaseScene, ...] = (
    ShowcaseScene(
        scene_id="dog_park_natural",
        title="Dog In Park",
        base_prompt=(
            "A golden retriever runs happily beside its owner through a sunlit park "
            "path, looking up toward them as they move together past green grass and "
            "trees, filmed in a smooth handheld tracking shot at waist height with warm "
            "natural afternoon light."
        ),
        shaping=PromptShapingOptions(
            audio_prompt=(
                "happy dog barks, light footsteps on the path, soft wind in the trees, "
                "distant birds, and quiet park ambience"
            ),
            natural_audio=True,
            no_music=True,
            duration_seconds=10.0,
            orientation="landscape",
        ),
        expected_subject_terms=("dog", "golden retriever"),
        expected_scene_terms=("park", "path"),
        expected_audio_types=("barking", "ambience"),
        notes="Naturalistic dog clip with no music allowed.",
    ),
    ShowcaseScene(
        scene_id="anime_neon_chase",
        title="Anime Neon Chase",
        base_prompt=(
            "An anime-style courier sprints through a neon city at night while glowing "
            "signs streak past, dodging traffic and leaping over puddles in a high-energy "
            "side-tracking chase shot with bold color and exaggerated motion."
        ),
        shaping=PromptShapingOptions(
            audio_prompt=(
                "fast urban footsteps, distant traffic, neon-city ambience, and energetic "
                "anime-style action sound design"
            ),
            duration_seconds=10.0,
            orientation="landscape",
        ),
        expected_subject_terms=("courier", "runner", "character", "girl"),
        expected_scene_terms=("city", "street", "neon"),
        expected_audio_types=("music", "other", "ambience"),
        notes="Stylized anime motion with energetic audio allowed.",
    ),
    ShowcaseScene(
        scene_id="vintage_old_school",
        title="Vintage Street Scene",
        base_prompt=(
            "A vintage 1970s street scene unfolds outside a neighborhood shop as people "
            "walk past parked cars and the late-afternoon sun creates warm flares, shot "
            "like restored 16mm footage with slight film softness and period styling."
        ),
        shaping=PromptShapingOptions(
            audio_prompt=(
                "street ambience, footsteps, distant conversation, and subtle vintage "
                "environment texture with no modern soundtrack feel"
            ),
            natural_audio=True,
            no_music=True,
            duration_seconds=10.0,
            orientation="landscape",
        ),
        expected_subject_terms=("person", "people", "pedestrian"),
        expected_scene_terms=("street", "shop", "cars", "sidewalk"),
        expected_audio_types=("ambience", "speech", "other"),
        notes="Vintage realism; avoid obvious modern music.",
    ),
    ShowcaseScene(
        scene_id="nature_documentary",
        title="Nature Documentary",
        base_prompt=(
            "A nature-documentary shot follows a red fox moving carefully through tall "
            "grass at the edge of a forest meadow in soft morning light, captured with a "
            "calm telephoto documentary feel and gentle camera tracking."
        ),
        shaping=PromptShapingOptions(
            audio_prompt=(
                "natural meadow ambience, birds, wind through grass, and soft animal "
                "movement only"
            ),
            natural_audio=True,
            no_music=True,
            duration_seconds=10.0,
            orientation="landscape",
        ),
        expected_subject_terms=("fox",),
        expected_scene_terms=("meadow", "forest", "grass"),
        expected_audio_types=("ambience", "other"),
        notes="Documentary scene; natural ambience only.",
    ),
    ShowcaseScene(
        scene_id="stop_motion_workshop",
        title="Stop Motion Workshop",
        base_prompt=(
            "A handcrafted stop-motion fox made of felt and clay explores a tiny workshop "
            "table filled with tools and paper props, moving in tactile little beats under "
            "warm lamp light with a charming miniature-cinema look."
        ),
        shaping=PromptShapingOptions(
            audio_prompt=(
                "tiny handcrafted foley, soft tabletop movement, paper rustle, and warm "
                "room ambience"
            ),
            duration_seconds=10.0,
            orientation="landscape",
        ),
        expected_subject_terms=("fox",),
        expected_scene_terms=("workshop", "table", "miniature", "desk", "workbench", "lamp"),
        expected_audio_types=("other", "ambience", "music"),
        notes="Stylized handcrafted scene; foley-like audio preferred.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and review the first LTX showcase clips."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Converted LTX artifact root to use for showcase generation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for showcase receipts and summary files.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        choices=tuple(scene.scene_id for scene in SHOWCASE_SCENES),
        help="Optional scene id to run. Repeat to run a subset.",
    )
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--num-frames", type=int, default=241)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--strict-review", action="store_true", default=True)
    parser.add_argument(
        "--no-strict-review",
        dest="strict_review",
        action="store_false",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = _selected_scenes(args.scene)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary: list[SceneSummary] = []
    for scene in selected:
        prompt = shape_text_first_prompt(scene.base_prompt, options=scene.shaping)
        run_name = f"showcase-{scene.scene_id}"
        run_result = _run_smoke(
            artifact_root=args.artifact_root.expanduser().resolve(),
            output_root=output_root,
            run_name=run_name,
            prompt=prompt,
            width=int(args.width),
            height=int(args.height),
            num_frames=int(args.num_frames),
            fps=int(args.fps),
        )
        ffprobe_path = _write_ffprobe(Path(run_result["video_path"]))
        gemini_review_error: str | None = None
        parsed_review: GeminiReview | None = None
        try:
            parsed_review = _run_gemini_review(
                video_path=Path(run_result["video_path"])
            )
        except RuntimeError as error:
            gemini_review_error = str(error)
        review_ok = parsed_review is not None and _review_matches_scene(
            parsed_review=parsed_review,
            scene=scene,
            strict=bool(args.strict_review),
        )
        scene_summary: SceneSummary = {
            "scene_id": scene.scene_id,
            "title": scene.title,
            "prompt": prompt,
            "video_path": run_result["video_path"],
            "manifest_path": run_result["manifest_path"],
            "ffprobe_path": str(ffprobe_path),
            "gemini_review": parsed_review or _unknown_review(),
            "review_ok": review_ok,
            "expected_subject": ", ".join(scene.expected_subject_terms),
            "expected_audio_types": list(scene.expected_audio_types),
            "gemini_review_error": gemini_review_error,
        }
        summary.append(scene_summary)

    summary_path = output_root / (
        f"showcase-summary-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    payload = {"summary_path": str(summary_path), "scenes": summary}
    print(json.dumps(payload, indent=2))
    if any(not scene["review_ok"] for scene in summary):
        return 2
    return 0


def _selected_scenes(scene_ids: list[str] | None) -> tuple[ShowcaseScene, ...]:
    if not scene_ids:
        return SHOWCASE_SCENES
    allowed = set(scene_ids)
    return tuple(scene for scene in SHOWCASE_SCENES if scene.scene_id in allowed)


def _run_smoke(
    *,
    artifact_root: Path,
    output_root: Path,
    run_name: str,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
) -> dict[str, str]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/ltx_debug_smoke.py",
        "--artifact-root",
        str(artifact_root),
        "--prompt",
        prompt,
        "--width",
        str(width),
        "--height",
        str(height),
        "--num-frames",
        str(num_frames),
        "--fps",
        str(fps),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        "--no-stage-debug",
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, cwd=REPO_ROOT
    )
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    return {
        "video_path": str(payload["video_path"]),
        "manifest_path": str(payload["manifest_path"]),
    }


def _write_ffprobe(video_path: Path) -> Path:
    ffprobe_output_path = video_path.parent / "ffprobe.json"
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, cwd=REPO_ROOT
    )
    ffprobe_output_path.write_text(result.stdout, encoding="utf-8")
    return ffprobe_output_path


def _run_gemini_review(*, video_path: Path) -> GeminiReview:
    review_dir = video_path.parent / "gemini-review"
    command = [
        "uv",
        "run",
        "python",
        "scripts/gemini_describe_video.py",
        str(video_path),
        "--save-dir",
        str(review_dir),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, cwd=REPO_ROOT
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Gemini review script returned a non-object JSON payload")

    expected_keys = (
        "visual_summary",
        "primary_subject",
        "scene",
        "style",
        "camera",
        "audio_present",
        "audio_type",
        "audio_summary",
        "confidence",
        "verdict",
    )
    normalized: dict[str, str] = {}
    for key in expected_keys:
        value = payload.get(key)
        if not isinstance(value, str):
            raise ValueError(f"Gemini review field {key!r} must be a string")
        normalized[key] = value
    return GeminiReview(
        visual_summary=normalized["visual_summary"],
        primary_subject=normalized["primary_subject"],
        scene=normalized["scene"],
        style=normalized["style"],
        camera=normalized["camera"],
        audio_present=normalized["audio_present"],
        audio_type=normalized["audio_type"],
        audio_summary=normalized["audio_summary"],
        confidence=normalized["confidence"],
        verdict=normalized["verdict"],
    )


def _review_matches_scene(
    *,
    parsed_review: GeminiReview,
    scene: ShowcaseScene,
    strict: bool,
) -> bool:
    subject = parsed_review["primary_subject"].lower()
    scene_text = parsed_review["scene"].lower()
    visual_summary = parsed_review["visual_summary"].lower()
    audio_type = parsed_review["audio_type"].lower()
    verdict = parsed_review["verdict"].lower()
    if strict and parsed_review["confidence"].lower() == "low":
        return False
    if verdict != "match":
        return False
    if not _contains_any(subject, scene.expected_subject_terms) and not _contains_any(
        visual_summary, scene.expected_subject_terms
    ):
        return False
    if not _contains_any(scene_text, scene.expected_scene_terms) and not _contains_any(
        visual_summary, scene.expected_scene_terms
    ):
        return False
    return audio_type in scene.expected_audio_types


def _contains_any(text: str, expected_terms: tuple[str, ...]) -> bool:
    return any(term.lower() in text for term in expected_terms)


def _unknown_review() -> GeminiReview:
    return GeminiReview(
        visual_summary="unknown",
        primary_subject="unknown",
        scene="unknown",
        style="unknown",
        camera="unknown",
        audio_present="unknown",
        audio_type="unknown",
        audio_summary="unknown",
        confidence="low",
        verdict="mismatch",
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

from mlxr.families.ltx.prompting import (
    PromptShapingOptions,
    shape_text_first_prompt_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "showcase-runs"
DEFAULT_SCENARIO_PACK = (
    REPO_ROOT / "tests" / "fixtures" / "ltx" / "prompt_scenarios.json"
)
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


class ScenarioPayload(TypedDict):
    scenario_id: str
    purpose: str
    prompt: str
    audio_intent: str
    music_allowed: bool
    conditioning_mode: str
    style_family: str
    expected_subject: str
    expected_scene: str
    validation_notes: str
    accepted_subject_terms: list[str]
    accepted_scene_terms: list[str]


class ScenarioPackPayload(TypedDict):
    scenario_pack_id: str
    scenarios: list[ScenarioPayload]


class SceneSummary(TypedDict):
    scene_id: str
    purpose: str
    prompt: str
    negative_prompt: str | None
    video_path: str
    manifest_path: str
    ffprobe_path: str
    gemini_review: GeminiReview
    review_ok: bool
    expected_subject: str
    expected_scene: str
    music_allowed: bool
    gemini_review_error: str | None


@dataclass(frozen=True, slots=True)
class ShowcaseScene:
    scene_id: str
    purpose: str
    prompt: str
    audio_intent: str
    music_allowed: bool
    conditioning_mode: str
    style_family: str
    expected_subject: str
    expected_scene: str
    validation_notes: str
    accepted_subject_terms: tuple[str, ...]
    accepted_scene_terms: tuple[str, ...]


def _required_str(payload: object, key: str) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Scenario entries must be JSON objects")
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Scenario field {key!r} must be a string")
    return value


def _required_bool(payload: object, key: str) -> bool:
    if not isinstance(payload, dict):
        raise ValueError("Scenario entries must be JSON objects")
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Scenario field {key!r} must be a boolean")
    return value


def _optional_terms(payload: object, key: str) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        raise ValueError("Scenario entries must be JSON objects")
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Scenario field {key!r} must be a list of strings")
    return tuple(value)


def _optional_str(payload: object, key: str, *, default: str) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Scenario entries must be JSON objects")
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"Scenario field {key!r} must be a string when provided")
    return value


def _load_scenarios(scenario_pack_path: Path) -> tuple[ShowcaseScene, ...]:
    payload = json.loads(scenario_pack_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scenario pack must be a JSON object")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ValueError("Scenario pack must contain a 'scenarios' list")
    return tuple(
        ShowcaseScene(
            scene_id=_required_str(scenario, "scenario_id"),
            purpose=_required_str(scenario, "purpose"),
            prompt=_required_str(scenario, "prompt"),
            audio_intent=_required_str(scenario, "audio_intent"),
            music_allowed=_required_bool(scenario, "music_allowed"),
            conditioning_mode=_optional_str(
                scenario, "conditioning_mode", default="text_first"
            ),
            style_family=_required_str(scenario, "style_family"),
            expected_subject=_required_str(scenario, "expected_subject"),
            expected_scene=_required_str(scenario, "expected_scene"),
            validation_notes=_required_str(scenario, "validation_notes"),
            accepted_subject_terms=_optional_terms(scenario, "accepted_subject_terms"),
            accepted_scene_terms=_optional_terms(scenario, "accepted_scene_terms"),
        )
        for scenario in raw_scenarios
    )


SHOWCASE_SCENES = _load_scenarios(DEFAULT_SCENARIO_PACK)


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
        "--scenario-pack",
        type=Path,
        default=DEFAULT_SCENARIO_PACK,
        help="JSON scenario pack to use for showcase generation.",
    )
    parser.add_argument(
        "--scene",
        action="append",
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
    scenes = _load_scenarios(args.scenario_pack.expanduser().resolve())
    selected = _selected_scenes(scenes, args.scene)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary: list[SceneSummary] = []
    for scene in selected:
        prompt_options = _prompt_options_for_run(
            scene,
            num_frames=int(args.num_frames),
            fps=int(args.fps),
        )
        resolved_prompts = shape_text_first_prompt_bundle(
            scene.prompt, options=prompt_options
        )
        prompt = resolved_prompts.prompt
        run_name = f"showcase-{scene.scene_id}"
        run_result = _run_smoke(
            artifact_root=args.artifact_root.expanduser().resolve(),
            output_root=output_root,
            run_name=run_name,
            prompt=prompt,
            negative_prompt=resolved_prompts.negative_prompt,
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
            "purpose": scene.purpose,
            "prompt": prompt,
            "negative_prompt": resolved_prompts.negative_prompt,
            "video_path": run_result["video_path"],
            "manifest_path": run_result["manifest_path"],
            "ffprobe_path": str(ffprobe_path),
            "gemini_review": parsed_review or _unknown_review(),
            "review_ok": review_ok,
            "expected_subject": scene.expected_subject,
            "expected_scene": scene.expected_scene,
            "music_allowed": scene.music_allowed,
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


def _prompt_options_for_run(
    scene: ShowcaseScene, *, num_frames: int, fps: int
) -> PromptShapingOptions:
    audio_prompt: str | None = scene.audio_intent
    if scene.conditioning_mode != "text_first":
        audio_prompt = None
    return PromptShapingOptions(
        audio_prompt=audio_prompt,
        natural_audio=not scene.music_allowed,
        no_music=not scene.music_allowed,
        style_family=scene.style_family,
        duration_seconds=(num_frames - 1) / fps,
        orientation="landscape",
    )


def _selected_scenes(
    scenes: tuple[ShowcaseScene, ...], scene_ids: list[str] | None
) -> tuple[ShowcaseScene, ...]:
    if not scene_ids:
        return scenes
    allowed = set(scene_ids)
    unknown = sorted(allowed.difference(scene.scene_id for scene in scenes))
    if unknown:
        raise ValueError(f"Unknown scenario ids: {', '.join(unknown)}")
    return tuple(scene for scene in scenes if scene.scene_id in allowed)


def _run_smoke(
    *,
    artifact_root: Path,
    output_root: Path,
    run_name: str,
    prompt: str,
    negative_prompt: str | None,
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
    if negative_prompt is not None:
        command.extend(["--negative-prompt", negative_prompt])
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
        command, check=False, capture_output=True, text=True, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "Gemini review command failed without output"
        raise RuntimeError(detail)
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
    subject_terms = scene.accepted_subject_terms or _keyword_terms(
        scene.expected_subject
    )
    scene_terms = scene.accepted_scene_terms or _keyword_terms(scene.expected_scene)
    if not _contains_any(subject, subject_terms) and not _contains_any(
        visual_summary, subject_terms
    ):
        return False
    if not _contains_any(scene_text, scene_terms) and not _contains_any(
        visual_summary, scene_terms
    ):
        return False
    if parsed_review["audio_present"].lower() != "true":
        return False
    if not scene.music_allowed and audio_type == "music":
        return False
    return audio_type != "unknown"


def _contains_any(text: str, expected_terms: tuple[str, ...]) -> bool:
    return any(term.lower() in text for term in expected_terms)


def _keyword_terms(text: str) -> tuple[str, ...]:
    stopwords = {
        "a",
        "an",
        "and",
        "at",
        "for",
        "in",
        "of",
        "on",
        "over",
        "the",
        "to",
        "with",
    }
    normalized = "".join(
        character.lower() if character.isalnum() else " " for character in text
    )
    terms: list[str] = []
    for part in normalized.split():
        if len(part) < 3 or part in stopwords:
            continue
        if part not in terms:
            terms.append(part)
    return tuple(terms)


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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MODEL = "gemini-3.1-pro-preview"
MAX_INLINE_VIDEO_BYTES = 20 * 1024 * 1024
REQUIRED_XML_FIELDS = (
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


@dataclass(frozen=True, slots=True)
class ParsedReview:
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


@dataclass(frozen=True, slots=True)
class ReviewArtifacts:
    staged_video_path: Path
    staged_audio_path: Path | None
    raw_response_path: Path
    parsed_review_path: Path
    prompt_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Describe a local video file with Gemini CLI in headless mode, force an "
            "XML review envelope, and emit a parsed machine-readable result."
        )
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to the local video file to describe.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model name to pass through to the CLI.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to write the staged copy, prompt, raw Gemini response, "
            "and parsed review JSON."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "xml", "text"),
        default="json",
        help="Wrapper output format printed to stdout after parsing.",
    )
    parser.add_argument(
        "--extra-instruction",
        default="",
        help="Optional extra instruction appended to the built-in review prompt.",
    )
    parser.add_argument(
        "--keep-staged-copy",
        action="store_true",
        help="Preserve the staged non-ignored review copy even when --save-dir is not set.",
    )
    parser.add_argument(
        "--extract-audio-review-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Extract a mono WAV review track alongside the staged video and ask Gemini "
            "to use both in the same review pass."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if Gemini returns a low-confidence review. Use this when the review "
            "is part of a promotion or acceptance gate."
        ),
    )
    return parser


def _xml_schema_prompt() -> str:
    return (
        "Return exactly one XML block and nothing else. "
        "Do not include markdown fences, prose before the XML, chain-of-thought, or tool logs. "
        "Use this exact structure: "
        "<review>"
        "<visual_summary>...</visual_summary>"
        "<primary_subject>...</primary_subject>"
        "<scene>...</scene>"
        "<style>...</style>"
        "<camera>...</camera>"
        "<audio_present>true|false|unknown</audio_present>"
        "<audio_type>music|ambience|barking|speech|other|unknown</audio_type>"
        "<audio_summary>...</audio_summary>"
        "<confidence>high|medium|low</confidence>"
        "<verdict>match|partial|mismatch</verdict>"
        "</review>."
    )


def _build_prompt(
    staged_video_path: Path,
    staged_audio_path: Path | None,
    extra_instruction: str,
) -> str:
    instruction = (
        "Review the attached local media files. "
        "Focus on the visible scene, subject identity, motion, style, and camera feel. "
        "In the same review, inspect the audio and say whether audio is present and whether it sounds like music, ambience, barking, speech, or something else. "
        "If you are unsure, say that plainly in the relevant fields rather than guessing. "
        f"Use the attached file @{staged_video_path.name} as the source of truth for the visuals. "
        f"{_xml_schema_prompt()}"
    )
    if staged_audio_path is not None:
        instruction = (
            f"{instruction} Use the attached file @{staged_audio_path.name} as the "
            "source of truth for the audio review."
        )
    else:
        instruction = (
            f"{instruction} No separate extracted audio file is attached, so assess "
            "audio only from the video file if possible."
        )
    if extra_instruction.strip():
        instruction = f"{instruction} {extra_instruction.strip()}"
    return instruction


def _stage_video(video_path: Path, save_dir: Path | None) -> tuple[Path, Path, bool]:
    if video_path.stat().st_size > MAX_INLINE_VIDEO_BYTES:
        raise RuntimeError(
            "Video exceeds Gemini CLI inline review limit of 20 MB; downscale or trim the clip before review"
        )

    if save_dir is not None:
        review_dir = save_dir.expanduser().resolve()
        review_dir.mkdir(parents=True, exist_ok=True)
        keep_dir = True
    else:
        review_dir = Path(
            tempfile.mkdtemp(prefix="mlxr-gemini-review-", dir="/tmp")
        ).resolve()
        keep_dir = False

    staged_video_path = review_dir / video_path.name
    shutil.copy2(video_path, staged_video_path)
    return review_dir, staged_video_path, keep_dir


def _video_has_audio_stream(video_path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _extract_audio_review_track(video_path: Path, review_dir: Path) -> Path | None:
    if not _video_has_audio_stream(video_path):
        return None
    staged_audio_path = review_dir / f"{video_path.stem}-audio-review.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(staged_audio_path),
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return staged_audio_path


def _extract_review_xml(raw_output: str) -> str:
    matches = [
        match.group(0)
        for match in re.finditer(r"<review>.*?</review>", raw_output, flags=re.DOTALL)
    ]
    if not matches:
        raise RuntimeError("Gemini response did not contain a <review> XML block")
    if len(matches) != 1:
        raise RuntimeError(
            "Gemini response must contain exactly one <review> XML block"
        )
    return matches[0]


def _parse_review(xml_text: str) -> ParsedReview:
    root = ET.fromstring(xml_text)
    if root.tag != "review":
        raise RuntimeError("Gemini XML root must be <review>")

    values: dict[str, str] = {}
    for field in REQUIRED_XML_FIELDS:
        element = root.find(field)
        if element is None or element.text is None or not element.text.strip():
            raise RuntimeError(f"Gemini XML is missing required <{field}> content")
        values[field] = element.text.strip()

    if values["audio_present"] not in {"true", "false", "unknown"}:
        raise RuntimeError("Gemini XML audio_present must be true, false, or unknown")
    if values["audio_type"] not in {
        "music",
        "ambience",
        "barking",
        "speech",
        "other",
        "unknown",
    }:
        raise RuntimeError(
            "Gemini XML audio_type must be music, ambience, barking, speech, other, or unknown"
        )
    if values["confidence"] not in {"high", "medium", "low"}:
        raise RuntimeError("Gemini XML confidence must be high, medium, or low")
    if values["verdict"] not in {"match", "partial", "mismatch"}:
        raise RuntimeError("Gemini XML verdict must be match, partial, or mismatch")

    return ParsedReview(**values)


def _enforce_strict_review(parsed_review: ParsedReview) -> None:
    if parsed_review.confidence == "low":
        raise RuntimeError("Gemini review confidence is too low for strict mode")
    if parsed_review.verdict != "match":
        raise RuntimeError(
            "Gemini review verdict must be match when strict mode is enabled"
        )


def _write_artifacts(
    *,
    review_dir: Path,
    prompt: str,
    raw_output: str,
    parsed_review: ParsedReview,
    staged_video_path: Path,
    staged_audio_path: Path | None,
) -> ReviewArtifacts:
    raw_response_path = review_dir / "gemini-review-raw.txt"
    parsed_review_path = review_dir / "gemini-review.json"
    prompt_path = review_dir / "gemini-review-prompt.txt"

    prompt_path.write_text(prompt, encoding="utf-8")
    raw_response_path.write_text(raw_output, encoding="utf-8")
    parsed_review_path.write_text(
        json.dumps(asdict(parsed_review), indent=2), encoding="utf-8"
    )
    return ReviewArtifacts(
        staged_video_path=staged_video_path,
        staged_audio_path=staged_audio_path,
        raw_response_path=raw_response_path,
        parsed_review_path=parsed_review_path,
        prompt_path=prompt_path,
    )


def _print_result(
    *,
    output_format: str,
    parsed_review: ParsedReview,
    xml_text: str,
    artifacts: ReviewArtifacts | None,
) -> None:
    if output_format == "xml":
        sys.stdout.write(xml_text)
        if not xml_text.endswith("\n"):
            sys.stdout.write("\n")
        return
    if output_format == "text":
        sys.stdout.write(
            (
                f"Subject: {parsed_review.primary_subject}\n"
                f"Scene: {parsed_review.scene}\n"
                f"Style: {parsed_review.style}\n"
                f"Camera: {parsed_review.camera}\n"
                f"Audio present: {parsed_review.audio_present}\n"
                f"Audio type: {parsed_review.audio_type}\n"
                f"Verdict: {parsed_review.verdict}\n"
            )
        )
        return

    payload = {
        **asdict(parsed_review),
        "artifacts": None,
    }
    if artifacts is not None:
        payload["artifacts"] = {
            "staged_video_path": str(artifacts.staged_video_path),
            "staged_audio_path": (
                str(artifacts.staged_audio_path)
                if artifacts.staged_audio_path is not None
                else None
            ),
            "raw_response_path": str(artifacts.raw_response_path),
            "parsed_review_path": str(artifacts.parsed_review_path),
            "prompt_path": str(artifacts.prompt_path),
        }
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")


def main() -> int:
    args = _build_parser().parse_args()
    video_path = args.video_path.expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video file does not exist: {video_path}")
    if not video_path.is_file():
        raise SystemExit(f"Video path is not a file: {video_path}")

    review_dir, staged_video_path, keep_dir = _stage_video(video_path, args.save_dir)
    staged_audio_path: Path | None = None
    try:
        staged_audio_path = (
            _extract_audio_review_track(video_path, review_dir)
            if args.extract_audio_review_track
            else None
        )
        prompt = _build_prompt(
            staged_video_path, staged_audio_path, args.extra_instruction
        )
        command = [
            "gemini",
            "-m",
            args.model,
            "-p",
            prompt,
            "--include-directories",
            str(review_dir),
            "--output-format",
            "text",
            "-y",
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=review_dir,
        )
        if result.returncode != 0:
            if result.stdout:
                sys.stdout.write(result.stdout)
            if result.stderr:
                sys.stderr.write(result.stderr)
            return result.returncode

        xml_text = _extract_review_xml(result.stdout)
        parsed_review = _parse_review(xml_text)
        persist_artifacts = args.save_dir is not None or args.keep_staged_copy
        artifacts: ReviewArtifacts | None = None
        if persist_artifacts:
            artifacts = _write_artifacts(
                review_dir=review_dir,
                prompt=prompt,
                raw_output=result.stdout,
                parsed_review=parsed_review,
                staged_video_path=staged_video_path,
                staged_audio_path=staged_audio_path,
            )
        if args.strict:
            _enforce_strict_review(parsed_review)
        _print_result(
            output_format=args.output_format,
            parsed_review=parsed_review,
            xml_text=xml_text,
            artifacts=artifacts,
        )
        return 0
    except subprocess.CalledProcessError as error:
        sys.stderr.write(
            "gemini_describe_video.py: media staging failed "
            f"for command {error.cmd!r} with exit code {error.returncode}\n"
        )
        if error.stderr:
            sys.stderr.write(error.stderr)
        return 2
    except OSError as error:
        sys.stderr.write(
            "gemini_describe_video.py: required external command failed to start: "
            f"{error}\n"
        )
        return 2
    except RuntimeError as error:
        sys.stderr.write(f"gemini_describe_video.py: {error}\n")
        return 2
    finally:
        if args.save_dir is None and not args.keep_staged_copy:
            shutil.rmtree(review_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

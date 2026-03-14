#!/usr/bin/env python3
"""Review conditioned videos against prompt and reference-image expectations."""

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
REFERENCE_ROLES = ("start", "end", "anchor", "keyframe")
REQUIRED_XML_FIELDS = (
    "visual_summary",
    "opening_frame_match",
    "closing_frame_match",
    "subject_consistency",
    "motion_coherence",
    "prompt_alignment",
    "aspect_ratio_assessment",
    "stretch_or_squash",
    "issues",
    "confidence",
    "verdict",
)


@dataclass(frozen=True, slots=True)
class MediaProbe:
    width: int
    height: int
    duration_seconds: float | None

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


@dataclass(frozen=True, slots=True)
class StagedReference:
    role: str
    original_path: Path
    staged_path: Path
    probe: MediaProbe


@dataclass(frozen=True, slots=True)
class ParsedReview:
    visual_summary: str
    opening_frame_match: str
    closing_frame_match: str
    subject_consistency: str
    motion_coherence: str
    prompt_alignment: str
    aspect_ratio_assessment: str
    stretch_or_squash: str
    issues: str
    confidence: str
    verdict: str


@dataclass(frozen=True, slots=True)
class ReviewArtifacts:
    staged_video_path: Path
    raw_response_path: Path
    parsed_review_path: Path
    prompt_path: Path
    staged_references: tuple[StagedReference, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Review a local video clip against prompt semantics and optional "
            "conditioning images using Gemini CLI in headless mode."
        )
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to the local video file to review.",
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
        "--prompt-text",
        default="",
        help="Optional generation prompt to compare against the produced video.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional path to a text file containing the generation prompt.",
    )
    parser.add_argument(
        "--reference-image",
        action="append",
        nargs=2,
        metavar=("PATH", "ROLE"),
        default=[],
        help=(
            "Optional conditioning reference image plus role. Role must be one of "
            f"{', '.join(REFERENCE_ROLES)}. Repeat for multiple references."
        ),
    )
    parser.add_argument(
        "--extra-instruction",
        default="",
        help="Optional extra instruction appended to the built-in review prompt.",
    )
    parser.add_argument(
        "--keep-staged-copy",
        action="store_true",
        help="Preserve the staged review copy even when --save-dir is not set.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if Gemini returns a low-confidence or non-match review. Use this "
            "when the review is part of a promotion or acceptance gate."
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
        "<opening_frame_match>match|partial|mismatch|unknown</opening_frame_match>"
        "<closing_frame_match>match|partial|mismatch|unknown</closing_frame_match>"
        "<subject_consistency>high|medium|low</subject_consistency>"
        "<motion_coherence>high|medium|low</motion_coherence>"
        "<prompt_alignment>match|partial|mismatch|unknown</prompt_alignment>"
        "<aspect_ratio_assessment>...</aspect_ratio_assessment>"
        "<stretch_or_squash>none|minor|major|unknown</stretch_or_squash>"
        "<issues>...</issues>"
        "<confidence>high|medium|low</confidence>"
        "<verdict>match|partial|mismatch</verdict>"
        "</review>."
    )


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
            tempfile.mkdtemp(prefix="mlxr-gemini-conditioned-review-", dir="/tmp")
        ).resolve()
        keep_dir = False

    staged_video_path = review_dir / video_path.name
    shutil.copy2(video_path, staged_video_path)
    return review_dir, staged_video_path, keep_dir


def _probe_media(path: Path) -> MediaProbe:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(f"ffprobe did not return a video stream for {path}")
    stream0 = streams[0]
    width = stream0.get("width")
    height = stream0.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise RuntimeError(f"ffprobe did not return width/height for {path}")
    duration_value = payload.get("format", {}).get("duration")
    duration_seconds: float | None = None
    if isinstance(duration_value, str) and duration_value.strip():
        try:
            duration_seconds = float(duration_value)
        except ValueError:
            duration_seconds = None
    return MediaProbe(width=width, height=height, duration_seconds=duration_seconds)


def _format_probe_summary(probe: MediaProbe) -> str:
    summary = f"{probe.width}x{probe.height} (aspect {probe.aspect_ratio:.3f})"
    if probe.duration_seconds is not None:
        summary = f"{summary}, duration {probe.duration_seconds:.3f}s"
    return summary


def _stage_references(
    review_dir: Path,
    reference_args: list[list[str]],
) -> tuple[StagedReference, ...]:
    staged: list[StagedReference] = []
    for path_text, role in reference_args:
        normalized_role = role.strip().lower()
        if normalized_role not in REFERENCE_ROLES:
            raise RuntimeError(
                f"Reference role must be one of {', '.join(REFERENCE_ROLES)}"
            )
        source_path = Path(path_text).expanduser().resolve()
        if not source_path.exists():
            raise RuntimeError(f"Reference image does not exist: {source_path}")
        if not source_path.is_file():
            raise RuntimeError(f"Reference image path is not a file: {source_path}")
        staged_name = f"{normalized_role}-{source_path.name}"
        staged_path = review_dir / staged_name
        shutil.copy2(source_path, staged_path)
        staged.append(
            StagedReference(
                role=normalized_role,
                original_path=source_path,
                staged_path=staged_path,
                probe=_probe_media(source_path),
            )
        )
    return tuple(staged)


def _read_prompt_text(*, prompt_text: str, prompt_file: Path | None) -> str:
    parts: list[str] = []
    if prompt_text.strip():
        parts.append(prompt_text.strip())
    if prompt_file is not None:
        file_text = (
            prompt_file.expanduser().resolve().read_text(encoding="utf-8").strip()
        )
        if file_text:
            parts.append(file_text)
    return "\n\n".join(parts)


def _build_prompt(
    *,
    staged_video_path: Path,
    video_probe: MediaProbe,
    staged_references: tuple[StagedReference, ...],
    prompt_text: str,
    extra_instruction: str,
) -> str:
    reference_lines = []
    for ref in staged_references:
        reference_lines.append(
            f"- {ref.role} reference @{ref.staged_path.name}: {_format_probe_summary(ref.probe)}"
        )
    reference_block = "\n".join(reference_lines) if reference_lines else "- none"
    prompt_block = prompt_text if prompt_text else "No prompt text was provided."

    instruction = (
        "Review the attached local video and optional conditioning images. "
        "Decide whether the video is temporally coherent, whether the subject stays "
        "consistent enough for a single shot, whether the opening and closing frames "
        "match the relevant reference images, and whether the video semantically follows "
        "the generation prompt. Pay special attention to aspect-ratio mismatch: if the "
        "subject looks stretched, squashed, or compositionally distorted relative to the "
        "reference image, say so plainly. "
        f"Use the attached file @{staged_video_path.name} as the source of truth for the video. "
        f"{_xml_schema_prompt()} "
        f"Video facts: @{staged_video_path.name} is {_format_probe_summary(video_probe)}. "
        f"Reference facts:\n{reference_block}\n"
        f"Prompt to compare against:\n{prompt_block}"
    )
    if staged_references:
        reference_names = ", ".join(
            f"@{ref.staged_path.name} ({ref.role})" for ref in staged_references
        )
        instruction = (
            f"{instruction}\nUse these attached reference images as the source of truth "
            f"for frame matching: {reference_names}."
        )
    else:
        instruction = (
            f"{instruction}\nNo reference images are attached, so use unknown for "
            "frame-match fields unless the video itself makes the answer obvious."
        )
    if extra_instruction.strip():
        instruction = f"{instruction}\n{extra_instruction.strip()}"
    return instruction


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

    frame_values = {"match", "partial", "mismatch", "unknown"}
    if values["opening_frame_match"] not in frame_values:
        raise RuntimeError(
            "Gemini XML opening_frame_match must be match, partial, mismatch, or unknown"
        )
    if values["closing_frame_match"] not in frame_values:
        raise RuntimeError(
            "Gemini XML closing_frame_match must be match, partial, mismatch, or unknown"
        )
    if values["subject_consistency"] not in {"high", "medium", "low"}:
        raise RuntimeError(
            "Gemini XML subject_consistency must be high, medium, or low"
        )
    if values["motion_coherence"] not in {"high", "medium", "low"}:
        raise RuntimeError("Gemini XML motion_coherence must be high, medium, or low")
    if values["prompt_alignment"] not in {"match", "partial", "mismatch", "unknown"}:
        raise RuntimeError(
            "Gemini XML prompt_alignment must be match, partial, mismatch, or unknown"
        )
    if values["stretch_or_squash"] not in {"none", "minor", "major", "unknown"}:
        raise RuntimeError(
            "Gemini XML stretch_or_squash must be none, minor, major, or unknown"
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
    staged_references: tuple[StagedReference, ...],
) -> ReviewArtifacts:
    raw_response_path = review_dir / "gemini-conditioned-review-raw.txt"
    parsed_review_path = review_dir / "gemini-conditioned-review.json"
    prompt_path = review_dir / "gemini-conditioned-review-prompt.txt"

    prompt_path.write_text(prompt, encoding="utf-8")
    raw_response_path.write_text(raw_output, encoding="utf-8")
    payload = asdict(parsed_review)
    payload["references"] = [
        {
            "role": ref.role,
            "original_path": str(ref.original_path),
            "staged_path": str(ref.staged_path),
            "width": ref.probe.width,
            "height": ref.probe.height,
            "aspect_ratio": round(ref.probe.aspect_ratio, 6),
        }
        for ref in staged_references
    ]
    parsed_review_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ReviewArtifacts(
        staged_video_path=staged_video_path,
        raw_response_path=raw_response_path,
        parsed_review_path=parsed_review_path,
        prompt_path=prompt_path,
        staged_references=staged_references,
    )


def _print_result(
    *,
    output_format: str,
    parsed_review: ParsedReview,
    xml_text: str,
    artifacts: ReviewArtifacts | None,
    video_probe: MediaProbe,
) -> None:
    if output_format == "xml":
        sys.stdout.write(xml_text)
        if not xml_text.endswith("\n"):
            sys.stdout.write("\n")
        return
    if output_format == "text":
        sys.stdout.write(
            (
                f"Opening frame match: {parsed_review.opening_frame_match}\n"
                f"Closing frame match: {parsed_review.closing_frame_match}\n"
                f"Subject consistency: {parsed_review.subject_consistency}\n"
                f"Motion coherence: {parsed_review.motion_coherence}\n"
                f"Prompt alignment: {parsed_review.prompt_alignment}\n"
                f"Stretch or squash: {parsed_review.stretch_or_squash}\n"
                f"Verdict: {parsed_review.verdict}\n"
            )
        )
        return

    payload = {
        **asdict(parsed_review),
        "video": {
            "width": video_probe.width,
            "height": video_probe.height,
            "aspect_ratio": round(video_probe.aspect_ratio, 6),
            "duration_seconds": video_probe.duration_seconds,
        },
        "artifacts": None,
    }
    if artifacts is not None:
        payload["artifacts"] = {
            "staged_video_path": str(artifacts.staged_video_path),
            "staged_references": [
                {
                    "role": ref.role,
                    "path": str(ref.staged_path),
                }
                for ref in artifacts.staged_references
            ],
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

    review_dir, staged_video_path, _keep_dir = _stage_video(video_path, args.save_dir)
    try:
        video_probe = _probe_media(video_path)
        staged_references = _stage_references(review_dir, args.reference_image)
        prompt_text = _read_prompt_text(
            prompt_text=args.prompt_text,
            prompt_file=args.prompt_file,
        )
        prompt = _build_prompt(
            staged_video_path=staged_video_path,
            video_probe=video_probe,
            staged_references=staged_references,
            prompt_text=prompt_text,
            extra_instruction=args.extra_instruction,
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
                staged_references=staged_references,
            )
        if args.strict:
            _enforce_strict_review(parsed_review)
        _print_result(
            output_format=args.output_format,
            parsed_review=parsed_review,
            xml_text=xml_text,
            artifacts=artifacts,
            video_probe=video_probe,
        )
        return 0
    except subprocess.CalledProcessError as error:
        sys.stderr.write(
            "gemini_review_conditioned_video.py: media staging failed "
            f"for command {error.cmd!r} with exit code {error.returncode}\n"
        )
        if error.stderr:
            sys.stderr.write(error.stderr)
        return 2
    except OSError as error:
        sys.stderr.write(
            "gemini_review_conditioned_video.py: required external command failed to start: "
            f"{error}\n"
        )
        return 2
    except RuntimeError as error:
        sys.stderr.write(f"gemini_review_conditioned_video.py: {error}\n")
        return 2
    finally:
        if args.save_dir is None and not args.keep_staged_copy:
            shutil.rmtree(review_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Review generated images with Gemini and save strict XML-derived receipts."""

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
MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024
REFERENCE_ROLES = ("source", "style", "character", "composition", "other")
REQUIRED_XML_FIELDS = (
    "visual_summary",
    "primary_subject",
    "scene",
    "style",
    "text_rendering",
    "prompt_alignment",
    "reference_alignment",
    "issues",
    "confidence",
    "verdict",
)


@dataclass(frozen=True, slots=True)
class StagedReference:
    role: str
    original_path: Path
    staged_path: Path


@dataclass(frozen=True, slots=True)
class ParsedReview:
    visual_summary: str
    primary_subject: str
    scene: str
    style: str
    text_rendering: str
    prompt_alignment: str
    reference_alignment: str
    issues: str
    confidence: str
    verdict: str


@dataclass(frozen=True, slots=True)
class ReviewArtifacts:
    staged_image_path: Path
    raw_response_path: Path
    parsed_review_path: Path
    prompt_path: Path
    staged_references: tuple[StagedReference, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Review a local image against prompt semantics and optional reference "
            "images using Gemini CLI in headless mode."
        )
    )
    parser.add_argument("image_path", type=Path, help="Path to the generated image.")
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
            "Optional directory to write the staged copy, prompt, raw Gemini "
            "response, and parsed review JSON."
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
        help="Optional generation or editing prompt to compare against the image.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional path to a text file containing the prompt.",
    )
    parser.add_argument(
        "--reference-image",
        action="append",
        nargs=2,
        metavar=("PATH", "ROLE"),
        default=[],
        help=(
            "Optional reference image plus role. Role must be one of "
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
        help="Fail if Gemini returns a low-confidence or non-match review.",
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
        "<text_rendering>strong|adequate|weak|none|unknown</text_rendering>"
        "<prompt_alignment>match|partial|mismatch|unknown</prompt_alignment>"
        "<reference_alignment>match|partial|mismatch|unknown</reference_alignment>"
        "<issues>...</issues>"
        "<confidence>high|medium|low</confidence>"
        "<verdict>match|partial|mismatch</verdict>"
        "</review>."
    )


def _stage_image(image_path: Path, save_dir: Path | None) -> tuple[Path, Path, bool]:
    if image_path.stat().st_size > MAX_INLINE_IMAGE_BYTES:
        raise RuntimeError(
            "Image exceeds Gemini CLI inline review limit of 20 MB; resize or recompress before review"
        )
    if save_dir is not None:
        review_dir = save_dir.expanduser().resolve()
        review_dir.mkdir(parents=True, exist_ok=True)
        keep_dir = True
    else:
        review_dir = Path(
            tempfile.mkdtemp(prefix="mlxr-gemini-image-review-", dir="/tmp")
        ).resolve()
        keep_dir = False
    staged_image_path = review_dir / image_path.name
    shutil.copy2(image_path, staged_image_path)
    return review_dir, staged_image_path, keep_dir


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
        if source_path.stat().st_size > MAX_INLINE_IMAGE_BYTES:
            raise RuntimeError(
                f"Reference image exceeds Gemini CLI inline limit: {source_path}"
            )
        staged_path = review_dir / f"{normalized_role}-{source_path.name}"
        shutil.copy2(source_path, staged_path)
        staged.append(
            StagedReference(
                role=normalized_role,
                original_path=source_path,
                staged_path=staged_path,
            )
        )
    return tuple(staged)


def _build_prompt(
    staged_image_path: Path,
    prompt_text: str,
    staged_references: tuple[StagedReference, ...],
    extra_instruction: str,
) -> str:
    prompt_parts = [
        "Review the attached generated image.",
        f"Use @{staged_image_path.name} as the source of truth for the produced output.",
        "Focus on semantic correctness, subject identity, composition, realism or stylization, and whether the image looks intentionally made rather than obviously broken.",
        "If the prompt asks for visible text, judge whether the text rendering looks strong, adequate, weak, or absent.",
    ]
    if prompt_text.strip():
        prompt_parts.append(
            f"The original generation or edit prompt was: {prompt_text.strip()}"
        )
    if staged_references:
        reference_lines = ["Use the following reference images when judging alignment:"]
        for reference in staged_references:
            reference_lines.append(
                f"- @{reference.staged_path.name} ({reference.role})"
            )
        prompt_parts.append(" ".join(reference_lines))
    else:
        prompt_parts.append(
            "No reference images are attached, so set reference_alignment to unknown unless a reference relationship is obvious from the prompt itself."
        )
    prompt_parts.append(
        "If you are unsure, say that plainly in the relevant fields rather than guessing."
    )
    prompt_parts.append(_xml_schema_prompt())
    if extra_instruction.strip():
        prompt_parts.append(extra_instruction.strip())
    return " ".join(prompt_parts)


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

    if values["text_rendering"] not in {
        "strong",
        "adequate",
        "weak",
        "none",
        "unknown",
    }:
        raise RuntimeError(
            "Gemini <text_rendering> must be strong, adequate, weak, none, or unknown"
        )
    for field in ("prompt_alignment", "reference_alignment"):
        if values[field] not in {"match", "partial", "mismatch", "unknown"}:
            raise RuntimeError(
                f"Gemini <{field}> must be match, partial, mismatch, or unknown"
            )
    if values["confidence"] not in {"high", "medium", "low"}:
        raise RuntimeError("Gemini <confidence> must be high, medium, or low")
    if values["verdict"] not in {"match", "partial", "mismatch"}:
        raise RuntimeError("Gemini <verdict> must be match, partial, or mismatch")
    return ParsedReview(**values)


def _load_prompt_text(args: argparse.Namespace) -> str:
    prompt_parts: list[str] = []
    if args.prompt_text:
        prompt_parts.append(args.prompt_text)
    if args.prompt_file is not None:
        prompt_parts.append(args.prompt_file.expanduser().read_text(encoding="utf-8"))
    return "\n".join(part for part in prompt_parts if part.strip()).strip()


def _run_gemini(
    *,
    review_dir: Path,
    prompt: str,
    model: str,
) -> str:
    command = [
        "gemini",
        "-m",
        model,
        "-p",
        prompt,
        "--include-directories",
        str(review_dir),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=review_dir,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Gemini CLI returned a non-zero exit code"
        raise RuntimeError(stderr)
    return result.stdout


def _write_artifacts(
    *,
    review_dir: Path,
    raw_output: str,
    parsed_review: ParsedReview,
    prompt: str,
    staged_image_path: Path,
    staged_references: tuple[StagedReference, ...],
) -> ReviewArtifacts:
    raw_response_path = review_dir / "gemini-image-review-raw.txt"
    parsed_review_path = review_dir / "gemini-image-review.json"
    prompt_path = review_dir / "gemini-image-review-prompt.txt"
    raw_response_path.write_text(raw_output, encoding="utf-8")
    parsed_review_path.write_text(
        json.dumps(asdict(parsed_review), indent=2) + "\n",
        encoding="utf-8",
    )
    prompt_path.write_text(prompt, encoding="utf-8")
    return ReviewArtifacts(
        staged_image_path=staged_image_path,
        raw_response_path=raw_response_path,
        parsed_review_path=parsed_review_path,
        prompt_path=prompt_path,
        staged_references=staged_references,
    )


def _enforce_strict_review(review: ParsedReview) -> None:
    if review.confidence != "high":
        raise RuntimeError("Gemini review confidence is too low for strict mode")
    if review.verdict != "match":
        raise RuntimeError("Gemini review verdict must be match in strict mode")


def main() -> int:
    args = _build_parser().parse_args()
    image_path = args.image_path.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise RuntimeError(f"Image path is not a file: {image_path}")

    review_dir: Path | None = None
    keep_dir = False
    try:
        review_dir, staged_image_path, keep_dir = _stage_image(
            image_path, args.save_dir
        )
        staged_references = _stage_references(review_dir, args.reference_image)
        prompt_text = _load_prompt_text(args)
        prompt = _build_prompt(
            staged_image_path=staged_image_path,
            prompt_text=prompt_text,
            staged_references=staged_references,
            extra_instruction=args.extra_instruction,
        )
        raw_output = _run_gemini(
            review_dir=review_dir,
            prompt=prompt,
            model=args.model,
        )
        review_xml = _extract_review_xml(raw_output)
        parsed_review = _parse_review(review_xml)
        if args.strict:
            _enforce_strict_review(parsed_review)
        _write_artifacts(
            review_dir=review_dir,
            raw_output=raw_output,
            parsed_review=parsed_review,
            prompt=prompt,
            staged_image_path=staged_image_path,
            staged_references=staged_references,
        )
        if args.output_format == "xml":
            sys.stdout.write(review_xml + "\n")
        elif args.output_format == "text":
            sys.stdout.write(json.dumps(asdict(parsed_review), indent=2) + "\n")
        else:
            sys.stdout.write(json.dumps(asdict(parsed_review)) + "\n")
        return 0
    except (
        FileNotFoundError,
        subprocess.SubprocessError,
        RuntimeError,
        ET.ParseError,
        ValueError,
    ) as error:
        sys.stderr.write(f"gemini_review_image.py: {error}\n")
        return 1
    finally:
        if review_dir is not None and not keep_dir and not args.keep_staged_copy:
            shutil.rmtree(review_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

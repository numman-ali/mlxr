from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch


def _load_script_module() -> types.ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "gemini_review_image.py"
    )
    spec = importlib.util.spec_from_file_location("gemini_review_image", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/gemini_review_image.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GeminiReviewImageScriptTests(unittest.TestCase):
    def test_build_prompt_mentions_image_prompt_and_references(self) -> None:
        module = _load_script_module()
        prompt = module._build_prompt(
            staged_image_path=Path("/tmp/review/output.png"),
            prompt_text="A glowing cat in a rainy alley",
            staged_references=(
                module.StagedReference(
                    role="source",
                    original_path=Path("/tmp/original.png"),
                    staged_path=Path("/tmp/review/source-original.png"),
                ),
            ),
            extra_instruction="Check whether the cat still looks like the source.",
        )
        self.assertIn("@output.png", prompt)
        self.assertIn("A glowing cat in a rainy alley", prompt)
        self.assertIn("@source-original.png", prompt)
        self.assertIn("Return exactly one XML block and nothing else.", prompt)

    def test_parse_review_rejects_missing_field(self) -> None:
        module = _load_script_module()
        with self.assertRaisesRegex(
            RuntimeError, "missing required <reference_alignment> content"
        ):
            module._parse_review(
                (
                    "<review>"
                    "<visual_summary>cat in alley</visual_summary>"
                    "<primary_subject>cat</primary_subject>"
                    "<scene>rainy alley</scene>"
                    "<style>cinematic</style>"
                    "<text_rendering>none</text_rendering>"
                    "<prompt_alignment>match</prompt_alignment>"
                    "<issues>none</issues>"
                    "<confidence>high</confidence>"
                    "<verdict>match</verdict>"
                    "</review>"
                )
            )

    def test_parse_review_rejects_invalid_enum(self) -> None:
        module = _load_script_module()
        with self.assertRaisesRegex(RuntimeError, "text_rendering"):
            module._parse_review(
                (
                    "<review>"
                    "<visual_summary>cat in alley</visual_summary>"
                    "<primary_subject>cat</primary_subject>"
                    "<scene>rainy alley</scene>"
                    "<style>cinematic</style>"
                    "<text_rendering>excellent</text_rendering>"
                    "<prompt_alignment>match</prompt_alignment>"
                    "<reference_alignment>unknown</reference_alignment>"
                    "<issues>none</issues>"
                    "<confidence>high</confidence>"
                    "<verdict>match</verdict>"
                    "</review>"
                )
            )

    def test_strict_review_rejects_partial_verdict(self) -> None:
        module = _load_script_module()
        review = module.ParsedReview(
            visual_summary="cat in alley",
            primary_subject="cat",
            scene="rainy alley",
            style="cinematic",
            text_rendering="none",
            prompt_alignment="partial",
            reference_alignment="unknown",
            issues="subject is blurry",
            confidence="high",
            verdict="partial",
        )
        with self.assertRaisesRegex(RuntimeError, "verdict must be match"):
            module._enforce_strict_review(review)

    def test_main_runs_gemini_and_writes_receipts(self) -> None:
        module = _load_script_module()
        xml_response = (
            "<review>"
            "<visual_summary>A glowing cat stands in a rainy neon alley.</visual_summary>"
            "<primary_subject>cat</primary_subject>"
            "<scene>rainy neon alley</scene>"
            "<style>cinematic cyberpunk</style>"
            "<text_rendering>none</text_rendering>"
            "<prompt_alignment>match</prompt_alignment>"
            "<reference_alignment>unknown</reference_alignment>"
            "<issues>minor softness in the background</issues>"
            "<confidence>high</confidence>"
            "<verdict>match</verdict>"
            "</review>"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            image_path = temp_root / "output.png"
            reference_path = temp_root / "source.png"
            save_dir = temp_root / "review"
            image_path.write_bytes(b"fake-image")
            reference_path.write_bytes(b"fake-reference")
            stdout = io.StringIO()
            command_log: list[list[str]] = []

            def fake_run(
                command: list[str],
                *,
                check: bool,
                capture_output: bool,
                text: bool,
                cwd: Path | None = None,
            ) -> CompletedProcess[str]:
                command_log.append(command)
                self.assertFalse(check)
                self.assertTrue(capture_output)
                self.assertTrue(text)
                self.assertEqual(cwd, save_dir.resolve())
                return CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=xml_response,
                    stderr="",
                )

            argv = [
                "gemini_review_image.py",
                str(image_path),
                "--save-dir",
                str(save_dir),
                "--prompt-text",
                "A glowing cat in a rainy alley",
                "--reference-image",
                str(reference_path),
                "source",
                "--strict",
            ]
            with (
                patch.object(module.subprocess, "run", side_effect=fake_run),
                patch.object(sys, "argv", argv),
                patch("sys.stdout", stdout),
            ):
                exit_code = module.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual([command[0] for command in command_log], ["gemini"])
            self.assertEqual(command_log[0][1:3], ["-m", module.DEFAULT_MODEL])
            self.assertIn("--include-directories", command_log[0])
            self.assertIn(str(save_dir.resolve()), command_log[0])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["primary_subject"], "cat")
            self.assertEqual(payload["verdict"], "match")

            prompt_path = save_dir / "gemini-image-review-prompt.txt"
            raw_path = save_dir / "gemini-image-review-raw.txt"
            parsed_path = save_dir / "gemini-image-review.json"
            self.assertTrue((save_dir / "output.png").exists())
            self.assertTrue((save_dir / "source-source.png").exists())
            self.assertTrue(prompt_path.exists())
            self.assertTrue(raw_path.exists())
            self.assertTrue(parsed_path.exists())
            self.assertEqual(
                json.loads(parsed_path.read_text(encoding="utf-8")),
                {
                    "visual_summary": "A glowing cat stands in a rainy neon alley.",
                    "primary_subject": "cat",
                    "scene": "rainy neon alley",
                    "style": "cinematic cyberpunk",
                    "text_rendering": "none",
                    "prompt_alignment": "match",
                    "reference_alignment": "unknown",
                    "issues": "minor softness in the background",
                    "confidence": "high",
                    "verdict": "match",
                },
            )
            self.assertIn("@output.png", prompt_path.read_text(encoding="utf-8"))
            self.assertIn("@source-source.png", prompt_path.read_text(encoding="utf-8"))

    def test_main_returns_nonzero_when_gemini_output_has_no_review_xml(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            image_path = temp_root / "output.png"
            save_dir = temp_root / "review"
            image_path.write_bytes(b"fake-image")
            stderr = io.StringIO()
            argv = [
                "gemini_review_image.py",
                str(image_path),
                "--save-dir",
                str(save_dir),
            ]
            with (
                patch.object(
                    module.subprocess,
                    "run",
                    return_value=CompletedProcess(
                        args=["gemini"], returncode=0, stdout="no xml here", stderr=""
                    ),
                ),
                patch.object(sys, "argv", argv),
                patch("sys.stderr", stderr),
            ):
                exit_code = module.main()
            self.assertEqual(exit_code, 1)
            self.assertIn("did not contain a <review> XML block", stderr.getvalue())

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
        Path(__file__).resolve().parents[1] / "scripts" / "gemini_describe_video.py"
    )
    spec = importlib.util.spec_from_file_location("gemini_describe_video", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/gemini_describe_video.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GeminiDescribeVideoScriptTests(unittest.TestCase):
    def test_build_prompt_requires_xml_only_response(self) -> None:
        module = _load_script_module()
        prompt = module._build_prompt(Path("/tmp/review/dog.mp4"), "")
        self.assertIn("Return exactly one XML block and nothing else.", prompt)
        self.assertIn("@dog.mp4", prompt)
        self.assertIn("<audio_present>true|false|unknown</audio_present>", prompt)

    def test_parse_review_requires_all_expected_fields(self) -> None:
        module = _load_script_module()
        with self.assertRaisesRegex(RuntimeError, "missing required <scene> content"):
            module._parse_review(
                (
                    "<review>"
                    "<visual_summary>dog in park</visual_summary>"
                    "<primary_subject>dog</primary_subject>"
                    "<style>natural</style>"
                    "<camera>tracking</camera>"
                    "<audio_present>true</audio_present>"
                    "<audio_type>barking</audio_type>"
                    "<audio_summary>happy barking</audio_summary>"
                    "<confidence>high</confidence>"
                    "<verdict>match</verdict>"
                    "</review>"
                )
            )

    def test_parse_review_rejects_invalid_enums(self) -> None:
        module = _load_script_module()
        with self.assertRaisesRegex(RuntimeError, "audio_type must be"):
            module._parse_review(
                (
                    "<review>"
                    "<visual_summary>dog in park</visual_summary>"
                    "<primary_subject>dog</primary_subject>"
                    "<scene>park</scene>"
                    "<style>natural</style>"
                    "<camera>tracking</camera>"
                    "<audio_present>true</audio_present>"
                    "<audio_type>sirens</audio_type>"
                    "<audio_summary>strange siren-like sound</audio_summary>"
                    "<confidence>high</confidence>"
                    "<verdict>match</verdict>"
                    "</review>"
                )
            )

    def test_strict_review_rejects_low_confidence(self) -> None:
        module = _load_script_module()
        review = module.ParsedReview(
            visual_summary="dog in a park",
            primary_subject="dog",
            scene="park",
            style="naturalistic",
            camera="tracking",
            audio_present="true",
            audio_type="barking",
            audio_summary="happy barking",
            confidence="low",
            verdict="partial",
        )
        with self.assertRaisesRegex(RuntimeError, "confidence is too low"):
            module._enforce_strict_review(review)

    def test_strict_review_rejects_partial_verdict(self) -> None:
        module = _load_script_module()
        review = module.ParsedReview(
            visual_summary="dog in a park",
            primary_subject="dog",
            scene="park",
            style="naturalistic",
            camera="tracking",
            audio_present="true",
            audio_type="barking",
            audio_summary="happy barking",
            confidence="high",
            verdict="partial",
        )
        with self.assertRaisesRegex(RuntimeError, "verdict must be match"):
            module._enforce_strict_review(review)

    def test_main_stages_artifact_runs_gemini_and_writes_receipts(self) -> None:
        module = _load_script_module()
        xml_response = (
            "<review>"
            "<visual_summary>A dog runs through a grassy park.</visual_summary>"
            "<primary_subject>dog</primary_subject>"
            "<scene>grassy park</scene>"
            "<style>cinematic naturalism</style>"
            "<camera>tracking left to right</camera>"
            "<audio_present>true</audio_present>"
            "<audio_type>barking</audio_type>"
            "<audio_summary>happy barking over park ambience</audio_summary>"
            "<confidence>high</confidence>"
            "<verdict>match</verdict>"
            "</review>"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            video_path = temp_root / "dog.mp4"
            save_dir = temp_root / "review"
            video_path.write_bytes(b"fake-video")
            stdout = io.StringIO()
            command_log: list[list[str]] = []

            def fake_run(
                command: list[str],
                *,
                check: bool,
                capture_output: bool,
                text: bool,
                cwd: Path,
            ) -> CompletedProcess[str]:
                self.assertFalse(check)
                self.assertTrue(capture_output)
                self.assertTrue(text)
                self.assertEqual(cwd, save_dir.resolve())
                command_log.append(command)
                return CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=xml_response,
                    stderr="",
                )

            argv = [
                "gemini_describe_video.py",
                str(video_path),
                "--save-dir",
                str(save_dir),
                "--strict",
            ]
            with (
                patch.object(module.subprocess, "run", side_effect=fake_run),
                patch.object(sys, "argv", argv),
                patch("sys.stdout", stdout),
            ):
                exit_code = module.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(command_log), 1)
            self.assertEqual(command_log[0][0], "gemini")
            self.assertEqual(command_log[0][1:3], ["-m", module.DEFAULT_MODEL])
            self.assertIn("-p", command_log[0])
            self.assertIn("--include-directories", command_log[0])
            self.assertIn(str(save_dir.resolve()), command_log[0])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["primary_subject"], "dog")
            self.assertEqual(payload["audio_type"], "barking")
            self.assertEqual(payload["verdict"], "match")

            staged_video = save_dir / "dog.mp4"
            prompt_path = save_dir / "gemini-review-prompt.txt"
            raw_path = save_dir / "gemini-review-raw.txt"
            parsed_path = save_dir / "gemini-review.json"
            self.assertTrue(staged_video.exists())
            self.assertTrue(prompt_path.exists())
            self.assertTrue(raw_path.exists())
            self.assertTrue(parsed_path.exists())
            self.assertEqual(
                json.loads(parsed_path.read_text(encoding="utf-8")),
                {
                    "visual_summary": "A dog runs through a grassy park.",
                    "primary_subject": "dog",
                    "scene": "grassy park",
                    "style": "cinematic naturalism",
                    "camera": "tracking left to right",
                    "audio_present": "true",
                    "audio_type": "barking",
                    "audio_summary": "happy barking over park ambience",
                    "confidence": "high",
                    "verdict": "match",
                },
            )
            self.assertIn("@dog.mp4", prompt_path.read_text(encoding="utf-8"))

    def test_main_returns_nonzero_when_gemini_output_has_no_review_xml(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            video_path = temp_root / "dog.mp4"
            save_dir = temp_root / "review"
            video_path.write_bytes(b"fake-video")
            stderr = io.StringIO()
            argv = [
                "gemini_describe_video.py",
                str(video_path),
                "--save-dir",
                str(save_dir),
            ]
            with (
                patch.object(
                    module.subprocess,
                    "run",
                    return_value=CompletedProcess(
                        args=["gemini"],
                        returncode=0,
                        stdout="not xml",
                        stderr="",
                    ),
                ),
                patch.object(sys, "argv", argv),
                patch("sys.stderr", stderr),
            ):
                exit_code = module.main()

            self.assertEqual(exit_code, 2)
            self.assertIn("did not contain a <review> XML block", stderr.getvalue())

    def test_main_without_save_dir_does_not_claim_persisted_artifacts(self) -> None:
        module = _load_script_module()
        xml_response = (
            "<review>"
            "<visual_summary>A dog in a park.</visual_summary>"
            "<primary_subject>dog</primary_subject>"
            "<scene>park</scene>"
            "<style>natural</style>"
            "<camera>tracking</camera>"
            "<audio_present>false</audio_present>"
            "<audio_type>unknown</audio_type>"
            "<audio_summary>no audible track</audio_summary>"
            "<confidence>high</confidence>"
            "<verdict>match</verdict>"
            "</review>"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "dog.mp4"
            video_path.write_bytes(b"fake-video")
            stdout = io.StringIO()
            argv = [
                "gemini_describe_video.py",
                str(video_path),
            ]
            with (
                patch.object(
                    module.subprocess,
                    "run",
                    return_value=CompletedProcess(
                        args=["gemini"],
                        returncode=0,
                        stdout=xml_response,
                        stderr="",
                    ),
                ),
                patch.object(sys, "argv", argv),
                patch("sys.stdout", stdout),
            ):
                exit_code = module.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertIsNone(payload["artifacts"])

    def test_extract_review_xml_rejects_multiple_review_blocks(self) -> None:
        module = _load_script_module()
        with self.assertRaisesRegex(RuntimeError, "exactly one <review> XML block"):
            module._extract_review_xml(
                (
                    "<review><visual_summary>one</visual_summary></review>"
                    "<review><visual_summary>two</visual_summary></review>"
                )
            )

    def test_stage_video_rejects_oversize_media(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "oversize.mp4"
            with video_path.open("wb") as handle:
                handle.truncate(module.MAX_INLINE_VIDEO_BYTES + 1)
            with self.assertRaisesRegex(RuntimeError, "inline review limit of 20 MB"):
                module._stage_video(video_path, None)

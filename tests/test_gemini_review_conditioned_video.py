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
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "gemini_review_conditioned_video.py"
    )
    spec = importlib.util.spec_from_file_location(
        "gemini_review_conditioned_video", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/gemini_review_conditioned_video.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GeminiReviewConditionedVideoScriptTests(unittest.TestCase):
    def test_build_prompt_mentions_reference_roles_and_aspect_ratio(self) -> None:
        module = _load_script_module()
        prompt = module._build_prompt(
            staged_video_path=Path("/tmp/review/clip.mp4"),
            video_probe=module.MediaProbe(width=768, height=512, duration_seconds=6.0),
            staged_references=(
                module.StagedReference(
                    role="start",
                    original_path=Path("/tmp/source/start.png"),
                    staged_path=Path("/tmp/review/start-start.png"),
                    probe=module.MediaProbe(
                        width=768, height=768, duration_seconds=None
                    ),
                ),
                module.StagedReference(
                    role="end",
                    original_path=Path("/tmp/source/end.png"),
                    staged_path=Path("/tmp/review/end-end.png"),
                    probe=module.MediaProbe(
                        width=1344, height=768, duration_seconds=None
                    ),
                ),
            ),
            prompt_text="Samurai transforms into a fire-wreathed swordsman.",
            extra_instruction="Be strict about semantic drift.",
        )
        self.assertIn("aspect-ratio mismatch", prompt)
        self.assertIn("@clip.mp4", prompt)
        self.assertIn("@start-start.png", prompt)
        self.assertIn("@end-end.png", prompt)
        self.assertIn("768x512", prompt)
        self.assertIn("768x768", prompt)
        self.assertIn("Be strict about semantic drift.", prompt)

    def test_parse_review_requires_expected_fields(self) -> None:
        module = _load_script_module()
        with self.assertRaisesRegex(RuntimeError, "missing required <issues> content"):
            module._parse_review(
                (
                    "<review>"
                    "<visual_summary>Samurai in alley.</visual_summary>"
                    "<opening_frame_match>match</opening_frame_match>"
                    "<closing_frame_match>partial</closing_frame_match>"
                    "<subject_consistency>medium</subject_consistency>"
                    "<motion_coherence>medium</motion_coherence>"
                    "<prompt_alignment>partial</prompt_alignment>"
                    "<aspect_ratio_assessment>Looks slightly cropped.</aspect_ratio_assessment>"
                    "<stretch_or_squash>minor</stretch_or_squash>"
                    "<confidence>high</confidence>"
                    "<verdict>partial</verdict>"
                    "</review>"
                )
            )

    def test_main_runs_gemini_and_writes_receipts(self) -> None:
        module = _load_script_module()
        xml_response = (
            "<review>"
            "<visual_summary>The same samurai remains in a rainy alley and gains fire.</visual_summary>"
            "<opening_frame_match>match</opening_frame_match>"
            "<closing_frame_match>partial</closing_frame_match>"
            "<subject_consistency>medium</subject_consistency>"
            "<motion_coherence>medium</motion_coherence>"
            "<prompt_alignment>partial</prompt_alignment>"
            "<aspect_ratio_assessment>The start reference is square while the video is widescreen, so framing is adapted but not badly distorted.</aspect_ratio_assessment>"
            "<stretch_or_squash>minor</stretch_or_squash>"
            "<issues>The ending drifts away from the target still.</issues>"
            "<confidence>high</confidence>"
            "<verdict>partial</verdict>"
            "</review>"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            video_path = temp_root / "clip.mp4"
            start_path = temp_root / "start.png"
            end_path = temp_root / "end.png"
            save_dir = temp_root / "review"
            prompt_file = temp_root / "prompt.txt"
            video_path.write_bytes(b"fake-video")
            start_path.write_bytes(b"fake-start")
            end_path.write_bytes(b"fake-end")
            prompt_file.write_text(
                "Single continuous shot of a samurai turning into a fire-wreathed swordsman.",
                encoding="utf-8",
            )
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
                if command[0] == "ffprobe":
                    target = Path(command[-1]).name
                    if target == "clip.mp4":
                        stdout_payload = json.dumps(
                            {
                                "streams": [{"width": 768, "height": 512}],
                                "format": {"duration": "10.041667"},
                            }
                        )
                    elif target == "start.png":
                        stdout_payload = json.dumps(
                            {"streams": [{"width": 768, "height": 768}], "format": {}}
                        )
                    else:
                        stdout_payload = json.dumps(
                            {
                                "streams": [{"width": 1344, "height": 768}],
                                "format": {},
                            }
                        )
                    return CompletedProcess(
                        args=command,
                        returncode=0,
                        stdout=stdout_payload,
                        stderr="",
                    )
                return CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=xml_response,
                    stderr="",
                )

            argv = [
                "gemini_review_conditioned_video.py",
                str(video_path),
                "--save-dir",
                str(save_dir),
                "--prompt-file",
                str(prompt_file),
                "--reference-image",
                str(start_path),
                "start",
                "--reference-image",
                str(end_path),
                "end",
            ]
            with (
                patch.object(module.subprocess, "run", side_effect=fake_run),
                patch.object(sys, "argv", argv),
                patch("sys.stdout", stdout),
            ):
                exit_code = module.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                [command[0] for command in command_log],
                ["ffprobe", "ffprobe", "ffprobe", "gemini"],
            )
            self.assertIn("--include-directories", command_log[-1])
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["opening_frame_match"], "match")
            self.assertEqual(payload["closing_frame_match"], "partial")
            self.assertEqual(payload["stretch_or_squash"], "minor")
            self.assertEqual(payload["video"]["aspect_ratio"], 1.5)

            prompt_path = save_dir / "gemini-conditioned-review-prompt.txt"
            raw_path = save_dir / "gemini-conditioned-review-raw.txt"
            parsed_path = save_dir / "gemini-conditioned-review.json"
            self.assertTrue(prompt_path.exists())
            self.assertTrue(raw_path.exists())
            self.assertTrue(parsed_path.exists())
            parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed_payload["verdict"], "partial")
            self.assertEqual(len(parsed_payload["references"]), 2)
            self.assertIn("@start-start.png", prompt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

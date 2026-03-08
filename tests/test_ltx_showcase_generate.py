from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "ltx_showcase_generate.py"
    )
    spec = importlib.util.spec_from_file_location("ltx_showcase_generate", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/ltx_showcase_generate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LTXShowcaseGenerateScriptTests(unittest.TestCase):
    def test_selected_scenes_defaults_to_full_set(self) -> None:
        module = _load_module()

        scenes = module._selected_scenes(None)

        self.assertEqual(len(scenes), len(module.SHOWCASE_SCENES))

    def test_review_matches_scene_checks_subject_audio_and_verdict(self) -> None:
        module = _load_module()
        scene = next(
            candidate
            for candidate in module.SHOWCASE_SCENES
            if candidate.scene_id == "dog_park_natural"
        )

        self.assertTrue(
            module._review_matches_scene(
                parsed_review={
                    "visual_summary": "Dog and owner in a park.",
                    "primary_subject": "Golden retriever dog",
                    "scene": "Park path",
                    "style": "Naturalistic",
                    "camera": "Tracking shot",
                    "audio_present": "true",
                    "audio_type": "barking",
                    "audio_summary": "Happy barking and park ambience.",
                    "confidence": "high",
                    "verdict": "match",
                },
                scene=scene,
                strict=True,
            )
        )
        self.assertFalse(
            module._review_matches_scene(
                parsed_review={
                    "visual_summary": "Dog and owner in a park.",
                    "primary_subject": "Golden retriever dog",
                    "scene": "Park path",
                    "style": "Naturalistic",
                    "camera": "Tracking shot",
                    "audio_present": "true",
                    "audio_type": "music",
                    "audio_summary": "Soft piano music.",
                    "confidence": "high",
                    "verdict": "match",
                },
                scene=scene,
                strict=True,
            )
        )

    def test_review_matches_scene_uses_scene_terms_for_non_dog_scenes(self) -> None:
        module = _load_module()
        scene = next(
            candidate
            for candidate in module.SHOWCASE_SCENES
            if candidate.scene_id == "vintage_old_school"
        )

        self.assertTrue(
            module._review_matches_scene(
                parsed_review={
                    "visual_summary": "People walking on a vintage street outside a shop.",
                    "primary_subject": "People",
                    "scene": "A 1970s street scene outside a neighborhood shop with parked cars.",
                    "style": "Vintage film",
                    "camera": "Street-level",
                    "audio_present": "true",
                    "audio_type": "ambience",
                    "audio_summary": "Street ambience and distant conversation.",
                    "confidence": "high",
                    "verdict": "match",
                },
                scene=scene,
                strict=True,
            )
        )

    def test_review_matches_scene_accepts_anime_character_wording(self) -> None:
        module = _load_module()
        scene = next(
            candidate
            for candidate in module.SHOWCASE_SCENES
            if candidate.scene_id == "anime_neon_chase"
        )

        self.assertTrue(
            module._review_matches_scene(
                parsed_review={
                    "visual_summary": "An anime character running down a neon city street.",
                    "primary_subject": "A young female anime character.",
                    "scene": "A neon city street at night with glowing signs and traffic.",
                    "style": "2D anime",
                    "camera": "Tracking side shot",
                    "audio_present": "true",
                    "audio_type": "ambience",
                    "audio_summary": "Traffic and city noise.",
                    "confidence": "high",
                    "verdict": "match",
                },
                scene=scene,
                strict=True,
            )
        )

    def test_review_matches_scene_accepts_stop_motion_desk_wording(self) -> None:
        module = _load_module()
        scene = next(
            candidate
            for candidate in module.SHOWCASE_SCENES
            if candidate.scene_id == "stop_motion_workshop"
        )

        self.assertTrue(
            module._review_matches_scene(
                parsed_review={
                    "visual_summary": "A stop-motion fox figurine on a wooden desk under a lamp.",
                    "primary_subject": "A crafted stop-motion fox figurine.",
                    "scene": "A small workbench or desk with tools and a warm lamp.",
                    "style": "Stop-motion miniature",
                    "camera": "Static close-up",
                    "audio_present": "true",
                    "audio_type": "music",
                    "audio_summary": "Light whimsical instrumental music.",
                    "confidence": "high",
                    "verdict": "match",
                },
                scene=scene,
                strict=True,
            )
        )

    def test_run_gemini_review_uses_expected_save_dir(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"clip")

            captured: dict[str, object] = {}

            class _Completed:
                def __init__(self) -> None:
                    self.returncode = 0
                    self.stdout = (
                        '{"visual_summary":"Dog in a park.",'
                        '"primary_subject":"dog",'
                        '"scene":"park",'
                        '"style":"naturalistic",'
                        '"camera":"tracking",'
                        '"audio_present":"true",'
                        '"audio_type":"barking",'
                        '"audio_summary":"happy barking",'
                        '"confidence":"high",'
                        '"verdict":"match"}'
                    )

            def fake_run(command: list[str], **kwargs: object) -> _Completed:
                captured["command"] = command
                captured["kwargs"] = kwargs
                return _Completed()

            original_run = module.subprocess.run
            module.subprocess.run = fake_run
            try:
                payload = module._run_gemini_review(video_path=video_path)
            finally:
                module.subprocess.run = original_run

            self.assertEqual(payload["verdict"], "match")
            command = captured["command"]
            assert isinstance(command, list)
            self.assertNotIn("--strict", command)
            self.assertIn(str(video_path.parent / "gemini-review"), command)

    def test_run_gemini_review_surfaces_nonzero_exit_as_runtime_error(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"clip")

            class _Completed:
                def __init__(self) -> None:
                    self.returncode = 2
                    self.stdout = ""
                    self.stderr = "gemini_describe_video.py: media staging failed"

            def fake_run(command: list[str], **kwargs: object) -> _Completed:
                return _Completed()

            original_run = module.subprocess.run
            module.subprocess.run = fake_run
            try:
                with self.assertRaisesRegex(RuntimeError, "media staging failed"):
                    module._run_gemini_review(video_path=video_path)
            finally:
                module.subprocess.run = original_run

    def test_write_ffprobe_persists_json_beside_video(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"clip")

            class _Completed:
                def __init__(self) -> None:
                    self.stdout = '{"streams":[],"format":{"duration":"10.0"}}'

            def fake_run(command: list[str], **kwargs: object) -> _Completed:
                return _Completed()

            original_run = module.subprocess.run
            module.subprocess.run = fake_run
            try:
                ffprobe_path = module._write_ffprobe(video_path)
            finally:
                module.subprocess.run = original_run

            self.assertEqual(ffprobe_path, video_path.parent / "ffprobe.json")
            self.assertEqual(
                ffprobe_path.read_text(encoding="utf-8"),
                '{"streams":[],"format":{"duration":"10.0"}}',
            )


if __name__ == "__main__":
    unittest.main()

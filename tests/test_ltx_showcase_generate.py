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
    def test_default_scenario_pack_loads_expected_fixture_entries(self) -> None:
        module = _load_module()

        scenes = module._load_scenarios(module.DEFAULT_SCENARIO_PACK)

        self.assertEqual(len(scenes), 12)
        self.assertEqual(scenes[0].scene_id, "cat_kitchen_natural")
        self.assertEqual(scenes[-1].scene_id, "pier_sunset_combined_multimodal")

    def test_selected_scenes_defaults_to_full_set(self) -> None:
        module = _load_module()

        scenes = module._selected_scenes(module.SHOWCASE_SCENES, None)

        self.assertEqual(len(scenes), len(module.SHOWCASE_SCENES))

    def test_review_matches_scene_checks_subject_audio_and_verdict(self) -> None:
        module = _load_module()
        scene = next(
            candidate
            for candidate in module.SHOWCASE_SCENES
            if candidate.scene_id == "cat_kitchen_natural"
        )

        self.assertTrue(
            module._review_matches_scene(
                parsed_review={
                    "visual_summary": "A tabby cat stands on a kitchen counter near a window.",
                    "primary_subject": "Tabby cat",
                    "scene": "Sunlit kitchen with a counter and window.",
                    "style": "Naturalistic",
                    "camera": "Static medium shot",
                    "audio_present": "true",
                    "audio_type": "ambience",
                    "audio_summary": "Birdsong, refrigerator hum, and soft household ambience.",
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
                    "visual_summary": "A tabby cat stands on a kitchen counter near a window.",
                    "primary_subject": "Tabby cat",
                    "scene": "Sunlit kitchen with a counter and window.",
                    "style": "Naturalistic",
                    "camera": "Static medium shot",
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
            if candidate.scene_id == "vintage_laundromat"
        )

        self.assertTrue(
            module._review_matches_scene(
                parsed_review={
                    "visual_summary": "A young man leans against a washing machine in a vintage laundromat.",
                    "primary_subject": "Young man in denim jacket",
                    "scene": "A 1970s laundromat interior with washers and dryers.",
                    "style": "Vintage film",
                    "camera": "Medium shot from the end of the machine row",
                    "audio_present": "true",
                    "audio_type": "ambience",
                    "audio_summary": "Mechanical washer churn and fluorescent hum.",
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
            if candidate.scene_id == "anime_rooftop_duel"
        )

        self.assertTrue(
            module._review_matches_scene(
                parsed_review={
                    "visual_summary": "Two anime fighters clash swords on a rooftop above a neon city.",
                    "primary_subject": "Two anime swordsmen",
                    "scene": "Skyscraper rooftop at night with neon city below.",
                    "style": "2D anime",
                    "camera": "Fast orbital track",
                    "audio_present": "true",
                    "audio_type": "music",
                    "audio_summary": "Driving percussive action music with blade clashes.",
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
            if candidate.scene_id == "claymation_bakery"
        )

        self.assertTrue(
            module._review_matches_scene(
                parsed_review={
                    "visual_summary": "A clay baker kneads dough on a miniature bakery counter.",
                    "primary_subject": "Clay baker figurine",
                    "scene": "Miniature bakery counter with oven and bread.",
                    "style": "Stop-motion miniature",
                    "camera": "Static close-up",
                    "audio_present": "true",
                    "audio_type": "music",
                    "audio_summary": "Whimsical xylophone melody with playful foley.",
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

    def test_run_smoke_includes_negative_prompt_when_present(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            artifact_root.mkdir()
            output_root = Path(tmp_dir) / "output"
            output_root.mkdir()

            captured: dict[str, object] = {}

            class _Completed:
                def __init__(self) -> None:
                    self.returncode = 0
                    self.stdout = '{"video_path":"/tmp/clip.mp4","manifest_path":"/tmp/manifest.json"}'

            def fake_run(command: list[str], **kwargs: object) -> _Completed:
                captured["command"] = command
                return _Completed()

            original_run = module.subprocess.run
            module.subprocess.run = fake_run
            try:
                payload = module._run_smoke(
                    artifact_root=artifact_root,
                    output_root=output_root,
                    run_name="dog-scene",
                    prompt="dog in park",
                    negative_prompt="background music, soundtrack",
                    width=384,
                    height=224,
                    num_frames=241,
                    fps=24,
                )
            finally:
                module.subprocess.run = original_run

            self.assertEqual(payload["video_path"], "/tmp/clip.mp4")
            command = captured["command"]
            assert isinstance(command, list)
            self.assertIn("--negative-prompt", command)
            self.assertIn("background music, soundtrack", command)

    def test_prompt_options_for_run_uses_requested_duration(self) -> None:
        module = _load_module()
        scene = next(
            candidate
            for candidate in module.SHOWCASE_SCENES
            if candidate.scene_id == "cat_kitchen_natural"
        )

        adjusted = module._prompt_options_for_run(scene, num_frames=145, fps=24)

        self.assertAlmostEqual(adjusted.duration_seconds, 6.0)
        self.assertEqual(
            adjusted.audio_prompt,
            "Paw impact on countertop, chair creak, birdsong through window, refrigerator hum.",
        )
        self.assertTrue(adjusted.natural_audio)
        self.assertTrue(adjusted.no_music)
        self.assertEqual(adjusted.style_family, "naturalistic")
        self.assertEqual(adjusted.orientation, "landscape")

    def test_prompt_options_for_conditioned_scene_skip_text_audio_prompt(self) -> None:
        module = _load_module()
        scene = next(
            candidate
            for candidate in module.SHOWCASE_SCENES
            if candidate.scene_id == "rain_alley_audio_conditioned"
        )

        adjusted = module._prompt_options_for_run(scene, num_frames=241, fps=24)

        self.assertIsNone(adjusted.audio_prompt)
        self.assertEqual(scene.conditioning_mode, "audio_conditioned")

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

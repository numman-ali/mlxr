from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import numpy as np


def _load_script_module() -> types.ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "ltx_debug_smoke.py"
    spec = importlib.util.spec_from_file_location("ltx_debug_smoke", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/ltx_debug_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(slots=True)
class _FakeGeneratedVideo:
    frames: np.ndarray
    fps: int
    seed: int
    backend: str
    conditioning_count: int
    prompt_signature: str
    metadata: dict[str, object]


class LTXDebugSmokeScriptTests(unittest.TestCase):
    def test_build_config_resolves_artifact_root(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "payload"
            checkpoint = (
                artifact_root / "checkpoint" / "ltx-2.3-22b-distilled.safetensors"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("ckpt", encoding="utf-8")
            upsampler = (
                artifact_root
                / "spatial_upsampler"
                / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
            )
            upsampler.parent.mkdir(parents=True, exist_ok=True)
            upsampler.write_text("upsampler", encoding="utf-8")
            text_encoder = artifact_root / "text_encoder"
            text_encoder.mkdir(parents=True, exist_ok=True)

            args = module.parse_args(
                [
                    "--artifact-root",
                    str(artifact_root),
                    "--prompt",
                    "a dog in a park",
                ]
            )
            config = module.build_config(args)

            self.assertEqual(config.artifact_paths.artifact_root, artifact_root)
            self.assertEqual(config.artifact_paths.checkpoint_path, checkpoint)
            self.assertEqual(config.artifact_paths.spatial_upsampler_path, upsampler)
            self.assertEqual(config.artifact_paths.text_encoder_path, text_encoder)
            self.assertEqual(config.profile_name, "safe-smoke")
            self.assertEqual(config.width, 256)
            self.assertEqual(config.height, 160)
            self.assertEqual(config.num_frames, 17)
            self.assertEqual(config.fps, 24)
            self.assertTrue(config.stage_debug)
            self.assertTrue(config.clean_lifecycle)
            self.assertTrue(config.backend_progress)

    def test_build_config_profile_can_be_overridden(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "payload"
            checkpoint = (
                artifact_root / "checkpoint" / "ltx-2.3-22b-distilled.safetensors"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("ckpt", encoding="utf-8")
            upsampler = (
                artifact_root
                / "spatial_upsampler"
                / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
            )
            upsampler.parent.mkdir(parents=True, exist_ok=True)
            upsampler.write_text("upsampler", encoding="utf-8")
            text_encoder = artifact_root / "text_encoder"
            text_encoder.mkdir(parents=True, exist_ok=True)

            args = module.parse_args(
                [
                    "--artifact-root",
                    str(artifact_root),
                    "--prompt",
                    "a dog in a park",
                    "--profile",
                    "coherence",
                    "--num-frames",
                    "41",
                ]
            )
            config = module.build_config(args)

            self.assertEqual(config.profile_name, "coherence")
            self.assertEqual(config.width, 768)
            self.assertEqual(config.height, 512)
            self.assertEqual(config.num_frames, 41)
            self.assertEqual(config.fps, 24)

    def test_run_smoke_clean_lifecycle_writes_manifest_and_closes_resources(
        self,
    ) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "payload"
            checkpoint = (
                artifact_root / "checkpoint" / "ltx-2.3-22b-distilled.safetensors"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("ckpt", encoding="utf-8")
            upsampler = (
                artifact_root
                / "spatial_upsampler"
                / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
            )
            upsampler.parent.mkdir(parents=True, exist_ok=True)
            upsampler.write_text("upsampler", encoding="utf-8")
            text_encoder = artifact_root / "text_encoder"
            text_encoder.mkdir(parents=True, exist_ok=True)

            config = module.build_config(
                Namespace(
                    artifact_root=artifact_root,
                    checkpoint_path=None,
                    spatial_upsampler_path=None,
                    text_encoder_path=None,
                    prompt="a golden retriever dog playing in a grassy park",
                    profile="safe-smoke",
                    width=256,
                    height=160,
                    num_frames=17,
                    fps=24,
                    seed=1234,
                    output_root=Path(tmp_dir) / "runs",
                    run_name="dog-test",
                    stage_debug=True,
                    clean_lifecycle=True,
                    backend_progress=True,
                    heartbeat_seconds=0.01,
                    print_tmux_command=False,
                    tmux_session_name="mlxr-ltx-debug",
                )
            )

            call_log: list[str] = []
            test_case = self

            class FakeEncoder:
                def __init__(self) -> None:
                    self.closed = False

                def encode(
                    self, prompt: str, *, negative_prompt: str | None = None
                ) -> object:
                    call_log.append(f"encode:{prompt}")
                    test_case.assertIsNone(negative_prompt)
                    return types.SimpleNamespace(
                        video_context=object(),
                        audio_context=object(),
                        attention_mask=object(),
                        prompt_text=prompt,
                        token_count=10,
                        sequence_length=16,
                        video_context_shape=(1, 16, 4096),
                        attention_mask_shape=(1, 16),
                        audio_context_shape=(1, 16, 2048),
                        context_representation="post_connector",
                        caption_proj_before_connector=True,
                        rope_type="split",
                        double_precision_rope=True,
                        connector_apply_gated_attention=True,
                        transformer_context_dim=4096,
                    )

                def close(self) -> None:
                    call_log.append("encoder.close")
                    self.closed = True

            class FakeGenerator:
                def __init__(self) -> None:
                    self.closed = False

                def generate(
                    self,
                    *,
                    prompt_context: object,
                    conditioning_inputs: tuple[object, ...],
                    width: int,
                    height: int,
                    num_frames: int,
                    fps: int,
                    seed: int | None = None,
                ) -> _FakeGeneratedVideo:
                    del prompt_context, conditioning_inputs, fps
                    call_log.append("generator.generate")
                    test_case.assertEqual(os.environ["MLXR_LTX_DEBUG_PROGRESS"], "1")
                    test_case.assertTrue(
                        os.environ["MLXR_LTX_DEBUG_STAGE_DUMPS_DIR"].endswith("/debug")
                    )
                    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
                    frames[..., 1] = 180
                    return _FakeGeneratedVideo(
                        frames=frames,
                        fps=24,
                        seed=seed or 0,
                        backend="mlx_video_distilled_two_stage_bridge",
                        conditioning_count=0,
                        prompt_signature="sig",
                        metadata={
                            "pipeline_kind": "distilled_two_stage",
                            "output_width": width,
                            "output_height": height,
                        },
                    )

                def close(self) -> None:
                    call_log.append("generator.close")
                    self.closed = True

            encoder = FakeEncoder()
            generator = FakeGenerator()

            def fake_prompt_encoder_factory(**_: object) -> FakeEncoder:
                call_log.append("encoder.create")
                return encoder

            def fake_video_generator_factory(**_: object) -> FakeGenerator:
                call_log.append("generator.create")
                self.assertTrue(encoder.closed)
                return generator

            def fake_mp4_encoder(video: _FakeGeneratedVideo, output_path: Path) -> None:
                del video
                call_log.append("mp4.encode")
                output_path.write_bytes(b"\x00\x00\x00\x18ftypisom")

            now_values = iter(
                (
                    datetime(2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc),
                    datetime(2026, 3, 7, 12, 0, 1, tzinfo=timezone.utc),
                )
            )
            cache_clearer = Mock()
            manifest = module.run_smoke(
                config,
                now=lambda: next(now_values),
                prompt_encoder_factory=fake_prompt_encoder_factory,
                video_generator_factory=fake_video_generator_factory,
                mp4_encoder=fake_mp4_encoder,
                cache_clearer=cache_clearer,
            )

            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["profile_name"], "safe-smoke")
            self.assertEqual(
                call_log[:5],
                [
                    "encoder.create",
                    "encode:a golden retriever dog playing in a grassy park",
                    "encoder.close",
                    "generator.create",
                    "generator.generate",
                ],
            )
            self.assertIn("mp4.encode", call_log)
            self.assertEqual(call_log[-1], "generator.close")
            cache_clearer.assert_called()

            run_dir = Path(manifest["outputs"]["run_dir"])
            self.assertTrue((run_dir / "frame_0001.png").exists())
            self.assertTrue((run_dir / "frame_mid.png").exists())
            self.assertTrue((run_dir / "frame_last.png").exists())
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertTrue((run_dir / "dog-test_256x160_17f.mp4").exists())
            self.assertEqual(
                Path(manifest["review_frame_paths"]["frame_mid"]),
                run_dir / "frame_mid.png",
            )
            self.assertEqual(
                Path(manifest["review_frame_paths"]["frame_last"]),
                run_dir / "frame_last.png",
            )

    def test_build_tmux_command_points_to_repo_script(self) -> None:
        module = _load_script_module()
        config = module.SmokeConfig(
            artifact_paths=module.ArtifactPaths(
                checkpoint_path=Path("/tmp/checkpoint.safetensors"),
                spatial_upsampler_path=Path("/tmp/upsampler.safetensors"),
                text_encoder_path=Path("/tmp/text_encoder"),
                artifact_root=None,
            ),
            prompt="a dog running through a park",
            profile_name="safe-smoke",
            width=256,
            height=160,
            num_frames=17,
            fps=24,
            seed=7,
            output_root=Path("/tmp/manual-runs"),
            run_name="dog-test",
            stage_debug=True,
            clean_lifecycle=True,
            backend_progress=True,
            heartbeat_seconds=2.0,
        )

        command = module.build_tmux_command(
            config,
            session_name="dog-session",
            log_path=Path("/tmp/manual-runs/dog-session.log"),
        )

        self.assertIn("tmux new-session -d -s dog-session", command)
        self.assertIn("scripts/ltx_debug_smoke.py", command)
        self.assertIn("/tmp/manual-runs/dog-session.log", command)
        self.assertIn("--run-name", command)
        self.assertIn("--profile", command)

    def test_run_smoke_fails_if_preview_backend_is_used(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "payload"
            checkpoint = (
                artifact_root / "checkpoint" / "ltx-2.3-22b-distilled.safetensors"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("ckpt", encoding="utf-8")
            upsampler = (
                artifact_root
                / "spatial_upsampler"
                / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
            )
            upsampler.parent.mkdir(parents=True, exist_ok=True)
            upsampler.write_text("upsampler", encoding="utf-8")
            text_encoder = artifact_root / "text_encoder"
            text_encoder.mkdir(parents=True, exist_ok=True)

            config = module.build_config(
                Namespace(
                    artifact_root=artifact_root,
                    checkpoint_path=None,
                    spatial_upsampler_path=None,
                    text_encoder_path=None,
                    prompt="a dog in a park",
                    profile="safe-smoke",
                    width=256,
                    height=160,
                    num_frames=17,
                    fps=24,
                    seed=1234,
                    output_root=Path(tmp_dir) / "runs",
                    run_name="preview-should-fail",
                    stage_debug=False,
                    clean_lifecycle=True,
                    backend_progress=False,
                    heartbeat_seconds=0.01,
                    print_tmux_command=False,
                    tmux_session_name="mlxr-ltx-debug",
                )
            )

            class FakeEncoder:
                def encode(
                    self, prompt: str, *, negative_prompt: str | None = None
                ) -> object:
                    del negative_prompt
                    return types.SimpleNamespace(
                        video_context=object(),
                        audio_context=object(),
                        attention_mask=object(),
                        prompt_text=prompt,
                        token_count=10,
                        sequence_length=16,
                    )

                def close(self) -> None:
                    return None

            class FakePreviewGenerator:
                def generate(
                    self,
                    *,
                    prompt_context: object,
                    conditioning_inputs: tuple[object, ...],
                    width: int,
                    height: int,
                    num_frames: int,
                    fps: int,
                    seed: int | None = None,
                ) -> _FakeGeneratedVideo:
                    del prompt_context, conditioning_inputs, fps, seed
                    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
                    return _FakeGeneratedVideo(
                        frames=frames,
                        fps=24,
                        seed=0,
                        backend="mlx_prompt_conditioned_preview",
                        conditioning_count=0,
                        prompt_signature="sig",
                        metadata={"pipeline_kind": "preview"},
                    )

                def close(self) -> None:
                    return None

            with self.assertRaisesRegex(RuntimeError, "preview backend"):
                module.run_smoke(
                    config,
                    prompt_encoder_factory=lambda **_: FakeEncoder(),
                    video_generator_factory=lambda **_: FakePreviewGenerator(),
                    mp4_encoder=lambda *_args, **_kwargs: None,
                    cache_clearer=lambda: None,
                )

    def test_run_smoke_fails_if_backend_is_unknown(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "payload"
            checkpoint = (
                artifact_root / "checkpoint" / "ltx-2.3-22b-distilled.safetensors"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("ckpt", encoding="utf-8")
            upsampler = (
                artifact_root
                / "spatial_upsampler"
                / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
            )
            upsampler.parent.mkdir(parents=True, exist_ok=True)
            upsampler.write_text("upsampler", encoding="utf-8")
            text_encoder = artifact_root / "text_encoder"
            text_encoder.mkdir(parents=True, exist_ok=True)

            config = module.build_config(
                Namespace(
                    artifact_root=artifact_root,
                    checkpoint_path=None,
                    spatial_upsampler_path=None,
                    text_encoder_path=None,
                    prompt="a dog in a park",
                    profile="safe-smoke",
                    width=256,
                    height=160,
                    num_frames=17,
                    fps=24,
                    seed=1234,
                    output_root=Path(tmp_dir) / "runs",
                    run_name="unknown-backend-should-fail",
                    stage_debug=False,
                    clean_lifecycle=True,
                    backend_progress=False,
                    heartbeat_seconds=0.01,
                    print_tmux_command=False,
                    tmux_session_name="mlxr-ltx-debug",
                )
            )

            class FakeEncoder:
                def encode(
                    self, prompt: str, *, negative_prompt: str | None = None
                ) -> object:
                    del negative_prompt
                    return types.SimpleNamespace(
                        video_context=object(),
                        audio_context=object(),
                        attention_mask=object(),
                        prompt_text=prompt,
                        token_count=10,
                        sequence_length=16,
                    )

                def close(self) -> None:
                    return None

            class FakeUnknownGenerator:
                def generate(
                    self,
                    *,
                    prompt_context: object,
                    conditioning_inputs: tuple[object, ...],
                    width: int,
                    height: int,
                    num_frames: int,
                    fps: int,
                    seed: int | None = None,
                ) -> _FakeGeneratedVideo:
                    del prompt_context, conditioning_inputs, fps, seed
                    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
                    return _FakeGeneratedVideo(
                        frames=frames,
                        fps=24,
                        seed=0,
                        backend="unexpected_bridge_name",
                        conditioning_count=0,
                        prompt_signature="sig",
                        metadata={"pipeline_kind": "distilled_two_stage"},
                    )

                def close(self) -> None:
                    return None

            with self.assertRaisesRegex(RuntimeError, "unexpected backend"):
                module.run_smoke(
                    config,
                    prompt_encoder_factory=lambda **_: FakeEncoder(),
                    video_generator_factory=lambda **_: FakeUnknownGenerator(),
                    mp4_encoder=lambda *_args, **_kwargs: None,
                    cache_clearer=lambda: None,
                )


if __name__ == "__main__":
    unittest.main()

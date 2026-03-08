from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from mlx_runtime_cli.cli import (
    RuntimeClient,
    _default_uds_path,
    _generation_params,
    _media_type_for_path,
    _references_from_args,
    build_parser,
)
from mlx_runtime_schemas import InputHandleRecord


class _FakeClient(RuntimeClient):
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def import_file(self, path: Path, *, kind: str) -> InputHandleRecord:
        self.calls.append((path, kind))
        return InputHandleRecord(
            handle_id=f"{kind}-handle",
            media_type=None,
            role=kind,
            storage_key=f"inputs/{kind}-handle",
        )


class RuntimeCliGenerateTests(unittest.TestCase):
    def test_generate_parser_accepts_simple_generation_args(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(
            [
                "generate",
                "--model-id",
                "ltx-2.3-fast-local",
                "--prompt",
                "golden retriever in a park",
                "--width",
                "96",
                "--height",
                "64",
                "--num-frames",
                "9",
            ]
        )
        self.assertEqual(parsed.command, "generate")
        self.assertEqual(parsed.model_id, "ltx-2.3-fast-local")
        self.assertEqual(parsed.prompt, "golden retriever in a park")
        self.assertEqual(parsed.width, 96)
        self.assertEqual(parsed.num_frames, 9)

    def test_generate_parser_accepts_local_image_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "conditioning.png"
            image_path.write_bytes(b"png")
            parser = build_parser()
            parsed = parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "dog in a park",
                    "--image",
                    str(image_path),
                    "--plan-only",
                ]
            )
            self.assertEqual(parsed.image, image_path)
            self.assertTrue(parsed.plan_only)

    def test_default_uds_path_uses_runtime_home_temp_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = Path(tmp_dir) / "runtime-home"
            previous = None
            import os

            previous = os.environ.get("MLX_RUNTIME_HOME")
            os.environ["MLX_RUNTIME_HOME"] = str(runtime_home)
            try:
                self.assertEqual(
                    _default_uds_path(),
                    runtime_home / "temp" / "control-plane.sock",
                )
            finally:
                if previous is None:
                    os.environ.pop("MLX_RUNTIME_HOME", None)
                else:
                    os.environ["MLX_RUNTIME_HOME"] = previous

    def test_media_type_for_path_supports_current_image_and_audio_fixtures(
        self,
    ) -> None:
        self.assertEqual(
            _media_type_for_path(Path("conditioning.ppm"), "image"),
            "image/x-portable-pixmap",
        )
        self.assertEqual(
            _media_type_for_path(Path("bark.wav"), "audio"),
            "audio/wav",
        )
        self.assertEqual(
            _media_type_for_path(Path("unknown.bin"), "image"),
            "application/octet-stream",
        )

    def test_generation_params_only_emits_explicit_values(self) -> None:
        args = argparse.Namespace(
            width=384,
            height=224,
            fps=24,
            seed=1234,
            num_frames=17,
        )
        self.assertEqual(
            _generation_params(args),
            {
                "width": 384,
                "height": 224,
                "fps": 24,
                "seed": 1234,
                "num_frames": 17,
            },
        )

    def test_references_from_args_plan_only_does_not_import_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "conditioning.png"
            audio_path = Path(tmp_dir) / "bark.wav"
            image_path.write_bytes(b"png")
            audio_path.write_bytes(b"wav")
            client = _FakeClient()
            args = argparse.Namespace(
                image=image_path,
                audio=audio_path,
                plan_only=True,
            )
            references = _references_from_args(client, args)
            self.assertEqual(client.calls, [])
            self.assertEqual(
                [(reference.kind, reference.input_handle) for reference in references],
                [("image", None), ("audio", None)],
            )

    def test_references_from_args_binds_imported_handles_for_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "conditioning.png"
            audio_path = Path(tmp_dir) / "bark.wav"
            image_path.write_bytes(b"png")
            audio_path.write_bytes(b"wav")
            client = _FakeClient()
            args = argparse.Namespace(
                image=image_path,
                audio=audio_path,
                plan_only=False,
            )
            references = _references_from_args(client, args)
            self.assertEqual(
                client.calls,
                [(image_path, "image"), (audio_path, "audio")],
            )
            self.assertEqual(
                [(reference.kind, reference.input_handle) for reference in references],
                [("image", "image-handle"), ("audio", "audio-handle")],
            )

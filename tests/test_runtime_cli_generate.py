from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlx_runtime_cli.cli import _default_uds_path, build_parser


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

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlx_runtime_cli.cli import build_parser


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

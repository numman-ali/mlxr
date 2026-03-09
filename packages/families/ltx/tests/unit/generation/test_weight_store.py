from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlxr.families.ltx._generation_backend.weight_store import (
    CheckpointIndex,
    CheckpointReader,
    CheckpointReaderView,
)
from safetensors.numpy import save_file


class CheckpointReaderTests(unittest.TestCase):
    def test_checkpoint_index_reads_keys_and_metadata_without_residency(self) -> None:
        checkpoint_path = self._write_checkpoint(
            {
                "model.diffusion_model.weight": mx.ones((2, 2)),
                "vae.decoder.bias": mx.zeros((2,)),
            },
            metadata={"format": "owned-test"},
        )

        index = CheckpointIndex(checkpoint_path)

        self.assertEqual(
            index.keys(),
            frozenset({"model.diffusion_model.weight", "vae.decoder.bias"}),
        )
        self.assertEqual(index.metadata(), {"format": "owned-test"})

    def test_checkpoint_reader_load_exact_and_release(self) -> None:
        checkpoint_path = self._write_checkpoint(
            {
                "a": mx.ones((2,)),
                "b": mx.zeros((3,)),
            }
        )
        reader = CheckpointReader(checkpoint_path)

        loaded = reader.load_exact(("a",))
        self.assertEqual(set(loaded), {"a"})
        self.assertEqual(tuple(int(size) for size in loaded["a"].shape), (2,))

        reader.release(keys=("a",))
        reloaded = reader.load_exact(("a",))
        self.assertEqual(tuple(int(size) for size in reloaded["a"].shape), (2,))

    def test_checkpoint_reader_load_prefixes_is_scoped(self) -> None:
        checkpoint_path = self._write_checkpoint(
            {
                "model.diffusion_model.layer.weight": mx.ones((1,)),
                "model.diffusion_model.layer.bias": mx.zeros((1,)),
                "vae.decoder.weight": mx.ones((1,)),
            }
        )
        reader = CheckpointReader(checkpoint_path)

        loaded = reader.load_prefixes(("model.diffusion_model.",))

        self.assertEqual(
            set(loaded),
            {
                "model.diffusion_model.layer.weight",
                "model.diffusion_model.layer.bias",
            },
        )

    def test_checkpoint_reader_view_materialize_and_release(self) -> None:
        checkpoint_path = self._write_checkpoint(
            {
                "x": mx.ones((1,)),
                "y": mx.zeros((1,)),
            }
        )
        reader = CheckpointReader(checkpoint_path)
        view = CheckpointReaderView(reader, ("x",))

        self.assertEqual(set(view.materialize()), {"x"})
        view.release()
        self.assertEqual(set(view.materialize()), {"x"})

    def test_checkpoint_reader_raises_on_missing_required_prefix(self) -> None:
        checkpoint_path = self._write_checkpoint({"foo": mx.ones((1,))})
        reader = CheckpointReader(checkpoint_path)

        with self.assertRaisesRegex(RuntimeError, "required prefixed weights"):
            reader.load_prefixes(("bar.",))

    def _write_checkpoint(
        self,
        tensors: dict[str, mx.array],
        *,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        checkpoint_path = Path(tmp_dir.name) / "weights.safetensors"
        save_file(
            {
                key: np.asarray(value.astype(mx.float32))
                for key, value in tensors.items()
            },
            str(checkpoint_path),
            metadata=metadata,
        )
        return checkpoint_path


if __name__ == "__main__":
    unittest.main()

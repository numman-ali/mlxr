from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlxr.families.ltx._generation_backend.rope_ops import precompute_freqs_cis
from mlxr.families.ltx._prompt_encoding_backend.masks import (
    _connector_precomputed_freqs,
)


class ConnectorRopeTests(unittest.TestCase):
    def test_connector_freqs_match_owned_rope_ops_interleaved(self) -> None:
        seq_len = 32
        dim = 4096
        theta = 10000.0
        max_pos = (4096,)
        num_heads = 32

        cos_expected, sin_expected = precompute_freqs_cis(
            mx.array(np.arange(seq_len, dtype=np.float32)[None, None, :]),
            dim=dim,
            out_dtype=mx.float32,
            theta=theta,
            max_pos=[4096],
            use_middle_indices_grid=False,
            num_attention_heads=num_heads,
            rope_type="interleaved",
            double_precision=False,
        )
        cos_actual, sin_actual = _connector_precomputed_freqs(
            seq_len,
            dim,
            num_heads,
            theta,
            max_pos,
            "interleaved",
            False,
        )

        np.testing.assert_allclose(np.asarray(cos_expected), cos_actual)
        np.testing.assert_allclose(np.asarray(sin_expected), sin_actual)

    def test_connector_freqs_match_owned_rope_ops_split(self) -> None:
        seq_len = 32
        dim = 4096
        theta = 10000.0
        max_pos = (4096,)
        num_heads = 32

        cos_expected, sin_expected = precompute_freqs_cis(
            mx.array(np.arange(seq_len, dtype=np.float32)[None, None, :]),
            dim=dim,
            out_dtype=mx.float32,
            theta=theta,
            max_pos=[4096],
            use_middle_indices_grid=False,
            num_attention_heads=num_heads,
            rope_type="split",
            double_precision=True,
        )
        cos_actual, sin_actual = _connector_precomputed_freqs(
            seq_len,
            dim,
            num_heads,
            theta,
            max_pos,
            "split",
            True,
        )

        np.testing.assert_allclose(np.asarray(cos_expected), cos_actual)
        np.testing.assert_allclose(np.asarray(sin_expected), sin_actual)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlx_runtime_core import FetchPolicy, HuggingFaceProviderAdapter
from mlx_runtime_schemas import SourceAuth, SourcePolicy, SourceRef


class FakeHfApi:
    def __init__(self, model_info: SimpleNamespace) -> None:
        self._model_info = model_info
        self.calls: list[dict[str, object]] = []

    def model_info(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._model_info


class HuggingFaceProviderTests(unittest.TestCase):
    def test_resolve_inspect_and_provenance(self) -> None:
        model_info = SimpleNamespace(
            sha="abc123",
            private=False,
            gated=False,
            pipeline_tag="text-to-video",
            cardData={"license": "other", "auto_map": {"AutoModel": "modeling_custom"}},
            siblings=[
                SimpleNamespace(rfilename="model.safetensors", size=123),
                SimpleNamespace(rfilename="modeling_custom.py", size=10),
            ],
        )
        provider = HuggingFaceProviderAdapter(api=FakeHfApi(model_info))
        source_ref = SourceRef(
            provider="huggingface",
            locator={"repo": "example/model", "revision": "main"},
            policy=SourcePolicy(allow_remote_code=True),
        )

        resolved = provider.resolve(source_ref)
        inspection = provider.inspect(resolved)
        provenance = provider.provenance(resolved)

        self.assertEqual(resolved.pinned_ref, "abc123")
        self.assertEqual(resolved.access_state, "public")
        self.assertTrue(resolved.remote_code_required)
        self.assertEqual(inspection.bytes_total, 133)
        self.assertEqual(provenance.license, "other")
        self.assertTrue(provenance.remote_code_approved)
        self.assertEqual(provenance.metadata["remote_code_detection_confidence"], "low")

    def test_fetch_uses_snapshot_download_with_allow_patterns(self) -> None:
        model_info = SimpleNamespace(
            sha="abc123",
            private=False,
            gated=True,
            pipeline_tag="text-to-video",
            cardData={"license": "other"},
            siblings=[SimpleNamespace(rfilename="model.safetensors", size=123)],
        )
        provider = HuggingFaceProviderAdapter(api=FakeHfApi(model_info))
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_ref = SourceRef(
                provider="huggingface",
                locator={"repo": "example/model", "revision": "main"},
                auth=SourceAuth(token_ref="hf-default"),
            )
            with patch.dict(os.environ, {"HF_TOKEN": "hf-secret"}, clear=False):
                resolved = provider.resolve(source_ref)
                with patch(
                    "mlx_runtime_core.providers.snapshot_download",
                    return_value=str(Path(tmp_dir) / "snapshot"),
                ) as snapshot_download:
                    materialization = provider.fetch(
                        resolved,
                        FetchPolicy(allow_patterns=("*.json", "*.safetensors")),
                    )

        self.assertEqual(materialization.materialization_mode, "provider-cache-ref")
        self.assertEqual(materialization.local_refs, (str(materialization.local_path),))
        snapshot_download.assert_called_once()

    def test_unsupported_token_ref_is_rejected(self) -> None:
        provider = HuggingFaceProviderAdapter(
            api=FakeHfApi(
                SimpleNamespace(
                    sha="abc123",
                    private=False,
                    gated=False,
                    pipeline_tag="text-to-video",
                    cardData={"license": "other"},
                    siblings=[],
                )
            )
        )
        source_ref = SourceRef(
            provider="huggingface",
            locator={"repo": "example/model"},
            auth=SourceAuth(token_ref="hf-custom"),
        )

        with self.assertRaises(ValueError):
            provider.auth_requirements(source_ref)

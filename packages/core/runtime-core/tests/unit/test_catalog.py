from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlxr.core.runtime import (
    PortableArtifact,
    RuntimeCatalog,
    RuntimeHome,
    RuntimeRegistry,
)
from mlxr.core.runtime.supported_models import SupportedModelRecipe
from mlxr.core.schemas import (
    ArtifactConversionRequest,
    CapabilityDescriptor,
    ModelRecord,
    PolicyDescriptor,
    PortableArtifactRecord,
    ProvenanceRecord,
    SourceAuth,
    SourceRef,
)


class _NormalizingFamily:
    family_id = "test_family"

    def inspect_source(self, source):  # pragma: no cover - not used here
        raise AssertionError("not used")

    def fetch_policy_for_conversion(self, role, source):  # pragma: no cover
        raise AssertionError("not used")

    def convert(self, sources, plan):  # pragma: no cover
        raise AssertionError("not used")

    def normalize_capability(self, artifact: PortableArtifact) -> CapabilityDescriptor:
        constraints = dict(artifact.record.capability.constraints)
        constraints["guidance_scale"] = {"fixed": 1.0}
        return artifact.record.capability.model_copy(
            update={"constraints": constraints}
        )

    def load(self, artifact, profile):  # pragma: no cover
        raise AssertionError("not used")

    def capabilities(self, artifact):  # pragma: no cover
        return artifact.record.capability

    def run_stage(self, loaded, stage):  # pragma: no cover
        raise AssertionError("not used")

    def unload(self, loaded):  # pragma: no cover
        return None


def _source_ref() -> SourceRef:
    return SourceRef(
        provider="huggingface",
        locator={"repo": "repo/test", "revision": "main"},
        auth=SourceAuth(token_ref="hf-default"),
        family_hint="test_family",
    )


def _artifact_record() -> PortableArtifactRecord:
    capability = CapabilityDescriptor(
        model_id="z-image-turbo-local",
        artifact_digest="sha256:test",
        family="test_family",
        tasks=["image.generate"],
        modalities_in=["text"],
        modalities_out=["image"],
        artifacts_out=["png"],
        scheduler_class="image_diffusion",
        constraints={"width": {"multiple_of": 16}},
        policy=PolicyDescriptor(access_state="public"),
        metadata={},
    )
    return PortableArtifactRecord(
        model_id="z-image-turbo-local",
        artifact_digest="sha256:test",
        family="test_family",
        format_version="1",
        weight_format="mlx",
        storage_key="artifacts/test_family/z-image-turbo-local/sha256:test",
        capability=capability,
        provenance=ProvenanceRecord(
            provider="huggingface",
            locator={"repo": "repo/test"},
            resolved_ref="main",
        ),
        components=[],
        metadata={},
    )


class RuntimeCatalogTests(unittest.TestCase):
    def test_get_model_normalizes_stale_capability_and_persists_it(self) -> None:
        registry = RuntimeRegistry()
        registry.register_family(_NormalizingFamily())
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            catalog = RuntimeCatalog(registry=registry, runtime_home=runtime_home)
            artifact = _artifact_record()
            runtime_home.artifact_dir(
                artifact.family, artifact.model_id, artifact.artifact_digest
            ).mkdir(parents=True, exist_ok=True)
            catalog.artifacts.save(artifact)
            catalog.models.save(
                ModelRecord(
                    model_id=artifact.model_id,
                    family=artifact.family,
                    source=_source_ref(),
                    artifact=artifact,
                    loaded=False,
                    capability=artifact.capability,
                )
            )

            model = catalog.get_model(artifact.model_id)

            self.assertEqual(
                model.capability.constraints["guidance_scale"],
                {"fixed": 1.0},
            )
            persisted = catalog.models.get(artifact.model_id)
            assert persisted is not None
            assert persisted.capability is not None
            self.assertEqual(
                persisted.capability.constraints["guidance_scale"],
                {"fixed": 1.0},
            )


class SupportedModelRecipeTests(unittest.TestCase):
    def test_to_conversion_request_requires_source_configuration(self) -> None:
        recipe = SupportedModelRecipe(
            model_id="test-model",
            display_name="Test Model",
            family="test_family",
            family_variant=None,
            recommendation_tier="recommended",
            support_level="supported",
            tasks=("image.generate",),
            provider="huggingface",
            source_summary="repo/test",
            license="apache-2.0",
            access_state="public",
        )

        with self.assertRaisesRegex(ValueError, "missing source configuration"):
            recipe.to_conversion_request(registered_source_ids={})

    def test_to_conversion_request_builds_family_binding_request(self) -> None:
        recipe = SupportedModelRecipe(
            model_id="ltx-test",
            display_name="LTX Test",
            family="ltx",
            family_variant=None,
            recommendation_tier="recommended",
            support_level="supported",
            tasks=("video.generate",),
            provider="huggingface",
            source_summary="repo/test",
            license="other",
            access_state="public",
            source_bindings={"checkpoint": _source_ref()},
        )

        request = recipe.to_conversion_request(
            registered_source_ids={"checkpoint": "src_checkpoint"}
        )

        self.assertEqual(
            request,
            ArtifactConversionRequest(
                source_bindings={"checkpoint": "src_checkpoint"},
                family="ltx",
                model_id="ltx-test",
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

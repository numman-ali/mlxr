from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

from mlxr.core.runtime import RuntimeHome
from mlxr.core.runtime.manifests import write_json_atomic
from mlxr.core.schemas import (
    AuthRequirements,
    CapabilityDescriptor,
    ExtensionSchemaDescriptor,
    ModelRecord,
    PolicyDescriptor,
    PortableArtifactComponentRecord,
    PortableArtifactRecord,
    ProvenanceRecord,
    ResolvedSource,
    SourceFileRecord,
    SourceRef,
    SourceRegistrationRecord,
)


def _load_script_module() -> types.ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "runtime_clone_home.py"
    )
    spec = importlib.util.spec_from_file_location("runtime_clone_home", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/runtime_clone_home.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeCloneHomeScriptTests(unittest.TestCase):
    def test_clone_runtime_home_reuses_existing_payloads(self) -> None:
        module = _load_script_module()
        with (
            tempfile.TemporaryDirectory() as source_dir,
            tempfile.TemporaryDirectory() as target_dir,
        ):
            source_runtime = RuntimeHome(Path(source_dir))
            target_runtime = RuntimeHome(Path(target_dir) / "runtime-home")
            source_runtime.ensure_layout()

            source_ref = SourceRef(
                provider="local",
                locator={"path": str((Path(source_dir) / "bundle").resolve())},
                family_hint="demo",
            )
            source_record = SourceRegistrationRecord(
                source_id="src_demo",
                source=source_ref,
                resolved_source=ResolvedSource(
                    provider="local",
                    locator=source_ref.locator,
                    pinned_ref="structural-sha256:test",
                    access_state="local-only",
                    auth_requirements=AuthRequirements(required=False),
                    files=[SourceFileRecord(path="weights.bin", size_bytes=5)],
                ),
                provenance=ProvenanceRecord(
                    provider="local",
                    locator=source_ref.locator,
                    resolved_ref="structural-sha256:test",
                    access_state="local-only",
                ),
                family_hint="demo",
            )
            write_json_atomic(
                source_runtime.source_manifest_path("local", "src_demo"),
                source_record,
            )

            capability = CapabilityDescriptor(
                model_id="demo-model",
                artifact_digest="sha256:test",
                family="demo",
                family_variant="test",
                tasks=["image.generate"],
                modalities_in=["text"],
                modalities_out=["image"],
                artifacts_out=["png"],
                scheduler_class="image_diffusion",
                policy=PolicyDescriptor(access_state="local-only"),
                extensions_schema=ExtensionSchemaDescriptor(
                    namespace="demo", version="1"
                ),
            )
            artifact = PortableArtifactRecord(
                model_id="demo-model",
                artifact_digest="sha256:test",
                family="demo",
                family_variant="test",
                format_version="0.1.0",
                weight_format="linked-test",
                storage_key="artifacts-portable/demo/demo-model/sha256_test",
                capability=capability,
                provenance=source_record.provenance,
                components=[
                    PortableArtifactComponentRecord(
                        role="checkpoint",
                        kind="file",
                        relative_path="payload/checkpoint/weights.bin",
                        storage_key="artifacts-portable/demo/demo-model/sha256_test/payload/checkpoint/weights.bin",
                        source_id="src_demo",
                        resolved_ref="structural-sha256:test",
                        size_bytes=5,
                        component_digest="sha256:file",
                        provenance=source_record.provenance,
                    )
                ],
            )
            source_artifact_path = source_runtime.artifact_manifest_path(
                artifact.family, artifact.model_id, artifact.artifact_digest
            )
            write_json_atomic(source_artifact_path, artifact)
            payload_path = (
                source_runtime.artifact_dir(
                    artifact.family, artifact.model_id, artifact.artifact_digest
                )
                / "payload"
                / "checkpoint"
                / "weights.bin"
            )
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(b"hello")

            model = ModelRecord(
                model_id="demo-model",
                family="demo",
                source=source_ref,
                artifact=artifact,
                loaded=False,
                capability=capability,
            )
            write_json_atomic(source_runtime.model_manifest_path("demo-model"), model)

            module.clone_runtime_home(
                source_runtime=source_runtime,
                target_runtime=target_runtime,
                model_ids=["demo-model"],
                refresh_manifests=False,
                reset=False,
            )

            target_model_path = target_runtime.model_manifest_path("demo-model")
            target_artifact_path = target_runtime.artifact_manifest_path(
                artifact.family, artifact.model_id, artifact.artifact_digest
            )
            target_source_path = target_runtime.source_manifest_path(
                "local", "src_demo"
            )
            target_payload_path = (
                target_runtime.artifact_dir(
                    artifact.family, artifact.model_id, artifact.artifact_digest
                )
                / "payload"
                / "checkpoint"
                / "weights.bin"
            )
            target_payload_root = (
                target_runtime.artifact_dir(
                    artifact.family, artifact.model_id, artifact.artifact_digest
                )
                / "payload"
            )

            self.assertTrue(target_model_path.exists())
            self.assertTrue(target_artifact_path.exists())
            self.assertTrue(target_source_path.exists())
            self.assertTrue(target_payload_root.is_symlink())
            self.assertTrue(target_payload_path.exists())
            self.assertEqual(target_payload_path.resolve(), payload_path.resolve())


if __name__ == "__main__":
    unittest.main()

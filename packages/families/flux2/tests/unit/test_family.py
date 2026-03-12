from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlxr.core.runtime import (
    ConversionPlan,
    ConversionSource,
    ExecutionProfile,
    ExecutionStage,
    LocalFileProviderAdapter,
    PortableArtifact,
)
from mlxr.core.schemas import ResolvedSource, SourceFileRecord, SourceRef
from mlxr.families.flux2 import Flux2FamilyAdapter
from mlxr.families.flux2.generation import GeneratedImage


def make_local_flux_bundle(root: Path, *, variant_dir_name: str) -> Path:
    bundle_root = root / variant_dir_name
    files = {
        "model_index.json": b"{}",
        "transformer/config.json": b"{}",
        "transformer/diffusion_pytorch_model.safetensors": b"transformer",
        "vae/config.json": b"{}",
        "vae/diffusion_pytorch_model.safetensors": b"vae",
        "text_encoder/config.json": b"{}",
        "text_encoder/model.safetensors": b"text-encoder",
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}",
        "scheduler/scheduler_config.json": b"{}",
    }
    for relative_path, payload in files.items():
        destination = bundle_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return bundle_root


class Flux2FamilyTests(unittest.TestCase):
    def test_inspect_source_requires_model_index(self) -> None:
        adapter = Flux2FamilyAdapter()
        source = ResolvedSource(
            provider="local",
            locator={"path": "/tmp/FLUX.2-klein-9B"},
            pinned_ref=None,
            family_hint="flux2",
            files=(
                SourceFileRecord(
                    path="transformer/config.json",
                    size_bytes=1,
                    digest="sha256:test",
                ),
            ),
            metadata={},
        )

        with self.assertRaisesRegex(ValueError, "requires model_index.json"):
            adapter.inspect_source(source)

    def test_inspect_source_recognizes_primary_klein_row(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-9B"
            )
            resolved = provider.resolve(
                SourceRef(
                    provider="local",
                    locator={"path": str(bundle_root)},
                    family_hint="flux2",
                )
            )

        inspection = adapter.inspect_source(resolved)

        self.assertEqual(inspection.variant, "flux.2-klein-9b")
        self.assertEqual(inspection.tasks, ("image.generate", "image.edit"))

    def test_inspect_source_recognizes_klein_kv_row(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-9B-kv"
            )
            resolved = provider.resolve(
                SourceRef(
                    provider="local",
                    locator={"path": str(bundle_root)},
                    family_hint="flux2",
                )
            )

        inspection = adapter.inspect_source(resolved)

        self.assertEqual(inspection.variant, "flux.2-klein-9b-kv")
        self.assertEqual(
            inspection.metadata["edit_optimized_klein_variant"],
            "flux.2-klein-9b-kv",
        )

    def test_inspect_source_honors_explicit_variant_hint(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="scratch-download-dir"
            )
            resolved = provider.resolve(
                SourceRef(
                    provider="local",
                    locator={
                        "path": str(bundle_root),
                        "variant": "flux.2-klein-9b",
                    },
                    family_hint="flux2",
                )
            )

        inspection = adapter.inspect_source(resolved)

        self.assertEqual(inspection.variant, "flux.2-klein-9b")

    def test_inspect_source_rejects_partial_bundle_without_transformer_weights(
        self,
    ) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-4B"
            )
            (
                bundle_root / "transformer" / "diffusion_pytorch_model.safetensors"
            ).unlink()
            resolved = provider.resolve(
                SourceRef(
                    provider="local",
                    locator={"path": str(bundle_root)},
                    family_hint="flux2",
                )
            )

            with self.assertRaisesRegex(
                ValueError, "requires a diffusers-style bundle"
            ):
                adapter.inspect_source(resolved)

    def test_convert_creates_componentized_artifact(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-dev"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )
            artifact = adapter.convert(
                {
                    "bundle": ConversionSource(
                        role="bundle",
                        source_id="src_bundle",
                        source=source_ref,
                        materialization=materialization,
                    )
                },
                ConversionPlan(model_id="flux2-dev-local"),
            )

        self.assertEqual(artifact.record.family_variant, "flux.2-dev")
        self.assertEqual(
            artifact.record.capability.tasks,
            ["image.generate", "image.edit"],
        )
        self.assertEqual(artifact.record.capability.artifacts_out, ["png", "jpg"])

    def test_convert_preserves_klein_kv_variant_and_constraints(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-9B-kv"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )
            artifact = adapter.convert(
                {
                    "bundle": ConversionSource(
                        role="bundle",
                        source_id="src_bundle",
                        source=source_ref,
                        materialization=materialization,
                    )
                },
                ConversionPlan(model_id="flux2-klein-9b-kv-local"),
            )

        self.assertEqual(artifact.record.family_variant, "flux.2-klein-9b-kv")
        self.assertEqual(
            artifact.record.capability.constraints["num_inference_steps"],
            {"fixed": 4},
        )
        self.assertEqual(
            artifact.record.metadata["edit_optimized_klein_variant"],
            "flux.2-klein-9b-kv",
        )

    def test_convert_rejects_incomplete_index_backed_component(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-9B"
            )
            (bundle_root / "text_encoder" / "model.safetensors").unlink()
            (bundle_root / "text_encoder" / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "model-00001-of-00002.safetensors"
                        }
                    }
                ),
                encoding="utf-8",
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )

            with self.assertRaisesRegex(
                ValueError, "missing shard files: model-00001-of-00002.safetensors"
            ):
                adapter.convert(
                    {
                        "bundle": ConversionSource(
                            role="bundle",
                            source_id="src_bundle",
                            source=source_ref,
                            materialization=materialization,
                        )
                    },
                    ConversionPlan(model_id="flux2-klein-9b-local"),
                )

    def test_fetch_policy_rejects_non_bundle_role(self) -> None:
        adapter = Flux2FamilyAdapter()
        with self.assertRaisesRegex(ValueError, "single 'bundle' source role"):
            adapter.fetch_policy_for_conversion(
                "checkpoint", provider_source("FLUX.2-dev")
            )

    def test_load_requires_materialized_storage_path(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-9B"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )
            converted = adapter.convert(
                {
                    "bundle": ConversionSource(
                        role="bundle",
                        source_id="src_bundle",
                        source=source_ref,
                        materialization=materialization,
                    )
                },
                ConversionPlan(model_id="flux2-klein-9b-local"),
            )
            artifact = PortableArtifact(record=converted.record, storage_path=None)

            with self.assertRaisesRegex(ValueError, "requires a materialized artifact"):
                adapter.load(
                    artifact,
                    ExecutionProfile(task="image.generate", profile="default"),
                )

    def test_load_rejects_missing_component_directory(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-9B"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )
            converted = adapter.convert(
                {
                    "bundle": ConversionSource(
                        role="bundle",
                        source_id="src_bundle",
                        source=source_ref,
                        materialization=materialization,
                    )
                },
                ConversionPlan(model_id="flux2-klein-9b-local"),
            )
            payload_root = Path(tmp_dir) / "artifact" / "payload"
            payload_root.mkdir(parents=True, exist_ok=True)
            artifact = PortableArtifact(
                record=converted.record,
                storage_path=payload_root.parent,
            )

            with self.assertRaisesRegex(
                ValueError, "missing required component directory"
            ):
                adapter.load(
                    artifact,
                    ExecutionProfile(task="image.generate", profile="default"),
                )

    def test_load_and_run_stage_generates_runtime_managed_artifact(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-9B"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )
            converted = adapter.convert(
                {
                    "bundle": ConversionSource(
                        role="bundle",
                        source_id="src_bundle",
                        source=source_ref,
                        materialization=materialization,
                    )
                },
                ConversionPlan(model_id="flux2-klein-9b-local"),
            )
            artifact_root = Path(tmp_dir) / "artifact"
            for item in converted.payload_items:
                destination = artifact_root / item.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.source_path, destination)
            artifact = PortableArtifact(
                record=converted.record, storage_path=artifact_root
            )

            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="image.edit", profile="default"),
            )

            self.assertEqual(
                loaded.metadata["execution_status"],
                "experimental_native_mlx_backend",
            )
            self.assertEqual(
                loaded.capability.constraints["num_inference_steps"]["fixed"], 4
            )
            self.assertEqual(
                loaded.capability.constraints["guidance_scale"]["fixed"], 1.0
            )
            with patch(
                "mlxr.families.flux2.adapter.create_image_generator"
            ) as create_image_generator:
                create_image_generator.return_value.generate.return_value = (
                    GeneratedImage(
                        pixels=_tiny_pixels(),
                        seed=7,
                        backend="test",
                        prompt_signature="sig",
                    )
                )
                generate_result = adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="generate",
                        inputs={"prompt": "edit", "images": [{"input_handle": "inp"}]},
                        params={
                            "task": "image.edit",
                            "artifact_format": "png",
                            "resolved_inputs": {
                                "images": [
                                    {
                                        "payload_path": str(bundle_root / "ref.png"),
                                        "strength": 1.0,
                                    }
                                ],
                                "loras": [],
                            },
                            "family_extensions": {},
                            "width": 256,
                            "height": 256,
                            "num_inference_steps": 4,
                            "guidance_scale": 1.0,
                            "seed": 7,
                        },
                    ),
                )
                self.assertEqual(
                    generate_result.metrics["status"],
                    "generated",
                )
                encode_result = adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="encode_output",
                        inputs={},
                        params={
                            "artifact_id": "out_test",
                            "artifact_format": "png",
                            "output_dir": str(Path(tmp_dir) / "out"),
                            "storage_key": "jobs/out_test.png",
                        },
                    ),
                )
                self.assertEqual(len(encode_result.artifacts), 1)
            adapter.unload(loaded)
            self.assertEqual(loaded.metadata["execution_status"], "unloaded")

    def test_base_variant_load_does_not_advertise_distilled_fixed_constraints(
        self,
    ) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-base-4B"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )
            converted = adapter.convert(
                {
                    "bundle": ConversionSource(
                        role="bundle",
                        source_id="src_bundle",
                        source=source_ref,
                        materialization=materialization,
                    )
                },
                ConversionPlan(model_id="flux2-klein-base-4b-local"),
            )
            artifact_root = Path(tmp_dir) / "artifact"
            for item in converted.payload_items:
                destination = artifact_root / item.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.source_path, destination)
            artifact = PortableArtifact(
                record=converted.record, storage_path=artifact_root
            )

            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="image.generate", profile="default"),
            )

        self.assertNotIn("num_inference_steps", loaded.capability.constraints)
        self.assertNotIn("guidance_scale", loaded.capability.constraints)

    def test_policy_tracks_apache_vs_noncommercial_rows(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-4B"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )
            artifact = adapter.convert(
                {
                    "bundle": ConversionSource(
                        role="bundle",
                        source_id="src_bundle",
                        source=source_ref,
                        materialization=materialization,
                    )
                },
                ConversionPlan(model_id="flux2-klein-4b-local"),
            )

        self.assertEqual(artifact.record.capability.policy.license, "apache-2.0")

    def test_generate_rejects_prompt_upsampling_until_implemented(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-9B"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )
            converted = adapter.convert(
                {
                    "bundle": ConversionSource(
                        role="bundle",
                        source_id="src_bundle",
                        source=source_ref,
                        materialization=materialization,
                    )
                },
                ConversionPlan(model_id="flux2-klein-9b-local"),
            )
            artifact_root = Path(tmp_dir) / "artifact"
            for item in converted.payload_items:
                destination = artifact_root / item.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.source_path, destination)
            artifact = PortableArtifact(
                record=converted.record, storage_path=artifact_root
            )
            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="image.generate", profile="default"),
            )

            with self.assertRaisesRegex(
                ValueError, "prompt upsampling is not implemented"
            ):
                adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="generate",
                        inputs={"prompt": "portrait"},
                        params={
                            "task": "image.generate",
                            "family_extensions": {"prompt_upsampling_mode": "local"},
                        },
                    ),
                )

    def test_generate_requires_real_component_configs_without_mocking(self) -> None:
        adapter = Flux2FamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_flux_bundle(
                Path(tmp_dir), variant_dir_name="FLUX.2-klein-9B"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="flux2",
            )
            resolved = provider.resolve(source_ref)
            materialization = provider.fetch(
                resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
            )
            converted = adapter.convert(
                {
                    "bundle": ConversionSource(
                        role="bundle",
                        source_id="src_bundle",
                        source=source_ref,
                        materialization=materialization,
                    )
                },
                ConversionPlan(model_id="flux2-klein-9b-local"),
            )
            artifact_root = Path(tmp_dir) / "artifact"
            for item in converted.payload_items:
                destination = artifact_root / item.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.source_path, destination)
            artifact = PortableArtifact(
                record=converted.record, storage_path=artifact_root
            )
            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="image.generate", profile="default"),
            )

            with self.assertRaisesRegex((KeyError, ValueError, OSError), ".+"):
                adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="generate",
                        inputs={"prompt": "portrait"},
                        params={
                            "task": "image.generate",
                            "resolved_inputs": {"images": [], "loras": []},
                        },
                    ),
                )

    def test_encode_output_requires_generated_image(self) -> None:
        adapter = Flux2FamilyAdapter()
        with self.assertRaisesRegex(
            ValueError, "requires generate to run successfully first"
        ):
            adapter.run_stage(
                _loaded_handle_for_encode(),
                ExecutionStage(
                    stage_id="encode_output",
                    inputs={},
                    params={
                        "artifact_id": "out_test",
                        "artifact_format": "png",
                        "output_dir": "/tmp",
                        "storage_key": "jobs/out_test.png",
                    },
                ),
            )


def provider_source(variant_dir_name: str) -> ResolvedSource:
    return ResolvedSource(
        provider="local",
        locator={"path": f"/tmp/{variant_dir_name}"},
        files=[SourceFileRecord(path="model_index.json")],
    )


def _tiny_pixels() -> object:
    import numpy as np

    return np.full((8, 8, 3), 255, dtype=np.uint8)


def _loaded_handle_for_encode():
    from mlxr.core.runtime import LoadedModelHandle
    from mlxr.core.schemas import CapabilityDescriptor

    capability = CapabilityDescriptor(
        model_id="flux2-klein-9b-local",
        artifact_digest="sha256:test",
        family="flux2",
        family_variant="flux.2-klein-9b",
        scheduler_class="image_diffusion",
        tasks=["image.generate"],
    )
    return LoadedModelHandle(
        model_id="flux2-klein-9b-local",
        family="flux2",
        artifact_digest="sha256:test",
        capability=capability,
        metadata={},
    )


if __name__ == "__main__":
    unittest.main()

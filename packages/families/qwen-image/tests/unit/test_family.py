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
from mlxr.families.qwen_image import QwenImageFamilyAdapter
from mlxr.families.qwen_image.generation import GeneratedImage
from mlxr.families.qwen_image.prompt_encoding import PromptEncodingResult


def make_local_qwen_bundle(
    root: Path, *, variant_dir_name: str, edit_bundle: bool
) -> Path:
    bundle_root = root / variant_dir_name
    files = {
        "model_index.json": b"{}",
        "transformer/config.json": json.dumps(
            {
                "attention_head_dim": 128,
                "guidance_embeds": False,
                "in_channels": 64,
                "joint_attention_dim": 3584,
                "num_attention_heads": 24,
                "num_layers": 60,
                "out_channels": 16,
                "patch_size": 2,
                "pooled_projection_dim": 768,
            }
        ).encode("utf-8"),
        "transformer/model.safetensors": b"transformer",
        "vae/config.json": json.dumps(
            {
                "latents_mean": [0.1] * 16,
                "latents_std": [1.0] * 16,
                "temperal_downsample": [False, True, True],
                "z_dim": 16,
            }
        ).encode("utf-8"),
        "vae/diffusion_pytorch_model.safetensors": b"vae",
        "text_encoder/config.json": b"{}",
        "text_encoder/model.safetensors": b"text-encoder",
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}",
        "scheduler/scheduler_config.json": json.dumps(
            {
                "base_image_seq_len": 256,
                "base_shift": 0.5,
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": 0.9,
                "num_train_timesteps": 1000,
                "shift_terminal": 0.02,
                "time_shift_type": "exponential",
                "use_dynamic_shifting": True,
            }
        ).encode("utf-8"),
    }
    if edit_bundle:
        files["processor/preprocessor_config.json"] = b"{}"
    for relative_path, payload in files.items():
        destination = bundle_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return bundle_root


class QwenImageFamilyTests(unittest.TestCase):
    def test_inspect_source_requires_model_index(self) -> None:
        adapter = QwenImageFamilyAdapter()
        source = ResolvedSource(
            provider="local",
            locator={"path": "/tmp/Qwen-Image-2512"},
            pinned_ref=None,
            family_hint="qwen_image",
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

    def test_inspect_source_recognizes_primary_generate_bundle(self) -> None:
        adapter = QwenImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_qwen_bundle(
                Path(tmp_dir), variant_dir_name="Qwen-Image-2512", edit_bundle=False
            )
            resolved = provider.resolve(
                SourceRef(
                    provider="local",
                    locator={"path": str(bundle_root)},
                    family_hint="qwen_image",
                )
            )

        inspection = adapter.inspect_source(resolved)

        self.assertEqual(inspection.variant, "qwen-image-2512")
        self.assertEqual(inspection.tasks, ("image.generate",))

    def test_inspect_source_honors_explicit_variant_hint(self) -> None:
        adapter = QwenImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_qwen_bundle(
                Path(tmp_dir),
                variant_dir_name="scratch-download-dir",
                edit_bundle=False,
            )
            resolved = provider.resolve(
                SourceRef(
                    provider="local",
                    locator={
                        "path": str(bundle_root),
                        "variant": "qwen-image-2512",
                    },
                    family_hint="qwen_image",
                )
            )

        inspection = adapter.inspect_source(resolved)

        self.assertEqual(inspection.variant, "qwen-image-2512")

    def test_inspect_source_recognizes_primary_edit_bundle(self) -> None:
        adapter = QwenImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_qwen_bundle(
                Path(tmp_dir),
                variant_dir_name="Qwen-Image-Edit-2511",
                edit_bundle=True,
            )
            resolved = provider.resolve(
                SourceRef(
                    provider="local",
                    locator={"path": str(bundle_root)},
                    family_hint="qwen_image",
                )
            )

        inspection = adapter.inspect_source(resolved)

        self.assertEqual(inspection.variant, "qwen-image-edit-2511")
        self.assertEqual(inspection.tasks, ("image.edit",))

    def test_convert_creates_componentized_artifact_for_primary_edit_row(self) -> None:
        adapter = QwenImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_qwen_bundle(
                Path(tmp_dir),
                variant_dir_name="Qwen-Image-Edit-2511",
                edit_bundle=True,
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="qwen_image",
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
                ConversionPlan(model_id="qwen-image-edit-local"),
            )

        self.assertEqual(artifact.record.family_variant, "qwen-image-edit-2511")
        self.assertEqual(artifact.record.capability.tasks, ["image.edit"])
        self.assertIn("processor", {c.role for c in artifact.record.components})
        self.assertEqual(artifact.record.capability.artifacts_out, ["png", "jpg"])

    def test_fetch_policy_rejects_non_bundle_role(self) -> None:
        adapter = QwenImageFamilyAdapter()
        with self.assertRaisesRegex(ValueError, "single 'bundle' source role"):
            adapter.fetch_policy_for_conversion(
                "checkpoint",
                provider_source("Qwen-Image-2512"),
            )

    def test_load_requires_materialized_storage_path(self) -> None:
        adapter = QwenImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_qwen_bundle(
                Path(tmp_dir), variant_dir_name="Qwen-Image-2512", edit_bundle=False
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="qwen_image",
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
                ConversionPlan(model_id="qwen-image-local"),
            )
            artifact = PortableArtifact(record=converted.record, storage_path=None)

            with self.assertRaisesRegex(ValueError, "requires a materialized artifact"):
                adapter.load(
                    artifact,
                    ExecutionProfile(task="image.generate", profile="default"),
                )

    def test_load_rejects_missing_component_directory(self) -> None:
        adapter = QwenImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_qwen_bundle(
                Path(tmp_dir), variant_dir_name="Qwen-Image-2512", edit_bundle=False
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="qwen_image",
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
                ConversionPlan(model_id="qwen-image-local"),
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

    def test_inspect_source_rejects_blocked_variant(self) -> None:
        adapter = QwenImageFamilyAdapter()
        with self.assertRaisesRegex(ValueError, "not yet enabled"):
            adapter.inspect_source(provider_source("Qwen-Image-Layered"))

    def test_load_and_run_stage_generates_runtime_managed_artifact(self) -> None:
        adapter = QwenImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_qwen_bundle(
                Path(tmp_dir),
                variant_dir_name="Qwen-Image-Edit-2511",
                edit_bundle=True,
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="qwen_image",
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
                ConversionPlan(model_id="qwen-image-edit-local"),
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
                "owned_mlx_backend_required",
            )
            with (
                patch(
                    "mlxr.families.qwen_image.adapter.create_prompt_encoder"
                ) as create_prompt_encoder,
                patch(
                    "mlxr.families.qwen_image.adapter.create_image_generator"
                ) as create_image_generator,
            ):
                create_prompt_encoder.return_value.encode.return_value = (
                    PromptEncodingResult(
                        prompt_embeddings=object(),
                        prompt_attention_mask=object(),
                        prompt_text="edit",
                        token_count=7,
                        sequence_length=7,
                        hidden_size=3584,
                        negative_prompt_text="blurry",
                    )
                )
                create_image_generator.return_value.generate.return_value = (
                    GeneratedImage(
                        pixels=_tiny_pixels(),
                        seed=7,
                        backend="test",
                        prompt_signature="sig",
                    )
                )
                prompt_result = adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="prompt_encode",
                        inputs={"prompt": "edit", "negative_prompt": "blurry"},
                        params={},
                    ),
                )
                self.assertEqual(prompt_result.metrics["status"], "encoded")
                generate_result = adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="generate",
                        inputs={"prompt": "edit", "negative_prompt": "blurry"},
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
                            "num_inference_steps": 8,
                            "guidance_scale": 2.5,
                            "seed": 7,
                        },
                    ),
                )
                self.assertEqual(generate_result.metrics["status"], "generated")
                create_prompt_encoder.return_value.close.assert_called_once()
                output_result = adapter.run_stage(
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
                self.assertEqual(len(output_result.artifacts), 1)
            adapter.unload(loaded)
            self.assertEqual(loaded.metadata["execution_status"], "unloaded")

    def test_prompt_encode_stage_uses_prompt_encoder_and_stores_context(self) -> None:
        adapter = QwenImageFamilyAdapter()
        fake_context = PromptEncodingResult(
            prompt_embeddings=object(),
            prompt_attention_mask=object(),
            prompt_text="poster",
            token_count=9,
            sequence_length=9,
            hidden_size=3584,
        )
        loaded = _loaded_handle_for_prompt()

        with patch("mlxr.families.qwen_image.adapter.create_prompt_encoder") as factory:
            factory.return_value.encode.return_value = fake_context
            result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="prompt_encode",
                    inputs={"prompt": "poster"},
                    params={},
                ),
            )

        self.assertEqual(result.metrics["status"], "encoded")
        self.assertIs(loaded.metadata["prompt_context"], fake_context)
        self.assertEqual(loaded.metadata["execution_status"], "prompt_encode_ready")

    def test_generate_rejects_prompt_enhancement_until_implemented(self) -> None:
        adapter = QwenImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_qwen_bundle(
                Path(tmp_dir), variant_dir_name="Qwen-Image-2512", edit_bundle=False
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="qwen_image",
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
                ConversionPlan(model_id="qwen-image-local"),
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
                ValueError, "prompt enhancement is not implemented"
            ):
                adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="generate",
                        inputs={"prompt": "poster"},
                        params={
                            "task": "image.generate",
                            "family_extensions": {"prompt_enhance_mode": "dashscope"},
                        },
                    ),
                )

    def test_generate_surfaces_backend_errors(self) -> None:
        adapter = QwenImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_qwen_bundle(
                Path(tmp_dir), variant_dir_name="Qwen-Image-2512", edit_bundle=False
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="qwen_image",
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
                ConversionPlan(model_id="qwen-image-local"),
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
            loaded.metadata["prompt_context"] = PromptEncodingResult(
                prompt_embeddings=object(),
                prompt_attention_mask=object(),
                prompt_text="poster",
                token_count=7,
                sequence_length=7,
                hidden_size=3584,
            )

            with patch(
                "mlxr.families.qwen_image.adapter.create_image_generator"
            ) as create_image_generator:
                create_image_generator.side_effect = ValueError("backend boom")
                with self.assertRaisesRegex(ValueError, "backend boom"):
                    adapter.run_stage(
                        loaded,
                        ExecutionStage(
                            stage_id="generate",
                            inputs={"prompt": "poster"},
                            params={
                                "task": "image.generate",
                                "resolved_inputs": {"images": [], "loras": []},
                            },
                        ),
                    )

    def test_encode_output_requires_generated_image(self) -> None:
        adapter = QwenImageFamilyAdapter()
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


def _loaded_handle_for_prompt():
    from mlxr.core.runtime import LoadedModelHandle
    from mlxr.core.schemas import CapabilityDescriptor

    capability = CapabilityDescriptor(
        model_id="qwen-image-local",
        artifact_digest="sha256:test",
        family="qwen_image",
        family_variant="qwen-image-2512",
        scheduler_class="image_diffusion",
        tasks=["image.generate"],
    )
    return LoadedModelHandle(
        model_id="qwen-image-local",
        family="qwen_image",
        artifact_digest="sha256:test",
        capability=capability,
        metadata={
            "component_paths": {
                "text_encoder": "/tmp/text_encoder",
                "tokenizer": "/tmp/tokenizer",
            }
        },
    )


def _loaded_handle_for_encode():
    from mlxr.core.runtime import LoadedModelHandle
    from mlxr.core.schemas import CapabilityDescriptor

    capability = CapabilityDescriptor(
        model_id="qwen-image-local",
        artifact_digest="sha256:test",
        family="qwen_image",
        family_variant="qwen-image-2512",
        scheduler_class="image_diffusion",
        tasks=["image.generate"],
    )
    return LoadedModelHandle(
        model_id="qwen-image-local",
        family="qwen_image",
        artifact_digest="sha256:test",
        capability=capability,
        metadata={},
    )


if __name__ == "__main__":
    unittest.main()

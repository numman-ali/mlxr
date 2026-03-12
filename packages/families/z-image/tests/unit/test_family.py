from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mlxr.core.runtime import (
    ConversionPlan,
    ConversionSource,
    ExecutionProfile,
    ExecutionStage,
    LocalFileProviderAdapter,
    PortableArtifact,
)
from mlxr.core.schemas import JobOutputPolicy, ModelRecord, SourceRef, WorkflowIntent
from mlxr.core.workflows import WorkflowPlanningContext
from mlxr.families.z_image import ZImageFamilyAdapter, ZImageWorkflowStrategy
from mlxr.families.z_image.generation import GeneratedImage
from mlxr.families.z_image.prompt_encoding import PromptEncodingResult


def make_local_z_image_bundle(root: Path, *, variant_dir_name: str) -> Path:
    bundle_root = root / variant_dir_name
    files = {
        "model_index.json": b"{}",
        "transformer/config.json": b"{}",
        "transformer/model.safetensors": b"transformer",
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


class ZImageFamilyTests(unittest.TestCase):
    def test_inspect_source_recognizes_diffusers_bundle(self) -> None:
        adapter = ZImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_z_image_bundle(
                Path(tmp_dir), variant_dir_name="Z-Image-Turbo"
            )
            resolved = provider.resolve(
                SourceRef(
                    provider="local",
                    locator={"path": str(bundle_root)},
                    family_hint="z_image",
                )
            )

        inspection = adapter.inspect_source(resolved)

        self.assertEqual(inspection.family, "z_image")
        self.assertEqual(inspection.variant, "z-image-turbo")
        self.assertEqual(inspection.tasks, ("image.generate",))
        self.assertEqual(inspection.scheduler_class, "image_diffusion")

    def test_convert_creates_componentized_portable_artifact(self) -> None:
        adapter = ZImageFamilyAdapter()
        provider = LocalFileProviderAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_z_image_bundle(
                Path(tmp_dir), variant_dir_name="Z-Image"
            )
            source_ref = SourceRef(
                provider="local",
                locator={"path": str(bundle_root)},
                family_hint="z_image",
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
                ConversionPlan(model_id="z-image-local"),
            )

        self.assertEqual(artifact.record.family, "z_image")
        self.assertEqual(artifact.record.family_variant, "z-image")
        self.assertEqual(
            {component.role for component in artifact.record.components},
            {"transformer", "vae", "text_encoder", "tokenizer", "scheduler"},
        )
        self.assertEqual(artifact.record.capability.tasks, ["image.generate"])
        self.assertEqual(artifact.record.capability.artifacts_out, ["png", "jpg"])
        self.assertEqual(
            artifact.record.metadata["blocked_variants"],
            ["z-image-omni-base", "z-image-edit"],
        )
        self.assertGreaterEqual(len(artifact.payload_items), 9)

    def test_load_marks_runtime_execution_ready(self) -> None:
        adapter = ZImageFamilyAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_z_image_bundle(
                Path(tmp_dir), variant_dir_name="Z-Image"
            )
            converted = build_portable_artifact_for_test(
                adapter,
                bundle_root,
                model_id="z-image-local",
            )
            artifact_root = Path(tmp_dir) / "artifact"
            for item in converted.payload_items:
                destination = artifact_root / item.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.source_path, destination)
            artifact = PortableArtifact(
                record=converted.record,
                storage_path=artifact_root,
            )

            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="image.generate", profile="default"),
            )

        self.assertEqual(
            loaded.metadata["execution_status"],
            "experimental_native_mlx_backend",
        )

    def test_workflow_strategy_maps_prompt_only_generation(self) -> None:
        adapter = ZImageFamilyAdapter()
        strategy = ZImageWorkflowStrategy()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_z_image_bundle(
                Path(tmp_dir), variant_dir_name="Z-Image-Turbo"
            )
            artifact = build_portable_artifact_for_test(adapter, bundle_root)

        context = WorkflowPlanningContext(
            model=ModelRecord(
                model_id="z-image-turbo-local",
                family="z_image",
                artifact=artifact.record,
                capability=artifact.record.capability,
            ),
            capability=artifact.record.capability,
        )
        intent = WorkflowIntent(
            model_id="z-image-turbo-local",
            prompt="a cinematic bookstore interior",
            negative_prompt="low quality",
            output=JobOutputPolicy(artifact_format="png"),
            extensions={"z_image": {"cfg_normalization": True, "cfg_truncation": 0.5}},
        )

        plan = strategy.plan(context, intent)
        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(plan.selected_task, "image.generate")
        self.assertEqual(plan.stages[0].params["artifact_format"], "png")
        self.assertEqual(request.inputs["prompt"], "a cinematic bookstore interior")
        self.assertEqual(request.inputs["negative_prompt"], "low quality")
        self.assertEqual(request.extensions["z_image"]["cfg_normalization"], True)
        self.assertEqual(request.extensions["z_image"]["cfg_truncation"], 0.5)

    def test_prompt_encode_stage_uses_prompt_encoder_and_stores_context(self) -> None:
        class FakePromptEncoder:
            def __init__(self) -> None:
                self.closed = False

            def encode(
                self,
                prompt: str,
                *,
                max_length: int = 512,
                negative_prompt: str | None = None,
            ) -> PromptEncodingResult:
                self.last_call = (prompt, max_length, negative_prompt)
                return PromptEncodingResult(
                    prompt_embeddings=("positive-embedding",),
                    prompt_text=prompt,
                    token_count=42,
                    sequence_length=512,
                    hidden_size=2560,
                    config_source="fake://qwen3/config.json",
                    negative_prompt_text=negative_prompt,
                    negative_prompt_embeddings=(
                        ("negative-embedding",) if negative_prompt else None
                    ),
                )

            def close(self) -> None:
                self.closed = True

        adapter = ZImageFamilyAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_z_image_bundle(
                Path(tmp_dir), variant_dir_name="Z-Image-Turbo"
            )
            artifact = build_portable_artifact_for_test(adapter, bundle_root)
            artifact_root = Path(tmp_dir) / "artifact"
            for item in artifact.payload_items:
                destination = artifact_root / item.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.source_path, destination)
            loaded = adapter.load(
                PortableArtifact(record=artifact.record, storage_path=artifact_root),
                ExecutionProfile(task="image.generate", profile="default"),
            )
            fake_prompt_encoder = FakePromptEncoder()
            loaded.metadata["prompt_encoder"] = fake_prompt_encoder

            result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="prompt_encode",
                    inputs={
                        "prompt": "a cinematic bookstore interior",
                        "negative_prompt": "low quality",
                    },
                ),
            )

            self.assertEqual(result.metrics["status"], "encoded")
            self.assertEqual(result.metrics["token_count"], 42)
            self.assertTrue(result.metrics["negative_prompt_present"])
            self.assertEqual(loaded.metadata["execution_status"], "prompt_encode_ready")
            self.assertIsNotNone(loaded.metadata["prompt_context"])
            adapter.unload(loaded)
            self.assertTrue(fake_prompt_encoder.closed)

    def test_generate_and_encode_output_use_runtime_image_generator(self) -> None:
        class FakeImageGenerator:
            def __init__(self) -> None:
                self.closed = False

            def generate(
                self,
                *,
                prompt_context: PromptEncodingResult,
                width: int,
                height: int,
                num_inference_steps: int,
                guidance_scale: float,
                cfg_normalization: float = 0.0,
                cfg_truncation: float = 1.0,
                seed: int | None = None,
            ) -> GeneratedImage:
                self.last_call = (
                    prompt_context.prompt_text,
                    width,
                    height,
                    num_inference_steps,
                    guidance_scale,
                    cfg_normalization,
                    cfg_truncation,
                    seed,
                )
                pixels = np.full((height, width, 3), 127, dtype=np.uint8)
                return GeneratedImage(
                    pixels=pixels,
                    seed=0 if seed is None else seed,
                    backend="fake_z_image_backend",
                    prompt_signature="deadbeefcafebabe",
                    metadata={
                        "num_inference_steps": num_inference_steps,
                        "cfg_normalization": cfg_normalization,
                        "cfg_truncation": cfg_truncation,
                    },
                )

            def close(self) -> None:
                self.closed = True

        adapter = ZImageFamilyAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_root = make_local_z_image_bundle(
                Path(tmp_dir), variant_dir_name="Z-Image-Turbo"
            )
            artifact = build_portable_artifact_for_test(adapter, bundle_root)
            artifact_root = Path(tmp_dir) / "artifact"
            for item in artifact.payload_items:
                destination = artifact_root / item.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.source_path, destination)
            loaded = adapter.load(
                PortableArtifact(record=artifact.record, storage_path=artifact_root),
                ExecutionProfile(task="image.generate", profile="default"),
            )
            fake_image_generator = FakeImageGenerator()
            loaded.metadata["image_generator"] = fake_image_generator
            loaded.metadata["prompt_context"] = PromptEncodingResult(
                prompt_embeddings=(object(),),
                prompt_text="a bookstore with warm light",
                token_count=10,
                sequence_length=16,
                hidden_size=2560,
            )

            generate_result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="generate",
                    params={
                        "width": 64,
                        "height": 48,
                        "num_inference_steps": 6,
                        "guidance_scale": 0.0,
                        "family_extensions": {
                            "cfg_normalization": True,
                            "cfg_truncation": 0.5,
                        },
                        "seed": 7,
                    },
                ),
            )
            self.assertEqual(generate_result.metrics["status"], "generated")
            self.assertEqual(generate_result.metrics["width"], 64)
            self.assertEqual(generate_result.metrics["height"], 48)
            self.assertEqual(generate_result.metrics["cfg_normalization"], 1.0)
            self.assertEqual(generate_result.metrics["cfg_truncation"], 0.5)

            encode_result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="encode_output",
                    params={
                        "artifact_id": "out_test",
                        "artifact_format": "png",
                        "output_dir": str(Path(tmp_dir) / "out"),
                        "storage_key": "jobs/job_test/outputs/out_test/out_test.png",
                    },
                ),
            )
            self.assertEqual(len(encode_result.artifacts), 1)
            self.assertEqual(
                encode_result.artifacts[0].metadata["media_type"], "image/png"
            )
            adapter.unload(loaded)
            self.assertTrue(fake_image_generator.closed)


def build_portable_artifact_for_test(
    adapter: ZImageFamilyAdapter,
    bundle_root: Path,
    *,
    model_id: str = "z-image-turbo-local",
) -> PortableArtifact:
    provider = LocalFileProviderAdapter()
    source_ref = SourceRef(
        provider="local",
        locator={"path": str(bundle_root)},
        family_hint="z_image",
    )
    resolved = provider.resolve(source_ref)
    materialization = provider.fetch(
        resolved, adapter.fetch_policy_for_conversion("bundle", resolved)
    )
    return adapter.convert(
        {
            "bundle": ConversionSource(
                role="bundle",
                source_id="src_bundle",
                source=source_ref,
                materialization=materialization,
            )
        },
        ConversionPlan(model_id=model_id),
    )

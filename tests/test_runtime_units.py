from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlxr.core.runtime import (
    ConversionPlan,
    ConversionSource,
    ExecutionProfile,
    ExecutionStage,
    FamilyInspection,
    FetchPolicy,
    LoadedModelHandle,
    PortableArtifact,
    RuntimeHome,
    RuntimeRegistry,
    StageResult,
)
from mlxr.core.schemas import (
    ArtifactHandle,
    CapabilityDescriptor,
    InputHandleRecord,
    JobOutputPolicy,
    JobRecord,
    JobRequest,
    JobState,
    ModelRecord,
    OutputArtifactRecord,
    PortableArtifactComponentRecord,
    PortableArtifactRecord,
    ProvenanceRecord,
    RuntimeEvent,
    RuntimeEventKind,
)
from mlxr.core.server.jobs import JobManager, JobValidationError
from mlxr.core.server.settings import ServerSettings
from mlxr.core.server.store import InputStore, JobStore, OutputStore
from mlxr.core.server.worker import run_job_worker
from mlxr.families.ltx import LTXFamilyAdapter
from mlxr.families.ltx._generation_backend.conditioning import _resolve_padded_shape
from mlxr.families.ltx._generation_backend.reference_imports import (
    _REFERENCE_MLX_VIDEO_ROOT,
)
from safetensors.numpy import save_file

from tests.runtime_test_support import (
    LTX_CHECKPOINT_FILENAME,
    LTX_SPATIAL_UPSAMPLER_FILENAME,
    make_png_bytes,
    make_state,
    patched_ltx_prompt_encoder,
    patched_ltx_video_generator,
)


class RuntimeUnitTests(unittest.TestCase):
    def _artifactized_portable_artifact(
        self, runtime_home: RuntimeHome
    ) -> PortableArtifact:
        artifact_root = runtime_home.artifact_dir(
            "ltx", "ltx-2.3-fast-local", "sha256:test"
        )
        checkpoint_path = (
            artifact_root / "payload" / "checkpoint" / LTX_CHECKPOINT_FILENAME
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        upsampler_path = (
            artifact_root
            / "payload"
            / "spatial_upsampler"
            / LTX_SPATIAL_UPSAMPLER_FILENAME
        )
        upsampler_path.parent.mkdir(parents=True, exist_ok=True)
        upsampler_path.write_text("upsampler", encoding="utf-8")
        text_encoder_dir = artifact_root / "payload" / "text_encoder"
        text_encoder_dir.mkdir(parents=True, exist_ok=True)
        (text_encoder_dir / "config.json").write_text("{}", encoding="utf-8")
        (text_encoder_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (text_encoder_dir / "model-00001-of-00001.safetensors").write_text(
            "weights", encoding="utf-8"
        )
        artifact_storage_key = runtime_home.artifact_storage_key(
            "ltx", "ltx-2.3-fast-local", "sha256:test"
        )
        provenance = ProvenanceRecord(provider="local", locator={"path": "/tmp/model"})
        return PortableArtifact(
            record=PortableArtifactRecord(
                model_id="ltx-2.3-fast-local",
                artifact_digest="sha256:test",
                family="ltx",
                format_version="0.2.0",
                weight_format="source_packaged_fastpath_assets",
                storage_key=artifact_storage_key,
                capability=CapabilityDescriptor(
                    model_id="ltx-2.3-fast-local",
                    artifact_digest="sha256:test",
                    family="ltx",
                    tasks=["video.generate", "video.condition.image"],
                    artifacts_out=["mp4", "wav"],
                    scheduler_class="media_video_dit",
                ),
                provenance=provenance,
                components=[
                    PortableArtifactComponentRecord(
                        role="checkpoint",
                        kind="file",
                        relative_path=f"payload/checkpoint/{LTX_CHECKPOINT_FILENAME}",
                        storage_key=(
                            f"{artifact_storage_key}/payload/checkpoint/"
                            f"{LTX_CHECKPOINT_FILENAME}"
                        ),
                        source_id="src_bundle",
                        provenance=provenance,
                    ),
                    PortableArtifactComponentRecord(
                        role="spatial_upsampler",
                        kind="file",
                        relative_path=(
                            "payload/spatial_upsampler/"
                            f"{LTX_SPATIAL_UPSAMPLER_FILENAME}"
                        ),
                        storage_key=(
                            f"{artifact_storage_key}/payload/spatial_upsampler/"
                            f"{LTX_SPATIAL_UPSAMPLER_FILENAME}"
                        ),
                        source_id="src_bundle",
                        provenance=provenance,
                    ),
                    PortableArtifactComponentRecord(
                        role="text_encoder",
                        kind="directory",
                        relative_path="payload/text_encoder",
                        storage_key=f"{artifact_storage_key}/payload/text_encoder",
                        source_id="src_bundle",
                        provenance=provenance,
                    ),
                ],
            ),
            storage_path=artifact_root,
        )

    def _image_portable_artifact(self, runtime_home: RuntimeHome) -> PortableArtifact:
        artifact_storage_key = runtime_home.artifact_storage_key(
            "z_image", "z-image-turbo-local", "sha256:image-test"
        )
        provenance = ProvenanceRecord(provider="local", locator={"path": "/tmp/model"})
        return PortableArtifact(
            record=PortableArtifactRecord(
                model_id="z-image-turbo-local",
                artifact_digest="sha256:image-test",
                family="z_image",
                family_variant="z-image-turbo",
                format_version="0.1.0",
                weight_format="diffusers_component_bundle",
                storage_key=artifact_storage_key,
                capability=CapabilityDescriptor(
                    model_id="z-image-turbo-local",
                    artifact_digest="sha256:image-test",
                    family="z_image",
                    family_variant="z-image-turbo",
                    tasks=["image.generate"],
                    artifacts_out=["png", "jpg"],
                    scheduler_class="image_diffusion",
                    constraints={
                        "width": {"multiple_of": 16},
                        "height": {"multiple_of": 16},
                    },
                    metadata={
                        "stage_ids": [
                            "prompt_encode",
                            "generate",
                            "encode_output",
                        ]
                    },
                ),
                provenance=provenance,
            ),
            storage_path=runtime_home.artifact_dir(
                "z_image",
                "z-image-turbo-local",
                "sha256:image-test",
            ),
        )

    def test_ltx_adapter_without_runtime_state_reports_placeholder_metrics(
        self,
    ) -> None:
        adapter = LTXFamilyAdapter()
        capability = CapabilityDescriptor(
            model_id="ltx-2.3-fast-local",
            artifact_digest="sha256:test",
            family="ltx",
            scheduler_class="media_video_dit",
        )
        loaded = LoadedModelHandle(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            artifact_digest="sha256:test",
            capability=capability,
            metadata={"task": "video.generate"},
        )

        with tempfile.TemporaryDirectory():
            prompt_result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="prompt_encode",
                    inputs={"prompt": "cinematic fox in snow"},
                    params={"simulate_delay_seconds": 0.0},
                ),
            )
            self.assertEqual(prompt_result.metrics["status"], "placeholder")

    def test_ltx_adapter_prompt_encode_stores_context_and_unload_clears_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._artifactized_portable_artifact(runtime_home)
            adapter = LTXFamilyAdapter()
            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="video.generate", profile="bf16"),
            )

            with (
                patched_ltx_prompt_encoder(token_count=11) as encoders,
                patched_ltx_video_generator(include_audio=True) as generators,
            ):
                prompt_result = adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="prompt_encode",
                        inputs={"prompt": "cinematic fox in snow"},
                        params={"simulate_delay_seconds": 0.0},
                    ),
                )
                self.assertEqual(prompt_result.metrics["status"], "encoded")
                self.assertEqual(prompt_result.metrics["token_count"], 11)
                self.assertEqual(
                    prompt_result.metrics["video_context_shape"], [1, 1024, 3840]
                )
                self.assertEqual(
                    prompt_result.metrics["context_representation"], "post_connector"
                )
                self.assertTrue(prompt_result.metrics["caption_proj_before_connector"])
                self.assertEqual(prompt_result.metrics["rope_type"], "split")
                self.assertTrue(prompt_result.metrics["double_precision_rope"])
                self.assertEqual(prompt_result.metrics["transformer_context_dim"], 3840)

                generate_result = adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="generate",
                        inputs={"prompt": "cinematic fox in snow"},
                        params={
                            "task": "video.generate",
                            "width": 96,
                            "height": 64,
                            "num_frames": 9,
                            "fps": 12,
                            "seed": 11,
                            "resolved_inputs": {"images": []},
                            "simulate_delay_seconds": 0.0,
                        },
                    ),
                )
                self.assertEqual(generate_result.metrics["status"], "generated")
                self.assertEqual(generate_result.metrics["frames_generated"], 9)
                self.assertEqual(
                    generate_result.metrics["backend"],
                    "ltx_test_distilled_generator",
                )
                self.assertEqual(
                    generate_result.metrics["pipeline_kind"], "distilled_two_stage"
                )
                self.assertEqual(generate_result.metrics["output_width"], 96)
                self.assertEqual(generate_result.metrics["output_height"], 64)
                self.assertTrue(generate_result.metrics["audio_present"])
                self.assertEqual(generate_result.metrics["audio_sample_rate"], 24000)
                self.assertEqual(generate_result.metrics["audio_channels"], 2)

                with tempfile.TemporaryDirectory() as output_dir:
                    encode_result = adapter.run_stage(
                        loaded,
                        ExecutionStage(
                            stage_id="encode_output",
                            inputs={"prompt": "cinematic fox in snow"},
                            params={
                                "artifact_id": "out_job_1",
                                "artifact_format": "mp4",
                                "output_dir": output_dir,
                                "storage_key": (
                                    "jobs/job_1/outputs/out_job_1/out_job_1.mp4"
                                ),
                                "simulate_delay_seconds": 0.0,
                            },
                        ),
                    )
                    output_path = Path(output_dir) / "out_job_1.mp4"
                    self.assertTrue(output_path.exists())
                    self.assertIn(b"ftyp", output_path.read_bytes()[:32])
                    ffprobe = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-print_format",
                            "json",
                            "-show_streams",
                            str(output_path),
                        ],
                        capture_output=True,
                        check=True,
                        text=True,
                    )
                    probe_payload = json.loads(ffprobe.stdout)
                    streams = probe_payload["streams"]
                    video_stream = next(
                        stream
                        for stream in streams
                        if stream.get("codec_type") == "video"
                    )
                    self.assertEqual(video_stream["codec_name"], "h264")
                    self.assertEqual(encode_result.metrics["status"], "encoded")
                    self.assertTrue(encode_result.metrics["audio_present"])
                    self.assertEqual(encode_result.metrics["audio_sample_rate"], 24000)
                    self.assertEqual(encode_result.metrics["audio_channels"], 2)

                adapter.unload(loaded)
                self.assertEqual(len(encoders), 1)
                self.assertTrue(encoders[0].closed)
                self.assertEqual(len(generators), 1)
                self.assertTrue(generators[0].closed)

    def test_ltx_adapter_encode_output_supports_wav_when_audio_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._artifactized_portable_artifact(runtime_home)
            adapter = LTXFamilyAdapter()
            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="video.generate", profile="bf16"),
            )

            with (
                patched_ltx_prompt_encoder(token_count=11),
                patched_ltx_video_generator(include_audio=True),
            ):
                adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="prompt_encode",
                        inputs={"prompt": "cinematic fox in snow"},
                        params={"simulate_delay_seconds": 0.0},
                    ),
                )
                adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="generate",
                        inputs={"prompt": "cinematic fox in snow"},
                        params={
                            "task": "video.generate",
                            "width": 96,
                            "height": 64,
                            "num_frames": 9,
                            "fps": 12,
                            "seed": 11,
                            "resolved_inputs": {"images": []},
                            "simulate_delay_seconds": 0.0,
                        },
                    ),
                )

                with tempfile.TemporaryDirectory() as output_dir:
                    encode_result = adapter.run_stage(
                        loaded,
                        ExecutionStage(
                            stage_id="encode_output",
                            inputs={"prompt": "cinematic fox in snow"},
                            params={
                                "artifact_id": "out_job_1",
                                "artifact_format": "wav",
                                "output_dir": output_dir,
                                "storage_key": (
                                    "jobs/job_1/outputs/out_job_1/out_job_1.wav"
                                ),
                                "simulate_delay_seconds": 0.0,
                            },
                        ),
                    )
                    output_path = Path(output_dir) / "out_job_1.wav"
                    self.assertTrue(output_path.exists())
                    self.assertEqual(output_path.read_bytes()[:4], b"RIFF")
                    self.assertEqual(encode_result.metrics["artifact_format"], "wav")
                    self.assertTrue(encode_result.metrics["audio_present"])
                    self.assertEqual(encode_result.metrics["audio_sample_rate"], 24000)
                    self.assertEqual(encode_result.metrics["audio_channels"], 2)

    def test_ltx_adapter_prompt_encode_accepts_negative_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._artifactized_portable_artifact(runtime_home)
            adapter = LTXFamilyAdapter()
            with patched_ltx_prompt_encoder() as encoders:
                loaded = adapter.load(
                    artifact,
                    ExecutionProfile(task="video.generate", profile="bf16"),
                )

                result = adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="prompt_encode",
                        inputs={
                            "prompt": "cinematic fox in snow",
                            "negative_prompt": "blurry",
                        },
                        params={"simulate_delay_seconds": 0.0},
                    ),
                )
                self.assertTrue(result.metrics["negative_prompt_present"])
                self.assertEqual(len(encoders), 1)

    def test_ltx_adapter_condition_inputs_prepares_resolved_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._artifactized_portable_artifact(runtime_home)
            adapter = LTXFamilyAdapter()
            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="video.condition.image", profile="bf16"),
            )
            image_path = Path(tmp_dir) / "conditioning.png"
            image_path.write_bytes(make_png_bytes())

            result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="condition_inputs",
                    inputs={"images": [{"input_handle": "inp_1"}]},
                    params={
                        "task": "video.condition.image",
                        "num_frames": 9,
                        "resolved_inputs": {
                            "images": [
                                {
                                    "input_handle": "inp_1",
                                    "payload_path": str(image_path),
                                    "frame_index": 0,
                                    "strength": 1.0,
                                    "media_type": "image/png",
                                    "filename": "conditioning.png",
                                }
                            ]
                        },
                        "simulate_delay_seconds": 0.0,
                    },
                ),
            )
            self.assertEqual(result.metrics["status"], "prepared")
            self.assertEqual(result.metrics["conditioning_count"], 1)

    def test_ltx_adapter_condition_inputs_prepares_retake_video_and_window(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._artifactized_portable_artifact(runtime_home)
            adapter = LTXFamilyAdapter()
            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="video.retake", profile="bf16"),
            )
            video_path = Path(tmp_dir) / "source.mp4"
            video_path.write_bytes(b"mp4")

            result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="condition_inputs",
                    inputs={"videos": [{"input_handle": "vid_1"}]},
                    params={
                        "task": "video.retake",
                        "window_start_seconds": 1.25,
                        "window_end_seconds": 2.75,
                        "regenerate_audio": False,
                        "resolved_inputs": {
                            "videos": [
                                {
                                    "input_handle": "vid_1",
                                    "payload_path": str(video_path),
                                    "strength": 1.0,
                                    "media_type": "video/mp4",
                                    "filename": "source.mp4",
                                }
                            ]
                        },
                        "simulate_delay_seconds": 0.0,
                    },
                ),
            )
            self.assertEqual(result.metrics["status"], "prepared")
            self.assertEqual(result.metrics["video_input_count"], 1)
            self.assertTrue(result.metrics["retake_enabled"])

    def test_ltx_adapter_condition_inputs_prepares_video_and_lora_for_ic_lora(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._artifactized_portable_artifact(runtime_home)
            adapter = LTXFamilyAdapter()
            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="video.condition.video", profile="bf16"),
            )
            video_path = Path(tmp_dir) / "reference.mp4"
            lora_path = Path(tmp_dir) / "control.safetensors"
            video_path.write_bytes(b"mp4")
            lora_path.write_bytes(b"lora")

            result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="condition_inputs",
                    inputs={
                        "videos": [{"input_handle": "vid_1"}],
                        "loras": [{"input_handle": "lora_1"}],
                    },
                    params={
                        "task": "video.condition.video",
                        "resolved_inputs": {
                            "videos": [
                                {
                                    "input_handle": "vid_1",
                                    "payload_path": str(video_path),
                                    "strength": 0.8,
                                    "media_type": "video/mp4",
                                    "filename": "reference.mp4",
                                }
                            ],
                            "loras": [
                                {
                                    "input_handle": "lora_1",
                                    "payload_path": str(lora_path),
                                    "strength": 0.6,
                                    "media_type": "application/x-safetensors",
                                    "filename": "control.safetensors",
                                }
                            ],
                        },
                        "simulate_delay_seconds": 0.0,
                    },
                ),
            )
            self.assertEqual(result.metrics["status"], "prepared")
            self.assertEqual(result.metrics["video_input_count"], 1)
            self.assertEqual(result.metrics["lora_input_count"], 1)

    def test_ltx_adapter_generate_requires_prompt_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._artifactized_portable_artifact(runtime_home)
            adapter = LTXFamilyAdapter()
            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="video.generate", profile="bf16"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "requires prompt_encode to run successfully first",
            ):
                adapter.run_stage(
                    loaded,
                    ExecutionStage(
                        stage_id="generate",
                        inputs={"prompt": "cinematic fox in snow"},
                        params={
                            "task": "video.generate",
                            "width": 96,
                            "height": 64,
                            "num_frames": 9,
                            "resolved_inputs": {"images": []},
                            "simulate_delay_seconds": 0.0,
                        },
                    ),
                )

    def test_ltx_adapter_prompt_encode_surfaces_backend_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._artifactized_portable_artifact(runtime_home)
            adapter = LTXFamilyAdapter()
            loaded = adapter.load(
                artifact,
                ExecutionProfile(task="video.generate", profile="bf16"),
            )

            with patch(
                "mlxr.families.ltx.adapter.create_prompt_encoder",
                side_effect=RuntimeError("Local MLX LTX prompt encoder unavailable."),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Local MLX LTX prompt encoder unavailable",
                ):
                    adapter.run_stage(
                        loaded,
                        ExecutionStage(
                            stage_id="prompt_encode",
                            inputs={"prompt": "cinematic fox in snow"},
                            params={"simulate_delay_seconds": 0.0},
                        ),
                    )

    def test_ltx_prompt_backend_builds_v2_layout_from_current_config(self) -> None:
        from mlxr.families.ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        encoder = backend._MLXLTXPromptEncoder(
            checkpoint_path=Path("/tmp/ltx-checkpoint.safetensors"),
            text_encoder_path=Path("/tmp/gemma"),
        )
        encoder.language_model = SimpleNamespace(  # type: ignore[assignment]
            config=SimpleNamespace(hidden_size=3840, num_hidden_layers=48)
        )

        transformer_config = {
            "connector_num_attention_heads": 32,
            "connector_attention_head_dim": 128,
            "audio_connector_num_attention_heads": 32,
            "audio_connector_attention_head_dim": 64,
            "connector_num_layers": 8,
            "connector_num_learnable_registers": 128,
            "connector_positional_embedding_max_pos": [4096],
            "cross_attention_dim": 4096,
            "rope_type": "split",
            "frequencies_precision": "float64",
            "connector_apply_gated_attention": True,
            "apply_gated_attention": True,
            "cross_attention_adaln": True,
            "caption_proj_before_connector": True,
            "caption_projection_first_linear": False,
            "caption_proj_input_norm": False,
            "caption_projection_second_linear": False,
        }

        with patch.object(
            encoder,
            "_resolve_transformer_config",
            return_value=backend._TransformerConfigResolution(
                config=transformer_config,
                source="memory://ltx-v2-config",
            ),
        ):
            layout = encoder._prompt_layout()

        self.assertEqual(layout.version, "v2")
        self.assertEqual(layout.flat_dim, 3840 * 49)
        self.assertEqual(layout.video_dim, 4096)
        self.assertEqual(layout.audio_dim, 2048)
        self.assertEqual(layout.transformer_context_dim, 4096)
        self.assertEqual(layout.video_layers, 8)
        self.assertEqual(layout.rope_type, "split")
        self.assertTrue(layout.double_precision_rope)
        self.assertTrue(layout.connector_apply_gated_attention)
        self.assertTrue(layout.caption_proj_before_connector)
        self.assertTrue(layout.transformer_apply_gated_attention)
        self.assertTrue(layout.transformer_cross_attention_adaln)
        self.assertEqual(layout.config_source, "memory://ltx-v2-config")

    def test_ltx_prompt_backend_binary_mask_keeps_only_valid_tokens(self) -> None:
        from mlxr.families.ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        encoded = mx.ones((1, 4, 2), dtype=mx.float32)
        additive_mask = mx.array([[[[0.0, 0.0, -10000.0, -10000.0]]]], dtype=mx.float32)

        masked, binary_mask = backend._to_binary_mask(encoded, additive_mask)

        self.assertEqual(binary_mask.tolist(), [[1, 1, 0, 0]])
        self.assertEqual(masked[:, 2:, :].tolist(), [[[0.0, 0.0], [0.0, 0.0]]])

    def test_ltx_prompt_backend_builds_left_padded_gemma_masks(self) -> None:
        from mlxr.families.ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        hidden = mx.zeros((1, 8, 4), dtype=mx.bfloat16)
        attention_mask = mx.array([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=mx.int32)

        @dataclass
        class _MaskConfig:
            hidden_size: int
            num_hidden_layers: int
            sliding_window: int
            sliding_window_pattern: int

        config = _MaskConfig(
            hidden_size=32,
            num_hidden_layers=2,
            sliding_window=4,
            sliding_window_pattern=6,
        )

        global_mask, local_mask = backend._gemma_attention_masks(
            hidden=hidden,
            attention_mask=attention_mask,
            cache=[None] * 6,
            config=config,
        )
        self.assertIsNotNone(global_mask)
        self.assertIsNotNone(local_mask)
        self.assertIsInstance(global_mask, mx.array)
        self.assertIsInstance(local_mask, mx.array)
        assert isinstance(global_mask, mx.array)
        assert isinstance(local_mask, mx.array)

        self.assertEqual(global_mask.shape, (1, 1, 8, 8))
        self.assertEqual(local_mask.shape, (1, 1, 8, 8))
        self.assertEqual(
            global_mask[0, 0, 3].tolist(),
            [False, False, False, True, False, False, False, False],
        )
        self.assertEqual(
            global_mask[0, 0, 7].tolist(),
            [False, False, False, True, True, True, True, True],
        )
        self.assertEqual(
            local_mask[0, 0, 7].tolist(),
            [False, False, False, False, True, True, True, True],
        )

    def test_ltx_prompt_backend_connector_rope_cache_uses_official_shapes(self) -> None:
        from mlxr.families.ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        cached = backend._connector_precomputed_freqs(
            8,
            4096,
            32,
            10000.0,
            (4096,),
            "split",
            True,
        )
        cached_again = backend._connector_precomputed_freqs(
            8,
            4096,
            32,
            10000.0,
            (4096,),
            "split",
            True,
        )

        cos_freq, sin_freq = cached
        self.assertIs(cached, cached_again)
        self.assertEqual(cos_freq.shape, (1, 32, 8, 64))
        self.assertEqual(sin_freq.shape, (1, 32, 8, 64))

    def test_ltx_prompt_backend_prefers_dedicated_connector_source(self) -> None:
        from mlxr.families.ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "model.safetensors"
            checkpoint_path.write_bytes(b"checkpoint")
            connector_path = root / "connectors" / "ltx_text_connectors.safetensors"
            connector_path.parent.mkdir(parents=True, exist_ok=True)
            connector_path.write_bytes(b"connector")

            encoder = backend._MLXLTXPromptEncoder(
                checkpoint_path=root,
                text_encoder_path=Path("/tmp/gemma"),
            )

            sources = encoder._connector_sources()

        self.assertEqual(sources[0], connector_path)
        self.assertIn(checkpoint_path, sources)

    def test_ltx_generation_reference_backend_requires_vae_statistics(self) -> None:
        from mlxr.families.ltx import _generation_backend as backend

        checkpoint_metadata = {
            "transformer": {
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "rope_type": "split",
                "frequencies_precision": "float64",
                "caption_proj_before_connector": True,
            },
            "vae": {
                "latent_channels": 128,
                "out_channels": 3,
                "patch_size": 4,
                "decoder_base_channels": 128,
                "decoder_blocks": [["res_x", {"num_layers": 4}]],
                "norm_layer": "pixel_norm",
                "decoder_spatial_padding_mode": "reflect",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.safetensors"
            save_file(
                {
                    "transformer_blocks.0.attn1.to_gate_logits.weight": np.zeros(
                        (32, 4096), dtype=np.float32
                    )
                },
                str(checkpoint_path),
                metadata={"config": json.dumps(checkpoint_metadata)},
            )

            with self.assertRaisesRegex(
                RuntimeError, "missing required VAE per-channel statistics"
            ):
                backend._validate_reference_backend_compatibility(checkpoint_path)

    def test_ltx_generation_reference_import_root_points_to_repo_checkout(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "references"
            / "ecosystem"
            / "mlx-video"
        )
        self.assertEqual(_REFERENCE_MLX_VIDEO_ROOT, expected)
        self.assertTrue(_REFERENCE_MLX_VIDEO_ROOT.is_dir())

    def test_ltx_generation_padded_shape_is_constructible(self) -> None:
        padded = _resolve_padded_shape(width=384, height=224)
        self.assertEqual(padded.output_width, 384)
        self.assertEqual(padded.output_height, 224)
        self.assertEqual(padded.internal_width % 64, 0)
        self.assertEqual(padded.internal_height % 64, 0)

    def test_ltx_generation_prompt_contract_rejects_runtime_mismatch(self) -> None:
        from mlxr.families.ltx import _generation_backend as backend
        from mlxr.families.ltx.prompt_encoding import PromptEncodingResult

        prompt_context = PromptEncodingResult(
            video_context=SimpleNamespace(shape=(1, 1024, 4096), dtype="bf16"),
            audio_context=SimpleNamespace(shape=(1, 1024, 2048), dtype="bf16"),
            attention_mask=None,
            prompt_text="city skyline",
            token_count=8,
            sequence_length=1024,
            video_context_shape=(1, 1024, 4096),
            attention_mask_shape=(1, 1024),
            audio_context_shape=(1, 1024, 2048),
            context_representation="post_connector",
            caption_proj_before_connector=True,
            rope_type="split",
            double_precision_rope=True,
            connector_apply_gated_attention=True,
            transformer_context_dim=4096,
            transformer_apply_gated_attention=True,
            transformer_cross_attention_adaln=True,
            config_source="memory://prompt",
        )
        runtime_config = backend._RuntimeModelConfig(
            num_attention_heads=32,
            attention_head_dim=128,
            in_channels=128,
            out_channels=128,
            num_layers=48,
            cross_attention_dim=4096,
            audio_enabled=True,
            audio_num_attention_heads=32,
            audio_attention_head_dim=64,
            audio_in_channels=128,
            audio_out_channels=128,
            audio_latent_mel_bins=16,
            audio_cross_attention_dim=2048,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=[20, 2048, 2048],
            audio_positional_embedding_max_pos=[20],
            use_middle_indices_grid=True,
            rope_type="interleaved",
            double_precision_rope=True,
            timestep_scale_multiplier=1000,
            av_ca_timestep_scale_multiplier=1000,
            norm_eps=1e-6,
            apply_gated_attention=True,
            cross_attention_adaln=True,
            caption_proj_before_connector=True,
        )

        with self.assertRaisesRegex(ValueError, "rope_type"):
            backend._assert_prompt_runtime_contract(prompt_context, runtime_config)

    def test_ltx_generation_all_valid_attention_mask_is_omitted(self) -> None:
        from mlxr.families.ltx import _generation_backend as backend
        from mlxr.families.ltx.prompt_encoding import PromptEncodingResult

        prompt_context = PromptEncodingResult(
            video_context=mx.zeros((1, 4, 4096), dtype=mx.bfloat16),
            audio_context=None,
            attention_mask=mx.ones((1, 4), dtype=mx.int32),
            prompt_text="dog",
            token_count=4,
            sequence_length=4,
            video_context_shape=(1, 4, 4096),
            attention_mask_shape=(1, 4),
        )

        self.assertIsNone(backend._attention_mask(prompt_context))

    def test_ltx_generation_prompt_contract_rejects_partial_prompt_mask(self) -> None:
        from mlxr.families.ltx import _generation_backend as backend
        from mlxr.families.ltx.prompt_encoding import PromptEncodingResult

        prompt_context = PromptEncodingResult(
            video_context=mx.zeros((1, 4, 4096), dtype=mx.bfloat16),
            audio_context=mx.zeros((1, 4, 2048), dtype=mx.bfloat16),
            attention_mask=mx.array([[1, 1, 0, 0]], dtype=mx.int32),
            prompt_text="dog",
            token_count=2,
            sequence_length=4,
            video_context_shape=(1, 4, 4096),
            attention_mask_shape=(1, 4),
            audio_context_shape=(1, 4, 2048),
            context_representation="post_connector",
            caption_proj_before_connector=True,
            rope_type="split",
            double_precision_rope=True,
            connector_apply_gated_attention=True,
            transformer_context_dim=4096,
            transformer_apply_gated_attention=True,
            transformer_cross_attention_adaln=True,
        )
        runtime_config = backend._RuntimeModelConfig(
            num_attention_heads=32,
            attention_head_dim=128,
            in_channels=128,
            out_channels=128,
            num_layers=48,
            cross_attention_dim=4096,
            audio_enabled=True,
            audio_num_attention_heads=32,
            audio_attention_head_dim=64,
            audio_in_channels=128,
            audio_out_channels=128,
            audio_latent_mel_bins=16,
            audio_cross_attention_dim=2048,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=[20, 2048, 2048],
            audio_positional_embedding_max_pos=[20],
            use_middle_indices_grid=True,
            rope_type="split",
            double_precision_rope=True,
            timestep_scale_multiplier=1000,
            av_ca_timestep_scale_multiplier=1000,
            norm_eps=1e-6,
            apply_gated_attention=True,
            cross_attention_adaln=True,
            caption_proj_before_connector=True,
        )

        with self.assertRaisesRegex(ValueError, "all-valid post-connector prompt mask"):
            backend._assert_prompt_runtime_contract(prompt_context, runtime_config)

    def test_ltx_prompt_backend_extracts_v2_projection_weights(self) -> None:
        from mlxr.families.ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        encoder = backend._MLXLTXPromptEncoder(
            checkpoint_path=Path("/tmp/ltx-checkpoint.safetensors"),
            text_encoder_path=Path("/tmp/gemma"),
        )
        weights = {
            "text_embedding_projection.video_aggregate_embed.weight": mx.array([1.0]),
            "text_embedding_projection.video_aggregate_embed.bias": mx.array([2.0]),
            "text_embedding_projection.audio_aggregate_embed.weight": mx.array([3.0]),
            "text_embedding_projection.audio_aggregate_embed.bias": mx.array([4.0]),
        }

        projection = encoder._feature_projection(weights)

        self.assertIsNotNone(projection)
        assert projection is not None
        assert projection.video_bias is not None
        assert projection.audio_weight is not None
        assert projection.audio_bias is not None
        self.assertEqual(projection.version, "v2")
        self.assertEqual(projection.video_weight.tolist(), [1.0])
        self.assertEqual(projection.video_bias.tolist(), [2.0])
        self.assertEqual(projection.audio_weight.tolist(), [3.0])
        self.assertEqual(projection.audio_bias.tolist(), [4.0])

    def test_ltx_prompt_backend_rejects_v2_projection_without_audio_weights(
        self,
    ) -> None:
        from mlxr.families.ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        encoder = backend._MLXLTXPromptEncoder(
            checkpoint_path=Path("/tmp/ltx-checkpoint.safetensors"),
            text_encoder_path=Path("/tmp/gemma"),
        )
        setattr(
            encoder,
            "feature_extractor",
            SimpleNamespace(load_weights=lambda *args, **kwargs: None),
        )
        setattr(
            encoder,
            "video_connector",
            SimpleNamespace(
                load_weights=lambda *args, **kwargs: None,
                learnable_registers=None,
            ),
        )
        setattr(
            encoder,
            "audio_connector",
            SimpleNamespace(
                load_weights=lambda *args, **kwargs: None,
                learnable_registers=None,
            ),
        )
        setattr(encoder, "layout", SimpleNamespace(num_learnable_registers=128))
        weights = {
            "text_embedding_projection.video_aggregate_embed.weight": mx.array([1.0]),
            "video_connector.learnable_registers": mx.array([1.0]),
        }

        with (
            patch.object(
                encoder,
                "_connector_sources",
                return_value=[Path("/tmp/ltx.safetensors")],
            ),
            patch(
                "mlxr.families.ltx._prompt_encoding_backend.mx.load",
                return_value=weights,
            ),
            patch("mlxr.families.ltx._prompt_encoding_backend.mx.clear_cache"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires audio_aggregate_embed weights",
            ):
                encoder._load_connector_weights()

    def test_ltx_prompt_backend_rejects_missing_audio_connector_weights(self) -> None:
        from mlxr.families.ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        encoder = backend._MLXLTXPromptEncoder(
            checkpoint_path=Path("/tmp/ltx-checkpoint.safetensors"),
            text_encoder_path=Path("/tmp/gemma"),
        )
        setattr(
            encoder,
            "feature_extractor",
            SimpleNamespace(load_weights=lambda *args, **kwargs: None),
        )
        setattr(
            encoder,
            "video_connector",
            SimpleNamespace(
                load_weights=lambda *args, **kwargs: None,
                learnable_registers=None,
            ),
        )
        setattr(
            encoder,
            "audio_connector",
            SimpleNamespace(
                load_weights=lambda *args, **kwargs: None,
                learnable_registers=None,
            ),
        )
        setattr(encoder, "layout", SimpleNamespace(num_learnable_registers=128))
        weights = {
            "text_embedding_projection.video_aggregate_embed.weight": mx.array([1.0]),
            "text_embedding_projection.audio_aggregate_embed.weight": mx.array([2.0]),
            "video_connector.learnable_registers": mx.array([1.0]),
        }

        with (
            patch.object(
                encoder,
                "_connector_sources",
                return_value=[Path("/tmp/ltx.safetensors")],
            ),
            patch(
                "mlxr.families.ltx._prompt_encoding_backend.mx.load",
                return_value=weights,
            ),
            patch("mlxr.families.ltx._prompt_encoding_backend.mx.clear_cache"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires audio connector weights",
            ):
                encoder._load_connector_weights()

    def test_server_settings_parse_http_and_referer_rules(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MLX_RUNTIME_HTTP_HOST": "127.0.0.1",
                "MLX_RUNTIME_HTTP_PORT": "46321",
                "MLX_RUNTIME_HTTP_TOKEN": "secret-token",
                "MLX_RUNTIME_ALLOWED_ORIGINS": "http://localhost:3000, http://127.0.0.1:3000",
            },
            clear=False,
        ):
            settings = ServerSettings.from_env()

        self.assertTrue(settings.http_enabled)
        self.assertEqual(settings.http_host, "127.0.0.1")
        self.assertEqual(settings.http_port, 46321)
        self.assertTrue(settings.origin_allowed("http://localhost:3000"))
        self.assertTrue(
            settings.referer_allowed("http://localhost:3000/app/index.html")
        )
        self.assertFalse(settings.referer_allowed("http://evil.example/app"))

    def test_server_settings_parse_explicit_uds_path(self) -> None:
        with patch.dict(
            os.environ,
            {"MLX_RUNTIME_UDS_PATH": "/tmp/mlxr-tests/runtime.sock"},
            clear=False,
        ):
            settings = ServerSettings.from_env()

        self.assertFalse(settings.http_enabled)
        self.assertEqual(settings.uds_path, "/tmp/mlxr-tests/runtime.sock")

    def test_manifest_stores_round_trip_runtime_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()

            input_store = InputStore(runtime_home)
            input_record = InputHandleRecord(
                handle_id="inp_1",
                media_type="image/png",
                filename="frame.png",
                size_bytes=3,
                storage_key=runtime_home.input_storage_key("inp_1", "frame.png"),
            )
            input_store.save(input_record, b"abc")
            self.assertEqual(len(input_store.list_records()), 1)
            self.assertEqual(input_store.get("inp_1"), input_record)

            job_store = JobStore(runtime_home)
            job_record = JobRecord(
                job_id="job_1",
                request=JobRequest(
                    model_id="ltx-2.3-fast-local",
                    task="video.generate",
                    inputs={"prompt": "hello"},
                ),
                state=JobState.ACCEPTED,
            )
            job_store.save(job_record)
            job_store.append_event(
                "job_1",
                RuntimeEvent(
                    job_id="job_1",
                    kind=RuntimeEventKind.JOB_ACCEPTED,
                    phase=JobState.ACCEPTED.value,
                ),
            )
            self.assertEqual(len(job_store.list_records()), 1)
            self.assertEqual(len(job_store.list_events("job_1")), 1)

            output_store = OutputStore(runtime_home)
            output_path = runtime_home.output_artifact_path(
                "job_1", "out_1", "out_1.mp4"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"video")
            output_record = OutputArtifactRecord(
                artifact_id="out_1",
                artifact_format="mp4",
                job_id="job_1",
                filename="out_1.mp4",
                media_type="video/mp4",
                size_bytes=5,
                storage_key=runtime_home.output_artifact_storage_key(
                    "job_1", "out_1", "out_1.mp4"
                ),
            )
            output_store.save(output_record)
            self.assertEqual(output_store.get("out_1"), output_record)
            self.assertEqual(
                output_store.payload_path(output_record).read_bytes(), b"video"
            )

    def test_worker_runs_scaffold_job_and_emits_terminal_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._artifactized_portable_artifact(runtime_home)
            model_record = ModelRecord(
                model_id="ltx-2.3-fast-local",
                family="ltx",
                artifact=artifact.record,
            )
            request = JobRequest(
                model_id="ltx-2.3-fast-local",
                task="video.generate",
                inputs={"prompt": "direct worker test"},
                params={"width": 96, "height": 64, "num_frames": 9, "fps": 12},
                output=JobOutputPolicy(artifact_format="mp4"),
                extensions={"simulate_delay_seconds": 0.0},
            )

            event_queue: queue.Queue[dict[str, object]] = queue.Queue()
            command_queue: queue.Queue[dict[str, object]] = queue.Queue()

            with (
                patched_ltx_prompt_encoder(token_count=9),
                patched_ltx_video_generator(),
            ):
                run_job_worker(
                    job_id="job_worker_test",
                    request_data=request.model_dump(mode="json"),
                    model_data=model_record.model_dump(mode="json"),
                    runtime_home_root=str(runtime_home.root),
                    event_queue=event_queue,
                    command_queue=command_queue,
                )

            messages: list[dict[str, object]] = []
            while True:
                try:
                    messages.append(event_queue.get_nowait())
                except queue.Empty:
                    break

            event_payloads = [
                message["event"]
                for message in messages
                if message.get("type") == "event"
                and isinstance(message.get("event"), dict)
            ]
            self.assertTrue(
                any(
                    isinstance(payload, dict)
                    and payload.get("kind") == RuntimeEventKind.JOB_COMPLETED.value
                    for payload in event_payloads
                )
            )
            self.assertTrue(
                any(message.get("type") == "artifacts" for message in messages)
            )
            self.assertTrue(
                any(
                    isinstance(payload, dict)
                    and payload.get("kind") == RuntimeEventKind.JOB_METRICS.value
                    and isinstance(payload.get("data"), dict)
                    and isinstance(payload["data"].get("metrics"), dict)
                    and payload["data"]["metrics"].get("status") == "encoded"
                    and isinstance(payload["data"].get("memory"), dict)
                    and "telemetry_available" in payload["data"]["memory"]
                    for payload in event_payloads
                )
            )
            output_path = runtime_home.output_artifact_path(
                "job_worker_test",
                "out_job_worker_test",
                "out_job_worker_test.mp4",
            )
            self.assertTrue(output_path.exists())
            self.assertIn(b"ftyp", output_path.read_bytes()[:32])

    def test_worker_uses_capability_stage_ids_for_image_jobs(self) -> None:
        class FakeImageFamilyAdapter:
            family_id = "z_image"

            def inspect_source(self, source: object) -> FamilyInspection:
                del source
                return FamilyInspection(
                    family="z_image",
                    tasks=("image.generate",),
                    scheduler_class="image_diffusion",
                )

            def fetch_policy_for_conversion(
                self, role: str, source: object
            ) -> FetchPolicy:
                del role, source
                return FetchPolicy()

            def convert(
                self, sources: dict[str, ConversionSource], plan: ConversionPlan
            ) -> PortableArtifact:
                del sources, plan
                raise RuntimeError("test double does not implement convert")

            def normalize_capability(
                self, artifact: PortableArtifact
            ) -> CapabilityDescriptor:
                return artifact.record.capability

            def load(
                self, artifact: PortableArtifact, profile: ExecutionProfile
            ) -> LoadedModelHandle:
                return LoadedModelHandle(
                    model_id=artifact.record.model_id,
                    family=artifact.record.family,
                    artifact_digest=artifact.record.artifact_digest,
                    capability=artifact.record.capability,
                    metadata={"task": profile.task},
                )

            def capabilities(self, artifact: PortableArtifact) -> CapabilityDescriptor:
                return artifact.record.capability

            def run_stage(
                self, loaded: LoadedModelHandle, stage: ExecutionStage
            ) -> StageResult:
                if stage.stage_id == "prompt_encode":
                    loaded.metadata["prompt_ready"] = True
                    return StageResult(
                        metrics={"stage": stage.stage_id, "status": "encoded"}
                    )
                if stage.stage_id == "generate":
                    if not loaded.metadata.get("prompt_ready"):
                        raise ValueError("prompt_encode must run before generate")
                    loaded.metadata["image_bytes"] = make_png_bytes()
                    return StageResult(
                        metrics={"stage": stage.stage_id, "status": "generated"}
                    )
                if stage.stage_id == "encode_output":
                    image_bytes = loaded.metadata.get("image_bytes")
                    if not isinstance(image_bytes, bytes):
                        raise ValueError(
                            "generate must run before encode_output for image jobs"
                        )
                    artifact_id = str(stage.params["artifact_id"])
                    artifact_format = str(stage.params["artifact_format"])
                    output_dir = Path(str(stage.params["output_dir"]))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{artifact_id}.{artifact_format}"
                    output_path = output_dir / filename
                    output_path.write_bytes(image_bytes)
                    return StageResult(
                        artifacts=[
                            ArtifactHandle(
                                artifact_id=artifact_id,
                                artifact_format=artifact_format,
                                metadata={
                                    "filename": filename,
                                    "media_type": "image/png",
                                    "storage_key": str(stage.params["storage_key"]),
                                    "size_bytes": output_path.stat().st_size,
                                },
                            )
                        ],
                        metrics={"stage": stage.stage_id, "status": "encoded"},
                    )
                raise ValueError(f"Unexpected stage '{stage.stage_id}'")

            def unload(self, loaded: LoadedModelHandle) -> None:
                del loaded
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()
            artifact = self._image_portable_artifact(runtime_home)
            model_record = ModelRecord(
                model_id="z-image-turbo-local",
                family="z_image",
                artifact=artifact.record,
            )
            request = JobRequest(
                model_id="z-image-turbo-local",
                task="image.generate",
                inputs={"prompt": "direct worker image test"},
                params={"width": 1024, "height": 1024, "seed": 7},
                output=JobOutputPolicy(artifact_format="png"),
                extensions={"simulate_delay_seconds": 0.0},
            )

            registry = RuntimeRegistry()
            registry.register_family(FakeImageFamilyAdapter())
            event_queue: queue.Queue[dict[str, object]] = queue.Queue()
            command_queue: queue.Queue[dict[str, object]] = queue.Queue()

            with patch(
                "mlxr.core.server.worker.default_runtime_registry",
                return_value=registry,
            ):
                run_job_worker(
                    job_id="job_image_worker_test",
                    request_data=request.model_dump(mode="json"),
                    model_data=model_record.model_dump(mode="json"),
                    runtime_home_root=str(runtime_home.root),
                    event_queue=event_queue,
                    command_queue=command_queue,
                )

            messages: list[dict[str, object]] = []
            while True:
                try:
                    messages.append(event_queue.get_nowait())
                except queue.Empty:
                    break

            event_payloads = [
                message["event"]
                for message in messages
                if message.get("type") == "event"
                and isinstance(message.get("event"), dict)
            ]
            phase_order = [
                str(payload["data"]["stage_id"])
                for payload in event_payloads
                if isinstance(payload, dict)
                and payload.get("kind") == RuntimeEventKind.JOB_PHASE_CHANGED.value
                and isinstance(payload.get("data"), dict)
                and isinstance(payload["data"].get("stage_id"), str)
            ]
            self.assertEqual(
                phase_order,
                [
                    "load_model",
                    "prompt_encode",
                    "generate",
                    "encode_output",
                    "finalize",
                ],
            )
            output_path = runtime_home.output_artifact_path(
                "job_image_worker_test",
                "out_job_image_worker_test",
                "out_job_image_worker_test.png",
            )
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_job_validation_uses_capability_constraints_for_image_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = make_state(Path(tmp_dir))
            capability = CapabilityDescriptor(
                model_id="z-image-turbo-local",
                artifact_digest="sha256:image-test",
                family="z_image",
                tasks=["image.generate"],
                artifacts_out=["png", "jpg"],
                scheduler_class="image_diffusion",
                constraints={
                    "width": {"multiple_of": 16},
                    "height": {"multiple_of": 16},
                },
            )
            request = JobRequest(
                model_id="z-image-turbo-local",
                task="image.generate",
                inputs={"prompt": "still image"},
                params={"width": 1024, "height": 1024},
                output=JobOutputPolicy(artifact_format="png"),
            )

            state.job_manager._validate_request(
                model_id="z-image-turbo-local",
                capability=capability,
                request=request,
            )

            with self.assertRaisesRegex(
                JobValidationError, "width must be an integer multiple of 16"
            ):
                state.job_manager._validate_request(
                    model_id="z-image-turbo-local",
                    capability=capability,
                    request=request.model_copy(
                        update={"params": {"width": 1025, "height": 1024}}
                    ),
                )

    def test_job_validation_requires_image_inputs_for_image_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = make_state(Path(tmp_dir))
            capability = CapabilityDescriptor(
                model_id="qwen-image-edit-local",
                artifact_digest="sha256:image-edit-test",
                family="qwen_image",
                tasks=["image.edit"],
                artifacts_out=["png", "jpg"],
                scheduler_class="image_diffusion",
            )
            request = JobRequest(
                model_id="qwen-image-edit-local",
                task="image.edit",
                inputs={"prompt": "move the same subject into a rainy alley"},
                params={"width": 1024, "height": 1024},
                output=JobOutputPolicy(artifact_format="png"),
            )

            with self.assertRaisesRegex(
                JobValidationError, "image.edit requires at least one image input"
            ):
                state.job_manager._validate_request(
                    model_id="qwen-image-edit-local",
                    capability=capability,
                    request=request,
                )

    def test_job_validation_requires_retake_window_params_and_video_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = make_state(Path(tmp_dir))
            handle = state.input_store.save(
                InputHandleRecord(
                    handle_id="inp_video",
                    role="video",
                    filename="source.mp4",
                    media_type="video/mp4",
                    storage_key="inputs/inp_video/source.mp4",
                    size_bytes=3,
                ),
                b"mp4",
            )
            capability = CapabilityDescriptor(
                model_id="ltx-2.3-fast-local",
                artifact_digest="sha256:retake-test",
                family="ltx",
                tasks=["video.retake"],
                artifacts_out=["mp4"],
                scheduler_class="media_video_dit",
            )
            request = JobRequest(
                model_id="ltx-2.3-fast-local",
                task="video.retake",
                inputs={
                    "videos": [{"input_handle": handle.handle_id, "strength": 1.0}]
                },
                params={"window_start_seconds": 1.0, "window_end_seconds": 2.0},
                output=JobOutputPolicy(artifact_format="mp4"),
            )

            state.job_manager._validate_request(
                model_id="ltx-2.3-fast-local",
                capability=capability,
                request=request,
            )

            with self.assertRaisesRegex(
                JobValidationError, "requires exactly one source video input"
            ):
                state.job_manager._validate_request(
                    model_id="ltx-2.3-fast-local",
                    capability=capability,
                    request=request.model_copy(update={"inputs": {}}),
                )

            with self.assertRaisesRegex(
                JobValidationError,
                "video.retake requires numeric params.window_start_seconds",
            ):
                state.job_manager._validate_request(
                    model_id="ltx-2.3-fast-local",
                    capability=capability,
                    request=request.model_copy(update={"params": {}}),
                )

    def test_job_validation_requires_two_distinct_keyframes_for_interpolation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = make_state(Path(tmp_dir))
            state.input_store.save(
                InputHandleRecord(
                    handle_id="img_1",
                    role="image",
                    filename="first.png",
                    media_type="image/png",
                    storage_key="inputs/img_1/first.png",
                    size_bytes=4,
                ),
                b"png1",
            )
            state.input_store.save(
                InputHandleRecord(
                    handle_id="img_2",
                    role="image",
                    filename="last.png",
                    media_type="image/png",
                    storage_key="inputs/img_2/last.png",
                    size_bytes=4,
                ),
                b"png2",
            )
            capability = CapabilityDescriptor(
                model_id="ltx-2.3-dev-local",
                artifact_digest="sha256:interpolate-test",
                family="ltx",
                tasks=["video.interpolate"],
                artifacts_out=["mp4"],
                scheduler_class="media_video_dit",
            )
            request = JobRequest(
                model_id="ltx-2.3-dev-local",
                task="video.interpolate",
                inputs={
                    "images": [
                        {"input_handle": "img_1", "frame_index": 0, "strength": 1.0},
                        {"input_handle": "img_2", "frame_index": 8, "strength": 1.0},
                    ]
                },
                params={"width": 96, "height": 64, "num_frames": 9},
                output=JobOutputPolicy(artifact_format="mp4"),
            )

            state.job_manager._validate_request(
                model_id="ltx-2.3-dev-local",
                capability=capability,
                request=request,
            )

            with self.assertRaisesRegex(
                JobValidationError,
                "video.interpolate requires at least two image inputs",
            ):
                state.job_manager._validate_request(
                    model_id="ltx-2.3-dev-local",
                    capability=capability,
                    request=request.model_copy(
                        update={
                            "inputs": {
                                "images": [
                                    {
                                        "input_handle": "img_1",
                                        "frame_index": 0,
                                        "strength": 1.0,
                                    }
                                ]
                            }
                        }
                    ),
                )

    def test_job_validation_requires_single_video_and_lora_for_video_condition_video(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = make_state(Path(tmp_dir))
            state.input_store.save(
                InputHandleRecord(
                    handle_id="vid_1",
                    role="video",
                    filename="reference.mp4",
                    media_type="video/mp4",
                    storage_key="inputs/vid_1/reference.mp4",
                    size_bytes=3,
                ),
                b"mp4",
            )
            state.input_store.save(
                InputHandleRecord(
                    handle_id="lora_1",
                    role="lora",
                    filename="control.safetensors",
                    media_type="application/x-safetensors",
                    storage_key="inputs/lora_1/control.safetensors",
                    size_bytes=4,
                ),
                b"lora",
            )
            capability = CapabilityDescriptor(
                model_id="ltx-2.3-fast-local",
                artifact_digest="sha256:ic-lora-test",
                family="ltx",
                tasks=["video.condition.video"],
                artifacts_out=["mp4", "wav"],
                scheduler_class="media_video_dit",
            )
            request = JobRequest(
                model_id="ltx-2.3-fast-local",
                task="video.condition.video",
                inputs={
                    "videos": [{"input_handle": "vid_1", "strength": 0.8}],
                    "loras": [{"input_handle": "lora_1", "strength": 0.6}],
                },
                params={"width": 96, "height": 64, "num_frames": 9},
                output=JobOutputPolicy(artifact_format="mp4"),
            )

            state.job_manager._validate_request(
                model_id="ltx-2.3-fast-local",
                capability=capability,
                request=request,
            )

            with self.assertRaisesRegex(
                JobValidationError,
                "requires exactly one reference video input",
            ):
                state.job_manager._validate_request(
                    model_id="ltx-2.3-fast-local",
                    capability=capability,
                    request=request.model_copy(
                        update={"inputs": {"loras": request.inputs["loras"]}}
                    ),
                )

            with self.assertRaisesRegex(
                JobValidationError,
                "requires exactly one LoRA input",
            ):
                state.job_manager._validate_request(
                    model_id="ltx-2.3-fast-local",
                    capability=capability,
                    request=request.model_copy(
                        update={"inputs": {"videos": request.inputs["videos"]}}
                    ),
                )

            with self.assertRaisesRegex(
                JobValidationError,
                "requires exactly one reference video input",
            ):
                state.job_manager._validate_request(
                    model_id="ltx-2.3-fast-local",
                    capability=capability,
                    request=request.model_copy(
                        update={
                            "inputs": {
                                "videos": [
                                    {"input_handle": "vid_1", "strength": 0.8},
                                    {"input_handle": "vid_1", "strength": 0.8},
                                ],
                                "loras": request.inputs["loras"],
                            }
                        }
                    ),
                )

            with self.assertRaisesRegex(
                JobValidationError,
                "video.condition.video currently only supports mp4 output",
            ):
                state.job_manager._validate_request(
                    model_id="ltx-2.3-fast-local",
                    capability=capability,
                    request=request.model_copy(
                        update={"output": JobOutputPolicy(artifact_format="wav")}
                    ),
                )

    def test_job_manager_marks_job_failed_when_worker_exits_without_event(self) -> None:
        class FakeEventQueue:
            def __init__(self) -> None:
                self.closed = False

            def put(self, item: dict[str, object]) -> None:
                del item

            def get(self, timeout: float | None = None) -> dict[str, object]:
                del timeout
                raise queue.Empty

            def get_nowait(self) -> dict[str, object]:
                raise queue.Empty

            def close(self) -> None:
                self.closed = True

        class FakeCommandQueue:
            def __init__(self) -> None:
                self.closed = False

            def put(self, item: dict[str, object]) -> None:
                del item

            def get(self, timeout: float | None = None) -> dict[str, object]:
                del timeout
                raise queue.Empty

            def get_nowait(self) -> dict[str, object]:
                raise queue.Empty

            def close(self) -> None:
                self.closed = True

        class FakeProcess:
            def __init__(self) -> None:
                self.join_calls = 0
                self.terminate_calls = 0
                self._exitcode = 1

            def start(self) -> None:
                return None

            def join(self, timeout: float | None = None) -> None:
                del timeout
                self.join_calls += 1

            def is_alive(self) -> bool:
                return False

            def terminate(self) -> None:
                self.terminate_calls += 1

            @property
            def exitcode(self) -> int | None:
                return self._exitcode

        with tempfile.TemporaryDirectory() as tmp_dir:
            state = make_state(Path(tmp_dir))
            job_id = "job_worker_exit_test"
            record = JobRecord(
                job_id=job_id,
                request=JobRequest(
                    model_id="ltx-2.3-fast-local",
                    task="video.generate",
                    inputs={"prompt": "fox"},
                    params={},
                    output=JobOutputPolicy(artifact_format="mp4"),
                ),
                state=JobState.RUNNING,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            state.job_store.save(record)

            event_queue = FakeEventQueue()
            command_queue = FakeCommandQueue()
            process = FakeProcess()

            state.job_manager._listen_to_worker(
                job_id,
                event_queue,
                command_queue,
                process,
            )

            failed = state.job_store.get(job_id)
            assert failed is not None
            self.assertEqual(failed.state, JobState.FAILED)
            self.assertEqual(
                failed.error,
                "Worker exited before reaching a terminal state (exitcode=1)",
            )
            self.assertTrue(event_queue.closed)
            self.assertTrue(command_queue.closed)
            self.assertGreaterEqual(process.join_calls, 1)
            self.assertEqual(process.terminate_calls, 0)

    def test_job_manager_reconciles_orphaned_nonterminal_jobs_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = make_state(Path(tmp_dir))
            job_id = "job_orphaned_restart_test"
            state.job_store.save(
                JobRecord(
                    job_id=job_id,
                    request=JobRequest(
                        model_id="ltx-2.3-fast-local",
                        task="video.generate",
                        inputs={"prompt": "fox"},
                        params={},
                        output=JobOutputPolicy(artifact_format="mp4"),
                    ),
                    state=JobState.RUNNING,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
            )

            JobManager(
                catalog=state.catalog,
                job_store=state.job_store,
                input_store=state.input_store,
                output_store=state.output_store,
                settings=state.settings,
            )

            failed = state.job_store.get(job_id)
            assert failed is not None
            self.assertEqual(failed.state, JobState.FAILED)
            self.assertEqual(
                failed.error,
                "Control-plane restarted before job reached a terminal state",
            )

    def test_job_phase_change_clears_stale_error_for_running_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = make_state(Path(tmp_dir))
            job_id = "job_phase_error_clear_test"
            state.job_store.save(
                JobRecord(
                    job_id=job_id,
                    request=JobRequest(
                        model_id="ltx-2.3-fast-local",
                        task="video.generate",
                        inputs={"prompt": "fox"},
                        params={},
                        output=JobOutputPolicy(artifact_format="mp4"),
                    ),
                    state=JobState.ACCEPTED,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    error="stale error",
                )
            )

            state.job_manager._update_job_from_event(
                RuntimeEvent(
                    job_id=job_id,
                    kind=RuntimeEventKind.JOB_PHASE_CHANGED,
                    phase="running",
                    data={"stage_id": "generate"},
                )
            )

            updated = state.job_store.get(job_id)
            assert updated is not None
            self.assertEqual(updated.state, JobState.RUNNING)
            self.assertIsNone(updated.error)

    def test_ltx_runtime_vae_encoder_rejects_missing_encoder_weights(self) -> None:
        from mlxr.families.ltx._generation_backend.video_stack import (
            _load_runtime_vae_encoder,
        )

        checkpoint_metadata = {
            "transformer": {
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
            },
            "vae": {
                "latent_channels": 128,
                "out_channels": 3,
                "patch_size": 4,
                "decoder_base_channels": 128,
                "decoder_blocks": [["res_x", {"num_layers": 4}]],
                "norm_layer": "pixel_norm",
                "decoder_spatial_padding_mode": "reflect",
                "encoder_spatial_padding_mode": "reflect",
                "encoder_blocks": [["res_x", {"num_layers": 4}]],
                "latent_log_var": "per_channel",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.safetensors"
            save_file(
                {
                    "vae.per_channel_statistics.mean-of-means": np.zeros(
                        (128,), dtype=np.float32
                    ),
                    "vae.per_channel_statistics.std-of-means": np.ones(
                        (128,), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={"config": json.dumps(checkpoint_metadata)},
            )

            with self.assertRaisesRegex(RuntimeError, "missing VAE encoder weights"):
                _load_runtime_vae_encoder(checkpoint_path)

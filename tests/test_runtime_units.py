from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from ltx import LTXFamilyAdapter
from ltx._generation_backend.conditioning import _resolve_padded_shape
from ltx._generation_backend.reference_imports import _REFERENCE_MLX_VIDEO_ROOT
from ltx._generation_backend.runtime_helpers import _normalize_audio_mel_layout
from mlx_runtime_core import (
    ExecutionProfile,
    ExecutionStage,
    LoadedModelHandle,
    PortableArtifact,
    RuntimeHome,
)
from mlx_runtime_schemas import (
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
from mlx_runtime_server.settings import ServerSettings
from mlx_runtime_server.store import InputStore, JobStore, OutputStore
from mlx_runtime_server.worker import run_job_worker
from safetensors.numpy import save_file

from tests.runtime_test_support import (
    LTX_CHECKPOINT_FILENAME,
    LTX_SPATIAL_UPSAMPLER_FILENAME,
    make_png_bytes,
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
                "ltx.adapter.create_prompt_encoder",
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

    def test_ltx_generation_backend_reads_nested_bwe_vocoder_contract(self) -> None:
        from ltx import _generation_backend as backend

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ltx-checkpoint.safetensors"
            save_file(
                {"dummy": np.zeros((1,), dtype=np.float32)},
                str(checkpoint_path),
                metadata={
                    "config": json.dumps(
                        {
                            "vocoder": {
                                "vocoder": {
                                    "upsample_initial_channel": 1536,
                                    "resblock": "AMP1",
                                    "upsample_rates": [5, 2, 2, 2, 2, 2],
                                    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
                                    "resblock_kernel_sizes": [3, 7, 11],
                                    "resblock_dilation_sizes": [
                                        [1, 3, 5],
                                        [1, 3, 5],
                                        [1, 3, 5],
                                    ],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                },
                                "bwe": {
                                    "upsample_initial_channel": 512,
                                    "resblock": "AMP1",
                                    "upsample_rates": [6, 5, 2, 2, 2],
                                    "upsample_kernel_sizes": [12, 11, 4, 4, 4],
                                    "resblock_kernel_sizes": [3, 7, 11],
                                    "resblock_dilation_sizes": [
                                        [1, 3, 5],
                                        [1, 3, 5],
                                        [1, 3, 5],
                                    ],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                    "apply_final_activation": False,
                                    "input_sampling_rate": 16000,
                                    "output_sampling_rate": 48000,
                                    "hop_length": 80,
                                    "n_fft": 512,
                                    "win_size": 512,
                                    "num_mels": 64,
                                },
                            }
                        }
                    )
                },
            )

            config = backend._runtime_vocoder_config(checkpoint_path)

        self.assertEqual(config.vocoder.resblock, "AMP1")
        self.assertEqual(config.vocoder.activation, "snakebeta")
        self.assertFalse(config.vocoder.use_tanh_at_final)
        self.assertTrue(config.vocoder.apply_final_activation)
        self.assertFalse(config.vocoder.use_bias_at_final)
        self.assertEqual(config.vocoder.upsample_initial_channel, 1536)
        self.assertEqual(config.output_sample_rate, 16000)
        self.assertTrue(config.uses_bwe)
        self.assertEqual(config.bwe_output_sample_rate, 48000)
        self.assertIsNotNone(config.bwe)
        assert config.bwe is not None
        self.assertEqual(config.bwe.input_sample_rate, 16000)
        self.assertEqual(config.bwe.output_sample_rate, 48000)
        self.assertEqual(config.bwe.hop_length, 80)
        self.assertEqual(config.bwe.n_fft, 512)
        self.assertEqual(config.bwe.win_size, 512)
        self.assertEqual(config.bwe.num_mels, 64)

    def test_ltx_generation_backend_uses_bwe_vocoder_when_checkpoint_has_full_contract(
        self,
    ) -> None:
        from ltx._generation_backend.video_stack import _load_runtime_vocoder

        def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
            sanitized: dict[str, mx.array] = {}
            for key, value in weights.items():
                if key.endswith(".weight") and value.ndim == 3:
                    if key.startswith("ups."):
                        value = mx.transpose(value, (1, 2, 0))
                    else:
                        value = mx.transpose(value, (0, 2, 1))
                sanitized[key] = value
            return sanitized

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ltx-checkpoint.safetensors"
            save_file(
                {
                    "vocoder.vocoder.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.vocoder.conv_post.weight": np.zeros(
                        (2, 2, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_post.weight": np.zeros(
                        (2, 2, 7), dtype=np.float32
                    ),
                    "vocoder.mel_stft.mel_basis": np.zeros((64, 257), dtype=np.float32),
                    "vocoder.mel_stft.stft_fn.forward_basis": np.zeros(
                        (514, 1, 512), dtype=np.float32
                    ),
                    "vocoder.mel_stft.stft_fn.inverse_basis": np.zeros(
                        (514, 1, 512), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={
                    "config": json.dumps(
                        {
                            "vocoder": {
                                "vocoder": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                },
                                "bwe": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                    "apply_final_activation": False,
                                    "input_sampling_rate": 16000,
                                    "output_sampling_rate": 48000,
                                    "hop_length": 80,
                                    "n_fft": 512,
                                    "win_size": 512,
                                    "num_mels": 64,
                                },
                            }
                        }
                    )
                },
            )
            checkpoint_weights = mx.load(str(checkpoint_path))
            if not isinstance(checkpoint_weights, dict):
                self.fail("Expected checkpoint weights to load into a mapping")

            vocoder, sample_rate, backend_name = _load_runtime_vocoder(
                checkpoint_path=checkpoint_path,
                checkpoint_weights=checkpoint_weights,
                sanitize_vocoder_weights=sanitize,
            )

        self.assertEqual(sample_rate, 48000)
        self.assertEqual(backend_name, "mlx_vocoder_with_bwe")
        self.assertTrue(hasattr(vocoder, "bwe_generator"))
        self.assertTrue(hasattr(vocoder, "mel_stft"))

    def test_ltx_generation_backend_bwe_loader_fails_closed_on_missing_stft_buffers(
        self,
    ) -> None:
        from ltx._generation_backend.video_stack import _load_runtime_vocoder

        def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
            sanitized: dict[str, mx.array] = {}
            for key, value in weights.items():
                if key.endswith(".weight") and value.ndim == 3:
                    if key.startswith("ups."):
                        value = mx.transpose(value, (1, 2, 0))
                    else:
                        value = mx.transpose(value, (0, 2, 1))
                sanitized[key] = value
            return sanitized

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ltx-checkpoint.safetensors"
            save_file(
                {
                    "vocoder.vocoder.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={
                    "config": json.dumps(
                        {
                            "vocoder": {
                                "vocoder": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                },
                                "bwe": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "input_sampling_rate": 16000,
                                    "output_sampling_rate": 48000,
                                    "hop_length": 80,
                                    "n_fft": 512,
                                    "win_size": 512,
                                    "num_mels": 64,
                                },
                            }
                        }
                    )
                },
            )
            checkpoint_weights = mx.load(str(checkpoint_path))
            if not isinstance(checkpoint_weights, dict):
                self.fail("Expected checkpoint weights to load into a mapping")

            with self.assertRaisesRegex(RuntimeError, "missing mel STFT buffers"):
                _load_runtime_vocoder(
                    checkpoint_path=checkpoint_path,
                    checkpoint_weights=checkpoint_weights,
                    sanitize_vocoder_weights=sanitize,
                )

    def test_ltx_generation_backend_bwe_loader_fails_closed_on_wrong_stft_shapes(
        self,
    ) -> None:
        from ltx._generation_backend.video_stack import _load_runtime_vocoder

        def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
            sanitized: dict[str, mx.array] = {}
            for key, value in weights.items():
                if key.endswith(".weight") and value.ndim == 3:
                    if key.startswith("ups."):
                        value = mx.transpose(value, (1, 2, 0))
                    else:
                        value = mx.transpose(value, (0, 2, 1))
                sanitized[key] = value
            return sanitized

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ltx-checkpoint.safetensors"
            save_file(
                {
                    "vocoder.vocoder.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.vocoder.conv_post.weight": np.zeros(
                        (2, 2, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_pre.weight": np.zeros(
                        (4, 128, 7), dtype=np.float32
                    ),
                    "vocoder.bwe_generator.conv_post.weight": np.zeros(
                        (2, 2, 7), dtype=np.float32
                    ),
                    "vocoder.mel_stft.mel_basis": np.zeros((63, 257), dtype=np.float32),
                    "vocoder.mel_stft.stft_fn.forward_basis": np.zeros(
                        (514, 1, 512), dtype=np.float32
                    ),
                    "vocoder.mel_stft.stft_fn.inverse_basis": np.zeros(
                        (514, 1, 512), dtype=np.float32
                    ),
                },
                str(checkpoint_path),
                metadata={
                    "config": json.dumps(
                        {
                            "vocoder": {
                                "vocoder": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                },
                                "bwe": {
                                    "upsample_initial_channel": 4,
                                    "resblock": "AMP1",
                                    "upsample_rates": [2],
                                    "upsample_kernel_sizes": [4],
                                    "resblock_kernel_sizes": [3],
                                    "resblock_dilation_sizes": [[1, 3, 5]],
                                    "stereo": True,
                                    "use_tanh_at_final": False,
                                    "activation": "snakebeta",
                                    "use_bias_at_final": False,
                                    "apply_final_activation": False,
                                    "input_sampling_rate": 16000,
                                    "output_sampling_rate": 48000,
                                    "hop_length": 80,
                                    "n_fft": 512,
                                    "win_size": 512,
                                    "num_mels": 64,
                                },
                            }
                        }
                    )
                },
            )
            checkpoint_weights = mx.load(str(checkpoint_path))
            if not isinstance(checkpoint_weights, dict):
                self.fail("Expected checkpoint weights to load into a mapping")

            with self.assertRaisesRegex(RuntimeError, "invalid mel basis shape"):
                _load_runtime_vocoder(
                    checkpoint_path=checkpoint_path,
                    checkpoint_weights=checkpoint_weights,
                    sanitize_vocoder_weights=sanitize,
                )

    def test_ltx_audio_bwe_wrapper_uses_batch_channel_time_waveform_contract(
        self,
    ) -> None:
        from ltx._audio_bwe import AudioVocoderWithBWE

        class FakeBaseVocoder:
            def __call__(self, mel_spec: mx.array) -> mx.array:
                del mel_spec
                return mx.zeros((1, 2, 5), dtype=mx.float32)

        class FakeMelSTFT:
            def mel_spectrogram(
                self, waveform: mx.array
            ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
                batch_channels = int(waveform.shape[0])
                log_mel = mx.zeros((batch_channels, 64, 2), dtype=mx.float32)
                magnitude = mx.zeros((batch_channels, 257, 2), dtype=mx.float32)
                phase = mx.zeros((batch_channels, 257, 2), dtype=mx.float32)
                energy = mx.zeros((batch_channels, 2), dtype=mx.float32)
                return log_mel, magnitude, phase, energy

        class FakeBWEGenerator:
            def __init__(self) -> None:
                self.last_input_shape: tuple[int, ...] | None = None

            def __call__(self, mel_spec: mx.array) -> mx.array:
                self.last_input_shape = tuple(int(v) for v in mel_spec.shape)
                return mx.zeros((1, 2, 15), dtype=mx.float32)

        class FakeResampler:
            def __call__(self, waveform: mx.array) -> mx.array:
                self.last_waveform_shape = tuple(int(v) for v in waveform.shape)
                return mx.zeros((1, 2, 15), dtype=mx.float32)

        bwe_generator = FakeBWEGenerator()
        wrapper = AudioVocoderWithBWE(
            vocoder=FakeBaseVocoder(),
            bwe_generator=bwe_generator,
            mel_stft=FakeMelSTFT(),
            input_sample_rate=16000,
            output_sample_rate=48000,
            hop_length=80,
        )
        wrapper.resampler = FakeResampler()

        output = wrapper(mx.zeros((1, 2, 2, 64), dtype=mx.float32))

        self.assertEqual(output.shape, (1, 2, 15))
        self.assertEqual(bwe_generator.last_input_shape, (1, 2, 2, 64))

    def test_ltx_audio_bwe_hann_filter_has_expected_shape(self) -> None:
        from ltx._audio_bwe import _hann_sinc_filter1d

        filter_ = _hann_sinc_filter1d(ratio=3)

        self.assertEqual(filter_.shape[0], 1)
        self.assertEqual(filter_.shape[2], 1)
        self.assertEqual(filter_.dtype, mx.float32)

    def test_ltx_audio_bwe_resampler_upsamples_batch_channel_time_waveform(
        self,
    ) -> None:
        from ltx._audio_bwe import _WaveformResampler

        resampler = _WaveformResampler(
            input_sample_rate=16000,
            output_sample_rate=48000,
        )

        waveform = mx.ones((1, 2, 5), dtype=mx.float32)
        upsampled = resampler(waveform)

        self.assertEqual(upsampled.shape, (1, 2, 15))

    def test_ltx_audio_mel_stft_returns_expected_shapes(self) -> None:
        from ltx._audio_bwe import AudioMelSTFT

        mel_stft = AudioMelSTFT(
            filter_length=4,
            hop_length=2,
            win_length=4,
            n_mel_channels=2,
        )
        mel_stft.stft_fn.forward_basis = mx.ones((6, 4, 1), dtype=mx.float32)
        mel_stft.mel_basis = mx.ones((2, 3), dtype=mx.float32)

        log_mel, magnitude, phase, energy = mel_stft.mel_spectrogram(
            mx.ones((1, 6), dtype=mx.float32)
        )

        self.assertEqual(log_mel.shape, (1, 2, 3))
        self.assertEqual(magnitude.shape, (1, 3, 3))
        self.assertEqual(phase.shape, (1, 3, 3))
        self.assertEqual(energy.shape, (1, 3))

    def test_ltx_audio_mel_stft_rejects_wrong_rank(self) -> None:
        from ltx._audio_bwe import AudioMelSTFT

        mel_stft = AudioMelSTFT(
            filter_length=4,
            hop_length=2,
            win_length=4,
            n_mel_channels=2,
        )

        with self.assertRaisesRegex(ValueError, "expects \\[batch, time\\]"):
            mel_stft.mel_spectrogram(mx.ones((1, 2, 3), dtype=mx.float32))

    def test_ltx_audio_vocoder_rejects_unsupported_activation(self) -> None:
        from ltx._audio_vocoder import AudioVocoder

        with self.assertRaisesRegex(ValueError, "Unsupported LTX vocoder activation"):
            AudioVocoder(
                resblock="AMP1",
                activation="gelu",
            )

    def test_ltx_prompt_backend_builds_v2_layout_from_current_config(self) -> None:
        from ltx import _prompt_encoding_backend as backend

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
        import mlx.core as mx
        from ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        encoded = mx.ones((1, 4, 2), dtype=mx.float32)
        additive_mask = mx.array([[[[0.0, 0.0, -10000.0, -10000.0]]]], dtype=mx.float32)

        masked, binary_mask = backend._to_binary_mask(encoded, additive_mask)

        self.assertEqual(binary_mask.tolist(), [[1, 1, 0, 0]])
        self.assertEqual(masked[:, 2:, :].tolist(), [[[0.0, 0.0], [0.0, 0.0]]])

    def test_ltx_prompt_backend_builds_left_padded_gemma_masks(self) -> None:
        import mlx.core as mx
        from ltx import _prompt_encoding_backend as backend

        if backend._RUNTIME_IMPORT_ERROR is not None:
            self.skipTest("LTX prompt backend dependencies are unavailable")

        hidden = mx.zeros((1, 8, 4), dtype=mx.bfloat16)
        attention_mask = mx.array([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=mx.int32)
        config = SimpleNamespace(sliding_window_pattern=6, sliding_window=4)

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
        from ltx import _prompt_encoding_backend as backend

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
        from ltx import _prompt_encoding_backend as backend

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

    def test_ltx_generation_runtime_config_infers_current_22b_flags(self) -> None:
        from ltx import _generation_backend as backend

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
                "decoder_blocks": [
                    ["res_x", {"num_layers": 4}],
                    ["compress_all", {"multiplier": 2, "residual": False}],
                    ["res_x", {"num_layers": 3}],
                    ["compress_all", {"multiplier": 2, "residual": False}],
                    ["res_x", {"num_layers": 3}],
                    ["compress_all", {"multiplier": 2, "residual": False}],
                    ["res_x", {"num_layers": 4}],
                ],
                "norm_layer": "pixel_norm",
                "decoder_spatial_padding_mode": "reflect",
                "timestep_conditioning": True,
                "causal_decoder": False,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.safetensors"
            save_file(
                {
                    "transformer_blocks.0.attn1.to_gate_logits.weight": np.zeros(
                        (32, 4096), dtype=np.float32
                    ),
                    "transformer_blocks.0.prompt_scale_shift_table": np.zeros(
                        (2, 4096), dtype=np.float32
                    ),
                    "audio_patchify_proj.weight": np.zeros(
                        (2048, 128), dtype=np.float32
                    ),
                    "audio_proj_out.weight": np.zeros((128, 2048), dtype=np.float32),
                    "audio_prompt_adaln_single.linear.weight": np.zeros(
                        (4096, 2048), dtype=np.float32
                    ),
                    "transformer_blocks.0.audio_prompt_scale_shift_table": np.zeros(
                        (2, 2048), dtype=np.float32
                    ),
                    "transformer_blocks.0.audio_attn1.to_q.weight": np.zeros(
                        (2048, 2048), dtype=np.float32
                    ),
                    "transformer_blocks.0.audio_attn2.to_q.weight": np.zeros(
                        (2048, 2048), dtype=np.float32
                    ),
                    "transformer_blocks.0.audio_to_video_attn.to_q.weight": np.zeros(
                        (2048, 4096), dtype=np.float32
                    ),
                    "transformer_blocks.0.video_to_audio_attn.to_q.weight": np.zeros(
                        (2048, 2048), dtype=np.float32
                    ),
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

            runtime_config = backend._runtime_model_config(checkpoint_path)

        self.assertTrue(runtime_config.apply_gated_attention)
        self.assertTrue(runtime_config.cross_attention_adaln)
        self.assertTrue(runtime_config.caption_proj_before_connector)
        self.assertEqual(runtime_config.rope_type, "split")
        self.assertTrue(runtime_config.double_precision_rope)
        self.assertTrue(runtime_config.audio_enabled)
        self.assertEqual(runtime_config.audio_num_attention_heads, 32)
        self.assertEqual(runtime_config.audio_attention_head_dim, 64)
        self.assertEqual(runtime_config.audio_in_channels, 128)
        self.assertEqual(runtime_config.audio_out_channels, 128)
        self.assertEqual(runtime_config.audio_cross_attention_dim, 2048)
        self.assertEqual(runtime_config.audio_positional_embedding_max_pos, [20])
        self.assertEqual(runtime_config.av_ca_timestep_scale_multiplier, 1000)

    def test_ltx_generation_reference_backend_requires_vae_statistics(self) -> None:
        from ltx import _generation_backend as backend

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
        from ltx import _generation_backend as backend
        from ltx.prompt_encoding import PromptEncodingResult

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
        import mlx.core as mx
        from ltx import _generation_backend as backend
        from ltx.prompt_encoding import PromptEncodingResult

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
        import mlx.core as mx
        from ltx import _generation_backend as backend
        from ltx.prompt_encoding import PromptEncodingResult

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
        import mlx.core as mx
        from ltx import _prompt_encoding_backend as backend

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
        import mlx.core as mx
        from ltx import _prompt_encoding_backend as backend

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
                "ltx._prompt_encoding_backend.mx.load",
                return_value=weights,
            ),
            patch("ltx._prompt_encoding_backend.mx.clear_cache"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires audio_aggregate_embed weights",
            ):
                encoder._load_connector_weights()

    def test_ltx_prompt_backend_rejects_missing_audio_connector_weights(self) -> None:
        import mlx.core as mx
        from ltx import _prompt_encoding_backend as backend

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
                "ltx._prompt_encoding_backend.mx.load",
                return_value=weights,
            ),
            patch("ltx._prompt_encoding_backend.mx.clear_cache"),
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

    def test_ltx_audio_conditioning_normalizes_mel_layout_for_encoder(self) -> None:
        mel_bins_first = np.zeros((1, 64, 2, 65), dtype=np.float32)
        normalized = _normalize_audio_mel_layout(mel_bins_first, input_channels=2)
        self.assertEqual(normalized.shape, (1, 2, 65, 64))

        channels_first = np.zeros((1, 2, 65, 64), dtype=np.float32)
        preserved = _normalize_audio_mel_layout(channels_first, input_channels=2)
        self.assertEqual(preserved.shape, (1, 2, 65, 64))

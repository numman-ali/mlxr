from __future__ import annotations

import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_runtime_core import (
    ExecutionProfile,
    ExecutionStage,
    LoadedModelHandle,
    PortableArtifact,
    RuntimeHome,
)
from mlx_runtime_family_ltx import LTXFamilyAdapter
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

from tests.runtime_test_support import (
    LTX_CHECKPOINT_FILENAME,
    LTX_SPATIAL_UPSAMPLER_FILENAME,
    make_png_bytes,
    patched_ltx_prompt_encoder,
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
                    tasks=["video.generate"],
                    artifacts_out=["mp4"],
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

            with patched_ltx_prompt_encoder(token_count=11) as encoders:
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
                    "mlx_prompt_conditioned_preview",
                )

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
                    self.assertEqual(encode_result.metrics["status"], "encoded")

                adapter.unload(loaded)
                self.assertEqual(len(encoders), 1)
                self.assertTrue(encoders[0].closed)

    def test_ltx_adapter_prompt_encode_rejects_negative_prompt(self) -> None:
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
                "does not support negative_prompt yet",
            ):
                adapter.run_stage(
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
                "mlx_runtime_family_ltx.adapter.create_prompt_encoder",
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

            with patched_ltx_prompt_encoder(token_count=9):
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

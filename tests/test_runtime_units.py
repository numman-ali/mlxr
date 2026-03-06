from __future__ import annotations

import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_runtime_core import ExecutionStage, LoadedModelHandle, RuntimeHome
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
    PortableArtifactRecord,
    ProvenanceRecord,
    RuntimeEvent,
    RuntimeEventKind,
)
from mlx_runtime_server.settings import ServerSettings
from mlx_runtime_server.store import InputStore, JobStore, OutputStore
from mlx_runtime_server.worker import run_job_worker


class RuntimeUnitTests(unittest.TestCase):
    def test_ltx_adapter_stage_scaffold_writes_output(self) -> None:
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="prompt_encode",
                    inputs={"prompt": "cinematic fox in snow"},
                    params={"simulate_delay_seconds": 0.0},
                ),
            )
            self.assertEqual(prompt_result.metrics["status"], "scaffold")

            encode_result = adapter.run_stage(
                loaded,
                ExecutionStage(
                    stage_id="encode_output",
                    inputs={"prompt": "cinematic fox in snow"},
                    params={
                        "artifact_id": "out_job_1",
                        "artifact_format": "mp4",
                        "output_dir": tmp_dir,
                        "storage_key": "jobs/job_1/outputs/out_job_1/out_job_1.mp4",
                        "simulate_delay_seconds": 0.0,
                    },
                ),
            )
            output_path = Path(tmp_dir) / "out_job_1.mp4"
            self.assertTrue(output_path.exists())
            self.assertEqual(len(encode_result.artifacts), 1)
            self.assertEqual(
                encode_result.artifacts[0].metadata["media_type"], "video/mp4"
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
            model_record = ModelRecord(
                model_id="ltx-2.3-fast-local",
                family="ltx",
                artifact=PortableArtifactRecord(
                    model_id="ltx-2.3-fast-local",
                    artifact_digest="sha256:test",
                    family="ltx",
                    format_version="0.1.0",
                    weight_format="mlx_safetensors_sharded",
                    storage_key="artifacts-portable/ltx/ltx-2.3-fast-local/sha256_test",
                    capability=CapabilityDescriptor(
                        model_id="ltx-2.3-fast-local",
                        artifact_digest="sha256:test",
                        family="ltx",
                        tasks=["video.generate"],
                        artifacts_out=["mp4"],
                        scheduler_class="media_video_dit",
                    ),
                    provenance=ProvenanceRecord(
                        provider="local", locator={"path": "/tmp/model"}
                    ),
                ),
            )
            request = JobRequest(
                model_id="ltx-2.3-fast-local",
                task="video.generate",
                inputs={"prompt": "direct worker test"},
                params={"num_frames": 9},
                output=JobOutputPolicy(artifact_format="mp4"),
                extensions={"simulate_delay_seconds": 0.0},
            )

            event_queue: queue.Queue[dict[str, object]] = queue.Queue()
            command_queue: queue.Queue[dict[str, object]] = queue.Queue()

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
                runtime_home.output_artifact_path(
                    "job_worker_test",
                    "out_job_worker_test",
                    "out_job_worker_test.mp4",
                ).exists()
            )

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from mlx_runtime_schemas import JobSubmitResult
from mlx_runtime_server.app import create_app
from mlx_runtime_server.settings import ServerSettings

from tests.runtime_test_support import (
    http_headers,
    import_input_handle,
    make_local_bundle,
    make_state,
    patched_inline_job_process_context,
    patched_ltx_prompt_encoder,
    register_local_ltx_model,
    response_model,
    wait_for_job_terminal_state,
)


def first_artifact_id(job_record: dict[str, object]) -> str:
    artifacts = job_record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AssertionError("Expected at least one output artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, dict):
        raise AssertionError(f"Expected artifact object, got {artifact!r}")
    artifact_id = artifact.get("artifact_id")
    if not isinstance(artifact_id, str):
        raise AssertionError(f"Expected artifact_id string, got {artifact_id!r}")
    return artifact_id


class RuntimeJobTests(unittest.TestCase):
    def test_job_submit_persists_events_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(client, source_dir)
                handle_id = import_input_handle(client, b"conditioning-bytes")

                submit = client.post(
                    "/v1/jobs",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "task": "video.condition.image",
                        "inputs": {
                            "prompt": "camera fly-through",
                            "images": [
                                {
                                    "input_handle": handle_id,
                                    "frame_index": 0,
                                    "strength": 1.0,
                                }
                            ],
                        },
                        "params": {"width": 768, "height": 512, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                        "extensions": {"simulate_delay_seconds": 0.02},
                    },
                )
                self.assertEqual(submit.status_code, 200, submit.text)
                job_id = response_model(submit, JobSubmitResult).job_id

                terminal = wait_for_job_terminal_state(client, job_id)
                self.assertEqual(terminal["state"], "completed")
                artifact_id = first_artifact_id(terminal)

                output_record = client.get(f"/v1/outputs/{artifact_id}")
                self.assertEqual(output_record.status_code, 200)
                download = client.get(f"/v1/outputs/{artifact_id}/download")
                self.assertEqual(download.status_code, 200)
                self.assertIn(b"MLXR scaffold output", download.content)

                events = client.get(f"/v1/jobs/{job_id}/events")
                self.assertEqual(events.status_code, 200)
                self.assertIn("event: job.accepted", events.text)
                self.assertIn("event: job.completed", events.text)

    def test_job_validation_and_admission_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(client, source_dir)

                first_job = client.post(
                    "/v1/jobs",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "task": "video.generate",
                        "inputs": {"prompt": "city skyline"},
                        "params": {"width": 768, "height": 512, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                        "extensions": {"simulate_delay_seconds": 0.15},
                    },
                )
                self.assertEqual(first_job.status_code, 200, first_job.text)

                second_job = client.post(
                    "/v1/jobs",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "task": "video.generate",
                        "inputs": {"prompt": "second request"},
                        "params": {"width": 768, "height": 512, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(second_job.status_code, 409)

                invalid_frames = client.post(
                    "/v1/jobs",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "task": "video.generate",
                        "inputs": {"prompt": "bad frames"},
                        "params": {"width": 768, "height": 512, "num_frames": 10},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(invalid_frames.status_code, 400)

                invalid_handle = client.post(
                    "/v1/jobs",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "task": "video.condition.image",
                        "inputs": {
                            "prompt": "missing handle",
                            "images": [{"input_handle": "inp_missing"}],
                        },
                        "params": {"width": 768, "height": 512, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(invalid_handle.status_code, 400)

                first_job_id = response_model(first_job, JobSubmitResult).job_id
                terminal = wait_for_job_terminal_state(client, first_job_id)
                self.assertEqual(terminal["state"], "completed")

    def test_job_fails_when_negative_prompt_is_requested_for_fast_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(client, source_dir)

                submit = client.post(
                    "/v1/jobs",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "task": "video.generate",
                        "inputs": {
                            "prompt": "city skyline",
                            "negative_prompt": "blurry",
                        },
                        "params": {"width": 768, "height": 512, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(submit.status_code, 200, submit.text)
                job_id = response_model(submit, JobSubmitResult).job_id

                terminal = wait_for_job_terminal_state(client, job_id)
                self.assertEqual(terminal["state"], "failed")
                error = terminal.get("error")
                self.assertIsInstance(error, str)
                assert isinstance(error, str)
                self.assertIn("does not support negative_prompt yet", error)

    def test_job_cancellation_marks_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(client, source_dir)
                submit = client.post(
                    "/v1/jobs",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "task": "video.generate",
                        "inputs": {"prompt": "cancel me"},
                        "params": {"width": 768, "height": 512, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                        "extensions": {"simulate_delay_seconds": 0.15},
                    },
                )
                self.assertEqual(submit.status_code, 200, submit.text)
                job_id = response_model(submit, JobSubmitResult).job_id

                cancel = client.post(f"/v1/jobs/{job_id}/cancel")
                self.assertEqual(cancel.status_code, 200)

                terminal = wait_for_job_terminal_state(client, job_id)
                self.assertEqual(terminal["state"], "cancelled")

                events = client.get(f"/v1/jobs/{job_id}/events")
                self.assertEqual(events.status_code, 200)
                self.assertIn("event: job.cancelled", events.text)

    def test_output_export_is_trusted_local_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            export_path = root / "exports" / "result.mp4"
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(client, source_dir)
                submit = client.post(
                    "/v1/jobs",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "task": "video.generate",
                        "inputs": {"prompt": "export local"},
                        "params": {"width": 768, "height": 512, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(submit.status_code, 200, submit.text)
                terminal = wait_for_job_terminal_state(
                    client, response_model(submit, JobSubmitResult).job_id
                )
                artifact_id = first_artifact_id(terminal)

                export = client.post(
                    f"/v1/outputs/{artifact_id}/export",
                    json={"destination_path": str(export_path)},
                )
                self.assertEqual(export.status_code, 200, export.text)
                self.assertTrue(export_path.exists())

            http_state = make_state(
                root,
                settings=ServerSettings(
                    http_enabled=True,
                    http_host="127.0.0.1",
                    http_bearer_token="secret-token",
                    allowed_origins=("http://localhost:3000",),
                ),
            )
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                TestClient(create_app(http_state)) as client,
            ):
                register_local_ltx_model(
                    client,
                    source_dir,
                    headers=http_headers(origin="http://localhost:3000"),
                )
                submit = client.post(
                    "/v1/jobs",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "task": "video.generate",
                        "inputs": {"prompt": "export denied"},
                        "params": {"width": 768, "height": 512, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                    headers=http_headers(origin="http://localhost:3000"),
                )
                self.assertEqual(submit.status_code, 200, submit.text)
                terminal = wait_for_job_terminal_state(
                    client, response_model(submit, JobSubmitResult).job_id
                )
                artifact_id = first_artifact_id(terminal)
                export = client.post(
                    f"/v1/outputs/{artifact_id}/export",
                    json={"destination_path": str(root / "exports" / "blocked.mp4")},
                    headers=http_headers(origin="http://localhost:3000"),
                )
                self.assertEqual(export.status_code, 400)

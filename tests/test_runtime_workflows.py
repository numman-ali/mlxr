from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from mlx_runtime_schemas import WorkflowPlanResult, WorkflowRunResult
from mlx_runtime_server.app import create_app

from tests.runtime_test_support import (
    import_input_handle,
    make_local_bundle,
    make_png_bytes,
    make_state,
    patched_inline_job_process_context,
    patched_ltx_prompt_encoder,
    patched_ltx_video_generator,
    register_local_ltx_model,
    response_model,
    wait_for_job_terminal_state,
)


class RuntimeWorkflowTests(unittest.TestCase):
    def test_workflow_plan_selects_text_to_video_for_prompt_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            source_dir = make_local_bundle(root)
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(client, source_dir)

                response = client.post(
                    "/v1/workflows/plan",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "prompt": "golden retriever in a park",
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(result.plan.selected_task, "video.generate")
                self.assertEqual(result.plan.family, "ltx")
                self.assertEqual(result.plan.pipeline_variant, "distilled_two_stage")

    def test_workflow_plan_selects_image_conditioning_when_image_reference_exists(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            source_dir = make_local_bundle(root)
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(client, source_dir)
                handle_id = import_input_handle(client, make_png_bytes())

                response = client.post(
                    "/v1/workflows/plan",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "prompt": "golden retriever in a park",
                        "references": [
                            {
                                "input_handle": handle_id,
                                "kind": "image",
                                "metadata": {"frame_index": 0, "strength": 1.0},
                            }
                        ],
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(result.plan.selected_task, "video.condition.image")

    def test_workflow_plan_rejects_audio_reference_until_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            source_dir = make_local_bundle(root)
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(client, source_dir)
                audio_handle = import_input_handle(
                    client,
                    b"RIFFfake",
                    media_type="audio/wav",
                )

                response = client.post(
                    "/v1/workflows/plan",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "prompt": "dog barking",
                        "references": [{"input_handle": audio_handle, "kind": "audio"}],
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("does not support audio references yet", response.text)

    def test_workflow_run_submits_a_job_and_job_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                patched_ltx_video_generator(include_audio=True),
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(client, source_dir)

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "golden retriever in a park",
                            "params": {
                                "width": 96,
                                "height": 64,
                                "num_frames": 9,
                                "fps": 12,
                                "seed": 17,
                            },
                            "output": {"artifact_format": "mp4"},
                        }
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowRunResult)
                terminal = wait_for_job_terminal_state(client, result.submit.job_id)
                self.assertEqual(terminal["state"], "completed")

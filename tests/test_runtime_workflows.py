from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from mlx_runtime_schemas import WorkflowPlanResult, WorkflowRunResult
from mlx_runtime_server.app import create_app
from mlx_runtime_workflows import WorkflowPlanner, WorkflowStrategyRegistry

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

    def test_workflow_plan_resolved_prompt_applies_text_first_audio_preferences(
        self,
    ) -> None:
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
                        "prompt": "golden retriever with its owner in a park",
                        "audio_prompt": "happy barking and light footsteps",
                        "preferences": {
                            "natural_audio": True,
                            "no_music": True,
                            "duration_seconds": 10.0,
                            "orientation": "landscape",
                        },
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertIn(
                    "Audio details: happy barking and light footsteps",
                    result.plan.resolved_prompt,
                )
                self.assertIn(
                    "Audio direction: use only natural diegetic environmental sound",
                    result.plan.resolved_prompt,
                )
                self.assertIn(
                    "Audio prohibition: no soundtrack, no score, no background music",
                    result.plan.resolved_prompt,
                )
                self.assertIn("no piano", result.plan.resolved_prompt)
                self.assertIn(
                    "Target duration: about 10.0 seconds.", result.plan.resolved_prompt
                )
                self.assertIn(
                    "Framing preference: landscape composition.",
                    result.plan.resolved_prompt,
                )
                self.assertIn(
                    "Natural-audio preference is text-first guidance only",
                    "\n".join(result.plan.warnings),
                )
                self.assertIn(
                    "may still drift toward soundtrack-like audio",
                    "\n".join(result.plan.warnings),
                )
                resolved_negative_prompt = result.plan.metadata.get(
                    "resolved_negative_prompt"
                )
                self.assertIsInstance(resolved_negative_prompt, str)
                assert isinstance(resolved_negative_prompt, str)
                self.assertIn("background music", resolved_negative_prompt)
                self.assertIn("chimes", resolved_negative_prompt)

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

    def test_workflow_plan_selects_audio_conditioning_when_audio_reference_exists(
        self,
    ) -> None:
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
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(result.plan.selected_task, "video.condition.audio")

    def test_workflow_plan_rejects_unsupported_lora_reference(self) -> None:
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
                        "prompt": "dog in a park",
                        "references": [{"kind": "lora"}],
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("does not support lora references yet", response.text)

    def test_workflow_plan_rejects_wrong_media_type_for_bound_handle(self) -> None:
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
                        "prompt": "dog in a park",
                        "references": [{"input_handle": audio_handle, "kind": "image"}],
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("must use an image media type", response.text)

    def test_workflow_run_submits_a_job_and_job_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                patched_ltx_video_generator(include_audio=True) as generators,
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(client, source_dir)

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "golden retriever in a park",
                            "audio_prompt": "happy barking and light footsteps",
                            "preferences": {
                                "natural_audio": True,
                                "no_music": True,
                            },
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
                self.assertTrue(generators[0].calls[0]["negative_prompt_present"])

    def test_workflow_run_supports_audio_conditioning_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                patched_ltx_video_generator(include_audio=True) as generators,
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(client, source_dir)
                audio_handle = import_input_handle(
                    client,
                    b"RIFFfake",
                    media_type="audio/wav",
                )

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "happy dog barking in a park",
                            "references": [
                                {
                                    "input_handle": audio_handle,
                                    "kind": "audio",
                                }
                            ],
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
                self.assertTrue(generators)
                self.assertTrue(generators[0].calls[0]["audio_conditioned"])

    def test_workflow_run_rejects_client_supplied_plan_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            source_dir = make_local_bundle(root)
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(client, source_dir)

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "golden retriever in a park",
                            "params": {"width": 96, "height": 64, "num_frames": 9},
                            "output": {"artifact_format": "mp4"},
                        },
                        "plan": {
                            "model_id": "ltx-2.3-fast-local",
                            "family": "ltx",
                            "selected_task": "video.generate",
                            "resolved_prompt": "tampered prompt",
                        },
                    },
                )
                self.assertEqual(response.status_code, 422)

    def test_workflow_plan_returns_not_implemented_when_family_has_no_strategy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            state.workflow_planner = WorkflowPlanner(WorkflowStrategyRegistry())
            state.workflow_service.planner = state.workflow_planner
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
                self.assertEqual(response.status_code, 501)
                self.assertIn("No workflow strategy is registered", response.text)

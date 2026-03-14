from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from mlxr.core.schemas import WorkflowPlanResult, WorkflowRunResult
from mlxr.core.server.app import create_app
from mlxr.core.workflows import WorkflowPlanner, WorkflowStrategyRegistry

from tests.runtime_test_support import (
    LTX_DEV_CHECKPOINT_FILENAME,
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
                self.assertEqual(result.presentation.primary_mode, "video")
                self.assertEqual(result.presentation.selected_task, "video.generate")

    def test_workflow_plan_selects_one_stage_for_dev_checkpoint_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            source_dir = make_local_bundle(
                root,
                directory_name="ltx-dev-bundle",
                checkpoint_filename=LTX_DEV_CHECKPOINT_FILENAME,
                include_spatial_upsampler=False,
            )
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(
                    client,
                    source_dir,
                    model_id="ltx-2.3-dev-local",
                )

                response = client.post(
                    "/v1/workflows/plan",
                    json={
                        "model_id": "ltx-2.3-dev-local",
                        "prompt": "golden retriever in a park",
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(result.plan.selected_task, "video.generate")
                self.assertEqual(result.plan.pipeline_variant, "one_stage")

    def test_workflow_plan_keeps_user_prompt_verbatim(self) -> None:
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
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(
                    result.plan.resolved_prompt,
                    "golden retriever with its owner in a park",
                )
                self.assertEqual(result.plan.warnings, [])

    def test_workflow_plan_rejects_legacy_prompt_authoring_fields(
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
                        "prompt": "Two people speak quietly inside a small bookshop.",
                        "audio_prompt": "soft page turns and hushed room tone",
                        "preferences": {
                            "natural_audio": True,
                            "no_music": True,
                        },
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 422)

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
                self.assertEqual(
                    [slot.slot_id for slot in result.presentation.reference_slots],
                    ["start-frame"],
                )

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
                self.assertEqual(
                    [slot.slot_id for slot in result.presentation.reference_slots],
                    ["audio-guide"],
                )

    def test_workflow_plan_keeps_image_reference_when_audio_and_image_exist(
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
                image_handle = import_input_handle(client, make_png_bytes())

                response = client.post(
                    "/v1/workflows/plan",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "prompt": "elderly man fishing on a pier at sunset",
                        "references": [
                            {"input_handle": image_handle, "kind": "image"},
                            {"input_handle": audio_handle, "kind": "audio"},
                        ],
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(result.plan.selected_task, "video.condition.audio")
                self.assertEqual(
                    sorted(reference.kind for reference in result.plan.references),
                    ["audio", "image"],
                )
                self.assertEqual(
                    result.presentation.selected_task, "video.condition.audio"
                )
                self.assertEqual(
                    [slot.slot_id for slot in result.presentation.reference_slots],
                    ["audio-guide"],
                )

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
                self.assertIn(
                    "LoRA references are currently only supported for video.condition.video",
                    response.text,
                )

    def test_workflow_plan_rejects_unsupported_workflow_variant(self) -> None:
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
                        "extensions": {"ltx": {"workflow_variant": "two_stage"}},
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(
                    "workflow variant 'two_stage' is not supported",
                    response.text,
                )

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
                self.assertFalse(generators[0].calls[0]["negative_prompt_present"])
                self.assertEqual(
                    generators[0].calls[0]["guidance_mode"], "positive_only"
                )

    def test_workflow_run_can_opt_in_family_local_distilled_cfg_guidance(
        self,
    ) -> None:
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
                            "prompt": "quiet bookshop conversation",
                            "extensions": {
                                "ltx": {
                                    "distilled_guidance_mode": "cfg",
                                }
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
                self.assertEqual(generators[0].calls[0]["guidance_mode"], "cfg")

    def test_workflow_run_forwards_pipeline_variant_and_inference_params(
        self,
    ) -> None:
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
                            "prompt": "neon alley samurai",
                            "extensions": {
                                "ltx": {
                                    "workflow_variant": "distilled_two_stage",
                                }
                            },
                            "params": {
                                "width": 96,
                                "height": 64,
                                "num_frames": 9,
                                "fps": 12,
                                "seed": 17,
                                "num_inference_steps": 12,
                                "guidance_scale": 2.5,
                            },
                            "output": {"artifact_format": "mp4"},
                        }
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowRunResult)
                terminal = wait_for_job_terminal_state(client, result.submit.job_id)
                self.assertEqual(terminal["state"], "completed")
                self.assertEqual(
                    generators[0].calls[0]["pipeline_variant"],
                    "distilled_two_stage",
                )
                self.assertEqual(generators[0].calls[0]["num_inference_steps"], 12)
                self.assertEqual(generators[0].calls[0]["guidance_scale"], 2.5)

    def test_workflow_run_uses_one_stage_pipeline_for_dev_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(
                root,
                directory_name="ltx-dev-bundle",
                checkpoint_filename=LTX_DEV_CHECKPOINT_FILENAME,
                include_spatial_upsampler=False,
            )
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                patched_ltx_video_generator(include_audio=True) as generators,
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(
                    client,
                    source_dir,
                    model_id="ltx-2.3-dev-local",
                )

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-dev-local",
                            "prompt": "lantern festival over the river",
                            "params": {
                                "width": 96,
                                "height": 64,
                                "num_frames": 9,
                                "fps": 12,
                                "seed": 17,
                                "num_inference_steps": 24,
                                "guidance_scale": 4.0,
                            },
                            "output": {"artifact_format": "mp4"},
                        }
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowRunResult)
                terminal = wait_for_job_terminal_state(client, result.submit.job_id)
                self.assertEqual(terminal["state"], "completed")
                self.assertEqual(
                    generators[0].calls[0]["pipeline_variant"], "one_stage"
                )
                self.assertEqual(generators[0].calls[0]["num_inference_steps"], 24)
                self.assertEqual(generators[0].calls[0]["guidance_scale"], 4.0)

    def test_workflow_plan_allows_explicit_two_stage_for_dev_model_with_assets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            source_dir = make_local_bundle(
                root,
                directory_name="ltx-dev-two-stage-bundle",
                checkpoint_filename=LTX_DEV_CHECKPOINT_FILENAME,
                include_spatial_upsampler=True,
                include_distilled_lora=True,
            )
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(
                    client,
                    source_dir,
                    model_id="ltx-2.3-dev-two-stage-local",
                )

                response = client.post(
                    "/v1/workflows/plan",
                    json={
                        "model_id": "ltx-2.3-dev-two-stage-local",
                        "prompt": "lantern festival over the river",
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "extensions": {"ltx": {"workflow_variant": "two_stage"}},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(result.plan.pipeline_variant, "two_stage")

    def test_workflow_run_uses_two_stage_pipeline_for_dev_model_with_assets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(
                root,
                directory_name="ltx-dev-two-stage-bundle",
                checkpoint_filename=LTX_DEV_CHECKPOINT_FILENAME,
                include_spatial_upsampler=True,
                include_distilled_lora=True,
            )
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                patched_ltx_video_generator(include_audio=True) as generators,
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(
                    client,
                    source_dir,
                    model_id="ltx-2.3-dev-two-stage-local",
                )

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-dev-two-stage-local",
                            "prompt": "lantern festival over the river",
                            "params": {
                                "width": 96,
                                "height": 64,
                                "num_frames": 9,
                                "fps": 12,
                                "seed": 17,
                                "num_inference_steps": 30,
                                "guidance_scale": 3.0,
                            },
                            "extensions": {"ltx": {"workflow_variant": "two_stage"}},
                            "output": {"artifact_format": "mp4"},
                        }
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowRunResult)
                terminal = wait_for_job_terminal_state(client, result.submit.job_id)
                self.assertEqual(terminal["state"], "completed")
                self.assertEqual(
                    generators[0].calls[0]["pipeline_variant"], "two_stage"
                )

    def test_workflow_plan_allows_explicit_two_stage_hq_for_dev_model_with_assets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            source_dir = make_local_bundle(
                root,
                directory_name="ltx-dev-two-stage-hq-bundle",
                checkpoint_filename=LTX_DEV_CHECKPOINT_FILENAME,
                include_spatial_upsampler=True,
                include_distilled_lora=True,
            )
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(
                    client,
                    source_dir,
                    model_id="ltx-2.3-dev-two-stage-hq-local",
                )

                response = client.post(
                    "/v1/workflows/plan",
                    json={
                        "model_id": "ltx-2.3-dev-two-stage-hq-local",
                        "prompt": "lantern festival over the river",
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "extensions": {"ltx": {"workflow_variant": "two_stage_hq"}},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(result.plan.pipeline_variant, "two_stage_hq")

    def test_workflow_plan_defaults_interpolation_to_two_stage_for_dev_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            source_dir = make_local_bundle(
                root,
                directory_name="ltx-dev-interpolation-bundle",
                checkpoint_filename=LTX_DEV_CHECKPOINT_FILENAME,
                include_spatial_upsampler=True,
                include_distilled_lora=True,
            )
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(
                    client,
                    source_dir,
                    model_id="ltx-2.3-dev-interpolation-local",
                )
                first_image = import_input_handle(client, make_png_bytes())
                last_image = import_input_handle(
                    client,
                    make_png_bytes(color=(180, 64, 112)),
                )

                response = client.post(
                    "/v1/workflows/plan",
                    json={
                        "model_id": "ltx-2.3-dev-interpolation-local",
                        "prompt": "interpolate between the two hero keyframes",
                        "task": "video.interpolate",
                        "references": [
                            {
                                "input_handle": first_image,
                                "kind": "image",
                                "metadata": {"frame_index": 0, "strength": 1.0},
                            },
                            {
                                "input_handle": last_image,
                                "kind": "image",
                                "metadata": {"frame_index": 8, "strength": 1.0},
                            },
                        ],
                        "params": {"width": 96, "height": 64, "num_frames": 9},
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(result.plan.selected_task, "video.interpolate")
                self.assertEqual(result.plan.pipeline_variant, "two_stage")
                self.assertEqual(
                    [slot.slot_id for slot in result.presentation.reference_slots],
                    ["start-frame", "end-frame"],
                )

    def test_workflow_run_forwards_interpolation_task_to_two_stage_pipeline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(
                root,
                directory_name="ltx-dev-interpolation-bundle",
                checkpoint_filename=LTX_DEV_CHECKPOINT_FILENAME,
                include_spatial_upsampler=True,
                include_distilled_lora=True,
            )
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                patched_ltx_video_generator(include_audio=True) as generators,
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(
                    client,
                    source_dir,
                    model_id="ltx-2.3-dev-interpolation-local",
                )
                first_image = import_input_handle(client, make_png_bytes())
                last_image = import_input_handle(
                    client,
                    make_png_bytes(color=(180, 64, 112)),
                )

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-dev-interpolation-local",
                            "prompt": "interpolate between the two hero keyframes",
                            "task": "video.interpolate",
                            "references": [
                                {
                                    "input_handle": first_image,
                                    "kind": "image",
                                    "metadata": {"frame_index": 0, "strength": 1.0},
                                },
                                {
                                    "input_handle": last_image,
                                    "kind": "image",
                                    "metadata": {"frame_index": 8, "strength": 1.0},
                                },
                            ],
                            "params": {
                                "width": 96,
                                "height": 64,
                                "num_frames": 9,
                                "fps": 12,
                                "seed": 17,
                                "num_inference_steps": 30,
                            },
                            "output": {"artifact_format": "mp4"},
                        }
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowRunResult)
                terminal = wait_for_job_terminal_state(client, result.submit.job_id)
                self.assertEqual(terminal["state"], "completed")
                self.assertEqual(generators[0].calls[0]["task"], "video.interpolate")
                self.assertEqual(
                    generators[0].calls[0]["pipeline_variant"], "two_stage"
                )
                self.assertEqual(generators[0].calls[0]["conditioning_count"], 2)

    def test_workflow_run_uses_two_stage_hq_pipeline_for_dev_model_with_assets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(
                root,
                directory_name="ltx-dev-two-stage-hq-bundle",
                checkpoint_filename=LTX_DEV_CHECKPOINT_FILENAME,
                include_spatial_upsampler=True,
                include_distilled_lora=True,
            )
            state = make_state(root)
            with (
                patched_inline_job_process_context(),
                patched_ltx_prompt_encoder(),
                patched_ltx_video_generator(include_audio=True) as generators,
                TestClient(create_app(state)) as client,
            ):
                register_local_ltx_model(
                    client,
                    source_dir,
                    model_id="ltx-2.3-dev-two-stage-hq-local",
                )

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-dev-two-stage-hq-local",
                            "prompt": "lantern festival over the river",
                            "params": {
                                "width": 96,
                                "height": 64,
                                "num_frames": 9,
                                "fps": 12,
                                "seed": 17,
                                "num_inference_steps": 15,
                                "guidance_scale": 3.0,
                            },
                            "extensions": {"ltx": {"workflow_variant": "two_stage_hq"}},
                            "output": {"artifact_format": "mp4"},
                        }
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowRunResult)
                terminal = wait_for_job_terminal_state(client, result.submit.job_id)
                self.assertEqual(terminal["state"], "completed")
                self.assertEqual(
                    generators[0].calls[0]["pipeline_variant"], "two_stage_hq"
                )

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

    def test_workflow_plan_supports_explicit_retake_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = make_state(root)
            source_dir = make_local_bundle(root)
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(client, source_dir)
                video_handle = import_input_handle(
                    client,
                    b"mp4",
                    media_type="video/mp4",
                )

                response = client.post(
                    "/v1/workflows/plan",
                    json={
                        "model_id": "ltx-2.3-fast-local",
                        "prompt": "replace the middle beat with a dramatic sword draw",
                        "task": "video.retake",
                        "references": [
                            {
                                "input_handle": video_handle,
                                "kind": "video",
                            }
                        ],
                        "params": {
                            "width": 96,
                            "height": 64,
                            "num_frames": 9,
                            "window_start_seconds": 1.25,
                            "window_end_seconds": 2.75,
                            "regenerate_audio": False,
                        },
                        "output": {"artifact_format": "mp4"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowPlanResult)
                self.assertEqual(result.plan.selected_task, "video.retake")
                self.assertEqual(
                    [slot.slot_id for slot in result.presentation.reference_slots],
                    ["source-video"],
                )

    def test_workflow_run_forwards_retake_video_and_window_params(self) -> None:
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
                video_handle = import_input_handle(
                    client,
                    b"mp4",
                    media_type="video/mp4",
                )

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "replace the middle beat with a dramatic sword draw",
                            "task": "video.retake",
                            "references": [
                                {
                                    "input_handle": video_handle,
                                    "kind": "video",
                                }
                            ],
                            "params": {
                                "width": 96,
                                "height": 64,
                                "num_frames": 9,
                                "fps": 12,
                                "seed": 17,
                                "window_start_seconds": 1.25,
                                "window_end_seconds": 2.75,
                                "regenerate_video": True,
                                "regenerate_audio": False,
                            },
                            "output": {"artifact_format": "mp4"},
                        }
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowRunResult)
                terminal = wait_for_job_terminal_state(client, result.submit.job_id)
                self.assertEqual(terminal["state"], "completed")
                self.assertEqual(generators[0].calls[0]["task"], "video.retake")
                self.assertEqual(generators[0].calls[0]["video_input_count"], 1)
                self.assertEqual(
                    generators[0].calls[0]["retake_options"],
                    {
                        "start_time_seconds": 1.25,
                        "end_time_seconds": 2.75,
                        "regenerate_video": True,
                        "regenerate_audio": False,
                    },
                )

    def test_workflow_run_defaults_retake_regenerate_flags_when_omitted(self) -> None:
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
                video_handle = import_input_handle(
                    client,
                    b"mp4",
                    media_type="video/mp4",
                )

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "replace the middle beat with a dramatic sword draw",
                            "task": "video.retake",
                            "references": [
                                {
                                    "input_handle": video_handle,
                                    "kind": "video",
                                }
                            ],
                            "params": {
                                "width": 96,
                                "height": 64,
                                "num_frames": 9,
                                "fps": 12,
                                "seed": 17,
                                "window_start_seconds": 1.25,
                                "window_end_seconds": 2.75,
                            },
                            "output": {"artifact_format": "mp4"},
                        }
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response_model(response, WorkflowRunResult)
                terminal = wait_for_job_terminal_state(client, result.submit.job_id)
                self.assertEqual(terminal["state"], "completed")
                self.assertEqual(
                    generators[0].calls[0]["retake_options"],
                    {
                        "start_time_seconds": 1.25,
                        "end_time_seconds": 2.75,
                        "regenerate_video": True,
                        "regenerate_audio": True,
                    },
                )

    def test_workflow_run_supports_combined_image_and_audio_conditioning(self) -> None:
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
                image_handle = import_input_handle(client, make_png_bytes())

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "elderly man fishing on a pier at sunset",
                            "references": [
                                {"input_handle": image_handle, "kind": "image"},
                                {"input_handle": audio_handle, "kind": "audio"},
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
                self.assertEqual(generators[0].calls[0]["conditioning_count"], 1)

    def test_workflow_run_supports_video_condition_video_with_lora(self) -> None:
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
                video_handle = import_input_handle(
                    client,
                    b"mp4",
                    media_type="video/mp4",
                )
                lora_handle = import_input_handle(
                    client,
                    b"lora",
                    media_type="application/x-safetensors",
                    filename="control.safetensors",
                )

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "fox made of fire running through snow",
                            "task": "video.condition.video",
                            "references": [
                                {
                                    "input_handle": video_handle,
                                    "kind": "video",
                                    "metadata": {"strength": 0.8},
                                },
                                {
                                    "input_handle": lora_handle,
                                    "kind": "lora",
                                    "metadata": {"strength": 0.6},
                                },
                            ],
                            "extensions": {
                                "ltx": {
                                    "control_variant": "ic_lora",
                                    "conditioning_attention_strength": 0.5,
                                }
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
                self.assertTrue(generators)
                self.assertEqual(
                    generators[0].calls[0]["task"], "video.condition.video"
                )
                self.assertEqual(generators[0].calls[0]["video_input_count"], 1)
                self.assertEqual(generators[0].calls[0]["lora_input_count"], 1)

    def test_workflow_run_supports_multiple_keyed_image_conditionings(self) -> None:
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
                first_image_handle = import_input_handle(
                    client,
                    make_png_bytes(color=(36, 108, 196)),
                )
                last_image_handle = import_input_handle(
                    client,
                    make_png_bytes(color=(196, 92, 36)),
                )

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "samurai in a rainy alley with fire blooming around him",
                            "references": [
                                {
                                    "input_handle": first_image_handle,
                                    "kind": "image",
                                    "metadata": {"frame_index": 0, "strength": 1.0},
                                },
                                {
                                    "input_handle": last_image_handle,
                                    "kind": "image",
                                    "metadata": {"frame_index": 8, "strength": 0.75},
                                },
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
                self.assertEqual(generators[0].calls[0]["conditioning_count"], 2)
                self.assertEqual(
                    generators[0].calls[0]["conditioning_frame_indices"],
                    [0, 8],
                )
                self.assertEqual(
                    generators[0].calls[0]["conditioning_strengths"],
                    [1.0, 0.75],
                )

    def test_workflow_run_rejects_control_variant_outside_video_condition_video(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(client, source_dir)

                response = client.post(
                    "/v1/workflows/run",
                    json={
                        "intent": {
                            "model_id": "ltx-2.3-fast-local",
                            "prompt": "dog in a park",
                            "extensions": {
                                "ltx": {
                                    "control_variant": "ic_lora",
                                    "conditioning_attention_strength": 0.5,
                                }
                            },
                            "params": {
                                "width": 96,
                                "height": 64,
                                "num_frames": 9,
                            },
                            "output": {"artifact_format": "mp4"},
                        }
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(
                    "only valid for video.condition.video workflows",
                    response.text,
                )

    def test_workflow_run_forwards_video_condition_video_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(root)
            with TestClient(create_app(state)) as client:
                register_local_ltx_model(client, source_dir)
                video_handle = import_input_handle(
                    client,
                    b"mp4",
                    media_type="video/mp4",
                )
                lora_handle = import_input_handle(
                    client,
                    b"lora",
                    media_type="application/x-safetensors",
                    filename="control.safetensors",
                )

                with (
                    patched_inline_job_process_context(),
                    patched_ltx_prompt_encoder(),
                    patched_ltx_video_generator(include_audio=False) as generators,
                ):
                    response = client.post(
                        "/v1/workflows/run",
                        json={
                            "intent": {
                                "model_id": "ltx-2.3-fast-local",
                                "prompt": "match the reference movement with a different hero",
                                "task": "video.condition.video",
                                "references": [
                                    {
                                        "kind": "video",
                                        "input_handle": video_handle,
                                        "metadata": {"strength": 1.0},
                                    },
                                    {
                                        "kind": "lora",
                                        "input_handle": lora_handle,
                                        "metadata": {"strength": 0.75},
                                    },
                                ],
                                "extensions": {
                                    "ltx": {
                                        "conditioning_attention_strength": 0.5,
                                    }
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
                    self.assertTrue(generators)
                    self.assertEqual(
                        generators[0].calls[0]["task"], "video.condition.video"
                    )
                    self.assertEqual(generators[0].calls[0]["video_input_count"], 1)
                    self.assertEqual(generators[0].calls[0]["lora_input_count"], 1)
                    self.assertEqual(generators[0].calls[0]["lora_strengths"], [0.75])
                    self.assertEqual(
                        generators[0].calls[0]["control_variant"], "ic_lora"
                    )
                    self.assertEqual(
                        generators[0].calls[0]["conditioning_attention_strength"],
                        0.5,
                    )

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

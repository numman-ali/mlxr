from __future__ import annotations

import unittest

from mlxr.core.schemas import (
    CapabilityDescriptor,
    JobOutputPolicy,
    ModelRecord,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowReference,
)
from mlxr.core.workflows import WorkflowPlanningContext
from mlxr.families.ltx.workflows import LTXWorkflowStrategy


def _context(
    *,
    tasks: list[str] | None = None,
    conditioning: dict[str, bool] | None = None,
    pipeline_variants: list[str] | None = None,
) -> WorkflowPlanningContext:
    capability = CapabilityDescriptor(
        model_id="ltx-2.3-fast-local",
        artifact_digest="sha256:test",
        family="ltx",
        scheduler_class="media_video_dit",
        tasks=tasks
        or [
            "video.generate",
            "video.condition.image",
            "video.condition.video",
            "video.condition.audio",
        ],
        conditioning=conditioning
        or {"image": True, "video": True, "audio": True, "lora": True},
        profiles_by_task={
            "video.generate": ["bf16"],
            "video.condition.image": ["bf16"],
            "video.condition.video": ["bf16"],
            "video.condition.audio": ["bf16"],
        },
        metadata={
            "implemented_surface": {
                "pipeline_variants": pipeline_variants or ["distilled_two_stage"],
            }
        },
    )
    model = ModelRecord(
        model_id="ltx-2.3-fast-local",
        family="ltx",
        capability=capability,
    )
    return WorkflowPlanningContext(model=model, capability=capability)


class LTXWorkflowStrategyTests(unittest.TestCase):
    def test_plan_rejects_explicit_unsupported_task(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(tasks=["video.generate"])
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="dog in a park",
            task="video.condition.image",
        )

        with self.assertRaisesRegex(ValueError, "does not support workflow task"):
            strategy.plan(context, intent)

    def test_plan_rejects_explicit_control_reference_kinds_until_supported(
        self,
    ) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(
            conditioning={"image": True, "video": True, "audio": True, "lora": True}
        )
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="dog in a park",
            references=[WorkflowReference(kind="lora")],
        )

        with self.assertRaisesRegex(
            ValueError,
            "LoRA references are currently only supported for video.condition.video",
        ):
            strategy.plan(context, intent)

    def test_to_job_request_rejects_missing_image_handle(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="dog in a park",
            references=[WorkflowReference(kind="image")],
            output=JobOutputPolicy(artifact_format="mp4"),
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            selected_task="video.condition.image",
            resolved_prompt="dog in a park",
            pipeline_variant="distilled_two_stage",
            references=intent.references,
        )

        with self.assertRaisesRegex(ValueError, "requires bound input handles"):
            strategy.to_job_request(context, intent, plan)

    def test_to_job_request_rejects_invalid_audio_metadata(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="dog barking",
            references=[
                WorkflowReference(
                    kind="audio",
                    input_handle="audio-handle",
                    metadata={"start_time_seconds": -1.0},
                )
            ],
            output=JobOutputPolicy(artifact_format="mp4"),
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            selected_task="video.condition.audio",
            resolved_prompt="dog barking",
            pipeline_variant="distilled_two_stage",
            references=intent.references,
        )

        with self.assertRaisesRegex(ValueError, "must be >= 0.0"):
            strategy.to_job_request(context, intent, plan)

    def test_to_job_request_rejects_video_control_extensions_outside_video_condition_video(
        self,
    ) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="dog in a park",
            references=[
                WorkflowReference(
                    kind="image",
                    input_handle="image-handle",
                    metadata={"frame_index": 0, "strength": 1.0},
                )
            ],
            output=JobOutputPolicy(artifact_format="mp4"),
            extensions={
                "ltx": {
                    "control_variant": "ic_lora",
                    "conditioning_attention_strength": 0.5,
                }
            },
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            selected_task="video.condition.image",
            resolved_prompt="dog in a park",
            pipeline_variant="distilled_two_stage",
            references=intent.references,
        )

        with self.assertRaisesRegex(ValueError, "only valid for video.condition.video"):
            strategy.to_job_request(context, intent, plan)

    def test_to_job_request_carries_video_conditioning_lora_inputs(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(
            tasks=[
                "video.generate",
                "video.condition.image",
                "video.condition.video",
                "video.condition.audio",
                "video.retake",
            ],
            conditioning={"image": True, "video": True, "audio": True, "lora": True},
        )
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="match the reference motion but keep the new subject prompt",
            task="video.condition.video",
            references=[
                WorkflowReference(
                    kind="video",
                    input_handle="video-handle",
                    metadata={"strength": 1.0},
                ),
                WorkflowReference(
                    kind="lora",
                    input_handle="lora-handle",
                    metadata={"strength": 0.75},
                ),
            ],
            output=JobOutputPolicy(artifact_format="mp4"),
            extensions={"ltx": {"conditioning_attention_strength": 0.5}},
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            selected_task="video.condition.video",
            resolved_prompt=intent.prompt,
            pipeline_variant="distilled_two_stage",
            references=intent.references,
        )

        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(
            request.inputs["videos"],
            [{"input_handle": "video-handle", "strength": 1.0}],
        )
        self.assertEqual(
            request.inputs["loras"],
            [{"input_handle": "lora-handle", "strength": 0.75}],
        )
        self.assertEqual(request.extensions["ltx"]["control_variant"], "ic_lora")
        self.assertEqual(
            request.extensions["ltx"]["conditioning_attention_strength"],
            0.5,
        )

    def test_to_job_request_preserves_motion_track_control_variant(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(
            tasks=[
                "video.generate",
                "video.condition.image",
                "video.condition.video",
            ],
            conditioning={"image": True, "video": True, "audio": False, "lora": True},
        )
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="keep the camera trajectory but change the subject",
            task="video.condition.video",
            references=[
                WorkflowReference(
                    kind="video",
                    input_handle="video-handle",
                    metadata={"strength": 1.0},
                ),
                WorkflowReference(
                    kind="lora",
                    input_handle="motion-track-lora",
                    metadata={"strength": 1.0},
                ),
            ],
            output=JobOutputPolicy(artifact_format="mp4"),
            extensions={
                "ltx": {
                    "control_variant": "motion_track_control",
                    "conditioning_attention_strength": 0.75,
                }
            },
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            selected_task="video.condition.video",
            resolved_prompt=intent.prompt,
            pipeline_variant="distilled_two_stage",
            references=intent.references,
        )

        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(
            request.extensions["ltx"]["control_variant"],
            "motion_track_control",
        )
        self.assertEqual(
            request.extensions["ltx"]["conditioning_attention_strength"],
            0.75,
        )
        self.assertEqual(
            request.inputs["loras"],
            [{"input_handle": "motion-track-lora", "strength": 1.0}],
        )

    def test_plan_requires_explicit_task_for_video_references(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="fox made of fire",
            references=[
                WorkflowReference(kind="video", input_handle="video-handle"),
                WorkflowReference(
                    kind="lora",
                    input_handle="lora-handle",
                    metadata={"strength": 0.6},
                ),
            ],
        )

        with self.assertRaisesRegex(ValueError, "require an explicit task"):
            strategy.plan(context, intent)

    def test_to_job_request_carries_video_and_lora_inputs_for_video_condition_video(
        self,
    ) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="fox made of fire",
            references=[
                WorkflowReference(
                    kind="video",
                    input_handle="video-handle",
                    metadata={"strength": 0.8},
                ),
                WorkflowReference(
                    kind="lora",
                    input_handle="lora-handle",
                    metadata={"strength": 0.6},
                ),
            ],
            output=JobOutputPolicy(artifact_format="mp4"),
            extensions={
                "ltx": {
                    "control_variant": "ic_lora",
                    "conditioning_attention_strength": 0.5,
                }
            },
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            selected_task="video.condition.video",
            resolved_prompt="fox made of fire",
            pipeline_variant="distilled_two_stage",
            references=intent.references,
        )

        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(request.inputs["videos"][0]["input_handle"], "video-handle")
        self.assertEqual(request.inputs["videos"][0]["strength"], 0.8)
        self.assertEqual(request.inputs["loras"][0]["input_handle"], "lora-handle")
        self.assertEqual(request.inputs["loras"][0]["strength"], 0.6)

    def test_to_job_request_carries_keyed_image_metadata_and_workflow_variant(
        self,
    ) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="samurai in rain",
            references=[
                WorkflowReference(
                    kind="image",
                    input_handle="first-image",
                    metadata={"frame_index": 0, "strength": 1.0},
                ),
                WorkflowReference(
                    kind="image",
                    input_handle="last-image",
                    metadata={"frame_index": 8, "strength": 0.75},
                ),
            ],
            output=JobOutputPolicy(artifact_format="mp4"),
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            selected_task="video.condition.image",
            resolved_prompt="samurai in rain",
            pipeline_variant="distilled_two_stage",
            references=intent.references,
        )

        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(
            request.inputs["images"],
            [
                {
                    "input_handle": "first-image",
                    "frame_index": 0,
                    "strength": 1.0,
                },
                {
                    "input_handle": "last-image",
                    "frame_index": 8,
                    "strength": 0.75,
                },
            ],
        )
        self.assertEqual(
            request.extensions["ltx"]["workflow_variant"], "distilled_two_stage"
        )

    def test_to_job_request_carries_retake_video_and_window_params(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(
            tasks=[
                "video.generate",
                "video.condition.image",
                "video.condition.audio",
                "video.retake",
            ],
            conditioning={"image": True, "video": True, "audio": True, "lora": False},
        )
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="replace the middle beat with a dramatic sword draw",
            task="video.retake",
            references=[
                WorkflowReference(
                    kind="video",
                    input_handle="video-handle",
                    metadata={"strength": 1.0},
                )
            ],
            params={
                "window_start_seconds": 1.25,
                "window_end_seconds": 2.75,
                "regenerate_video": True,
                "regenerate_audio": False,
            },
            output=JobOutputPolicy(artifact_format="mp4"),
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            selected_task="video.retake",
            resolved_prompt=intent.prompt,
            pipeline_variant="distilled_two_stage",
            references=intent.references,
        )

        request = strategy.to_job_request(context, intent, plan)

        self.assertEqual(
            request.inputs["videos"],
            [{"input_handle": "video-handle", "strength": 1.0}],
        )
        self.assertEqual(request.params["window_start_seconds"], 1.25)
        self.assertEqual(request.params["window_end_seconds"], 2.75)
        self.assertEqual(request.params["regenerate_audio"], False)

    def test_plan_defaults_interpolation_to_two_stage(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(
            tasks=[
                "video.generate",
                "video.condition.image",
                "video.interpolate",
                "video.retake",
            ],
            conditioning={"image": True, "video": True, "audio": False, "lora": False},
            pipeline_variants=["one_stage", "two_stage", "two_stage_hq"],
        )
        intent = WorkflowIntent(
            model_id="ltx-2.3-dev-local",
            prompt="interpolate between the two keyframes",
            task="video.interpolate",
            references=[
                WorkflowReference(
                    kind="image",
                    input_handle="first-image",
                    metadata={"frame_index": 0, "strength": 1.0},
                ),
                WorkflowReference(
                    kind="image",
                    input_handle="last-image",
                    metadata={"frame_index": 16, "strength": 1.0},
                ),
            ],
        )

        plan = strategy.plan(context, intent)

        self.assertEqual(plan.selected_task, "video.interpolate")
        self.assertEqual(plan.pipeline_variant, "two_stage")

    def test_to_job_request_rejects_interpolation_without_two_images(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(
            tasks=["video.generate", "video.condition.image", "video.interpolate"],
            conditioning={"image": True, "video": False, "audio": False, "lora": False},
            pipeline_variants=["one_stage", "two_stage"],
        )
        intent = WorkflowIntent(
            model_id="ltx-2.3-dev-local",
            prompt="interpolate between keyframes",
            task="video.interpolate",
            references=[
                WorkflowReference(
                    kind="image",
                    input_handle="only-image",
                    metadata={"frame_index": 0, "strength": 1.0},
                )
            ],
            output=JobOutputPolicy(artifact_format="mp4"),
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-dev-local",
            family="ltx",
            selected_task="video.interpolate",
            resolved_prompt=intent.prompt,
            pipeline_variant="two_stage",
            references=intent.references,
        )

        with self.assertRaisesRegex(
            ValueError, "video.interpolate requires at least two bound image references"
        ):
            strategy.to_job_request(context, intent, plan)

    def test_readiness_requires_control_video_and_lora_for_video_guidance(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context()
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="match the guide motion but change the subject",
            task="video.condition.video",
            output=JobOutputPolicy(artifact_format="mp4"),
        )

        plan = strategy.plan(context, intent)
        readiness = strategy.readiness(context, intent, plan)

        self.assertFalse(readiness.ready)
        self.assertIn("Choose exactly one guide video.", readiness.blocking_issues)
        self.assertIn(
            "Choose exactly one compatible control LoRA.",
            readiness.blocking_issues,
        )

    def test_readiness_requires_distinct_keyframes_for_interpolation(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(
            tasks=["video.generate", "video.condition.image", "video.interpolate"],
            conditioning={"image": True, "video": False, "audio": False, "lora": False},
            pipeline_variants=["one_stage", "two_stage"],
        )
        intent = WorkflowIntent(
            model_id="ltx-2.3-dev-local",
            prompt="blend between the two poses",
            task="video.interpolate",
            references=[
                WorkflowReference(
                    kind="image",
                    input_handle="first-image",
                    metadata={"frame_index": 0, "strength": 1.0},
                ),
                WorkflowReference(
                    kind="image",
                    input_handle="second-image",
                    metadata={"frame_index": 0, "strength": 1.0},
                ),
            ],
        )

        plan = strategy.plan(context, intent)
        readiness = strategy.readiness(context, intent, plan)

        self.assertFalse(readiness.ready)
        self.assertIn(
            "Set at least two distinct keyframe positions for interpolation.",
            readiness.blocking_issues,
        )

    def test_readiness_requires_retake_window(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(
            tasks=["video.generate", "video.condition.image", "video.retake"],
            conditioning={"image": True, "video": True, "audio": False, "lora": False},
        )
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="replace the middle beat",
            task="video.retake",
            references=[WorkflowReference(kind="video", input_handle="video-handle")],
        )

        plan = strategy.plan(context, intent)
        readiness = strategy.readiness(context, intent, plan)

        self.assertFalse(readiness.ready)
        self.assertIn(
            "Set a valid retake window before running this workflow.",
            readiness.blocking_issues,
        )

    def test_to_job_request_rejects_retake_without_window_params(self) -> None:
        strategy = LTXWorkflowStrategy()
        context = _context(
            tasks=[
                "video.generate",
                "video.condition.image",
                "video.condition.audio",
                "video.retake",
            ],
            conditioning={"image": True, "video": True, "audio": True, "lora": False},
        )
        intent = WorkflowIntent(
            model_id="ltx-2.3-fast-local",
            prompt="replace the middle beat with a dramatic sword draw",
            task="video.retake",
            references=[
                WorkflowReference(
                    kind="video",
                    input_handle="video-handle",
                )
            ],
            output=JobOutputPolicy(artifact_format="mp4"),
        )
        plan = WorkflowPlan(
            model_id="ltx-2.3-fast-local",
            family="ltx",
            selected_task="video.retake",
            resolved_prompt=intent.prompt,
            pipeline_variant="distilled_two_stage",
            references=intent.references,
        )

        with self.assertRaisesRegex(
            ValueError, "video.retake requires numeric params.window_start_seconds"
        ):
            strategy.to_job_request(context, intent, plan)


if __name__ == "__main__":
    unittest.main()

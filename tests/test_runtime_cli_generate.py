from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlxr.clients.cli.cli import (
    RuntimeClient,
    _coerce_action_values,
    _default_uds_path,
    _extensions_from_args,
    _generation_params,
    _media_type_for_path,
    _parse_extensions_json,
    _references_from_args,
    _run_generate_command,
    _wait_for_terminal_job,
    _workflow_quality,
    build_parser,
)
from mlxr.core.schemas import (
    ArtifactExportResult,
    InputHandleRecord,
    JobOutputPolicy,
    JobRecord,
    JobRequest,
    JobState,
    JobSubmitResult,
    OutputArtifactRecord,
    WorkflowIntent,
    WorkflowPlan,
    WorkflowRunResult,
)


class _FakeClient(RuntimeClient):
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []
        self.jobs: list[JobRecord] = []
        self.export_calls: list[tuple[str, Path, bool]] = []

    def close(self) -> None:
        return None

    def import_file(self, path: Path, *, kind: str) -> InputHandleRecord:
        self.calls.append((path, kind))
        return InputHandleRecord(
            handle_id=f"{kind}-handle",
            media_type=None,
            role=kind,
            storage_key=f"inputs/{kind}-handle",
        )

    def run(self, intent: WorkflowIntent) -> WorkflowRunResult:
        del intent
        return WorkflowRunResult(
            plan=WorkflowPlan(
                model_id="ltx-2.3-fast-local",
                family="ltx",
                selected_task="video.generate",
                resolved_prompt="golden retriever in a park",
                warnings=[],
            ),
            submit=JobSubmitResult(
                job_id="job_1",
                record=JobRecord(
                    job_id="job_1",
                    request=JobRequest(
                        model_id="ltx-2.3-fast-local",
                        task="video.generate",
                        output=JobOutputPolicy(artifact_format="mp4"),
                    ),
                    state=JobState.ACCEPTED,
                ),
            ),
        )

    def get_job(self, job_id: str) -> JobRecord:
        del job_id
        return self.jobs.pop(0)

    def export_output(
        self, artifact_id: str, *, destination_path: Path, overwrite: bool
    ) -> ArtifactExportResult:
        self.export_calls.append((artifact_id, destination_path, overwrite))
        return ArtifactExportResult(
            artifact_id=artifact_id,
            destination_path=str(destination_path),
        )


class RuntimeCliGenerateTests(unittest.TestCase):
    def test_workflow_quality_maps_expected_values(self) -> None:
        self.assertEqual(_workflow_quality("auto"), "auto")
        self.assertEqual(_workflow_quality("fast"), "fast")
        self.assertEqual(_workflow_quality("balanced"), "balanced")
        self.assertEqual(_workflow_quality("high"), "high")
        with self.assertRaisesRegex(ValueError, "Unsupported workflow quality"):
            _workflow_quality("ultra")

    def test_generate_parser_accepts_simple_generation_args(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(
            [
                "generate",
                "--model-id",
                "ltx-2.3-fast-local",
                "--prompt",
                "golden retriever in a park",
                "--width",
                "96",
                "--height",
                "64",
                "--num-frames",
                "9",
                "--task",
                "video.generate",
                "--negative-prompt",
                "low quality",
                "--num-inference-steps",
                "8",
                "--guidance-scale",
                "1.5",
                "--artifact-format",
                "png",
            ]
        )
        self.assertEqual(parsed.command, "generate")
        self.assertEqual(parsed.model_id, "ltx-2.3-fast-local")
        self.assertEqual(parsed.prompt, "golden retriever in a park")
        self.assertEqual(parsed.task, "video.generate")
        self.assertEqual(parsed.negative_prompt, "low quality")
        self.assertEqual(parsed.width, 96)
        self.assertEqual(parsed.num_frames, 9)
        self.assertEqual(parsed.num_inference_steps, 8)
        self.assertEqual(parsed.guidance_scale, 1.5)
        self.assertEqual(parsed.artifact_format, "png")

    def test_generate_parser_accepts_local_image_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "conditioning.png"
            image_path.write_bytes(b"png")
            parser = build_parser()
            parsed = parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "dog in a park",
                    "--image",
                    str(image_path),
                    "--image-frame-index",
                    "8",
                    "--image-strength",
                    "0.75",
                    "--plan-only",
                ]
            )
        self.assertEqual(parsed.image, [image_path])
        self.assertEqual(parsed.image_frame_index, 8)
        self.assertEqual(parsed.image_strength, 0.75)
        self.assertTrue(parsed.plan_only)

    def test_generate_parser_accepts_keyframe_images_lora_and_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "first.png"
            lora_path = Path(tmp_dir) / "control.safetensors"
            extensions_path = Path(tmp_dir) / "extensions.json"
            image_path.write_bytes(b"png")
            lora_path.write_bytes(b"lora")
            extensions_path.write_text('{"ltx": {"workflow_variant": "two_stage"}}')
            parser = build_parser()
            parsed = parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "dog in a park",
                    "--keyframe-image",
                    str(image_path),
                    "16",
                    "0.8",
                    "--lora",
                    str(lora_path),
                    "0.6",
                    "--extensions-json",
                    f"@{extensions_path}",
                    "--plan-only",
                ]
            )
            self.assertEqual(
                _extensions_from_args(parsed),
                {"ltx": {"workflow_variant": "two_stage"}},
            )
        self.assertEqual(len(parsed.keyframe_image), 1)
        self.assertEqual(parsed.keyframe_image[0].path, image_path)
        self.assertEqual(parsed.keyframe_image[0].frame_index, 16)
        self.assertEqual(parsed.keyframe_image[0].strength, 0.8)
        self.assertEqual(len(parsed.lora), 1)
        self.assertEqual(parsed.lora[0].path, lora_path)
        self.assertEqual(parsed.lora[0].strength, 0.6)

    def test_generate_parser_accepts_local_video_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "conditioning.mp4"
            video_path.write_bytes(b"mp4")
            parser = build_parser()
            parsed = parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "dog in a park",
                    "--video",
                    str(video_path),
                    "--plan-only",
                ]
            )
        self.assertEqual(parsed.video, [video_path])
        self.assertTrue(parsed.plan_only)

    def test_generate_parser_accepts_retake_window_flags(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(
            [
                "generate",
                "--model-id",
                "ltx-2.3-fast-local",
                "--prompt",
                "replace the middle beat with a dramatic sword draw",
                "--task",
                "video.retake",
                "--video",
                "/tmp/source.mp4",
                "--window-start-seconds",
                "1.25",
                "--window-end-seconds",
                "2.75",
                "--no-regenerate-audio",
                "--plan-only",
            ]
        )
        self.assertEqual(parsed.task, "video.retake")
        self.assertEqual(parsed.window_start_seconds, 1.25)
        self.assertEqual(parsed.window_end_seconds, 2.75)
        self.assertTrue(parsed.no_regenerate_audio)

    def test_generate_parser_accepts_wait_and_export_flags(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(
            [
                "generate",
                "--model-id",
                "ltx-2.3-fast-local",
                "--prompt",
                "golden retriever in a park",
                "--wait",
                "--timeout-seconds",
                "30",
                "--poll-interval-seconds",
                "0.5",
                "--export-path",
                "/tmp/result.mp4",
                "--overwrite-export",
            ]
        )
        self.assertTrue(parsed.wait)
        self.assertEqual(parsed.timeout_seconds, 30.0)
        self.assertEqual(parsed.poll_interval_seconds, 0.5)
        self.assertEqual(parsed.export_path, Path("/tmp/result.mp4"))
        self.assertTrue(parsed.overwrite_export)

    def test_generate_parser_rejects_removed_prompt_authoring_flags(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--no-music",
                ]
            )

    def test_generate_parser_rejects_export_without_wait(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--export-path",
                    "/tmp/result.mp4",
                ]
            )

    def test_generate_parser_rejects_wait_with_plan_only(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--plan-only",
                    "--wait",
                ]
            )

    def test_generate_parser_rejects_invalid_wait_values(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--wait",
                    "--timeout-seconds",
                    "0",
                ]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--wait",
                    "--poll-interval-seconds",
                    "-0.1",
                ]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--image-frame-index",
                    "-1",
                ]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--image-strength",
                    "1.5",
                ]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--num-inference-steps",
                    "0",
                ]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--guidance-scale",
                    "-0.1",
                ]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--keyframe-image",
                    "/tmp/frame.png",
                    "8",
                    "--image-strength",
                    "0.5",
                ]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "generate",
                    "--model-id",
                    "ltx-2.3-fast-local",
                    "--prompt",
                    "golden retriever in a park",
                    "--extensions-json",
                    "[]",
                ]
            )

    def test_default_uds_path_uses_runtime_home_temp_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = Path(tmp_dir) / "runtime-home"
            previous = None
            import os

            previous = os.environ.get("MLX_RUNTIME_HOME")
            os.environ["MLX_RUNTIME_HOME"] = str(runtime_home)
            try:
                self.assertEqual(
                    _default_uds_path(),
                    runtime_home / "temp" / "control-plane.sock",
                )
            finally:
                if previous is None:
                    os.environ.pop("MLX_RUNTIME_HOME", None)
                else:
                    os.environ["MLX_RUNTIME_HOME"] = previous

    def test_default_uds_path_prefers_explicit_env_socket(self) -> None:
        import os

        previous = os.environ.get("MLX_RUNTIME_UDS_PATH")
        os.environ["MLX_RUNTIME_UDS_PATH"] = "/tmp/custom.sock"
        try:
            self.assertEqual(_default_uds_path(), Path("/tmp/custom.sock"))
        finally:
            if previous is None:
                os.environ.pop("MLX_RUNTIME_UDS_PATH", None)
            else:
                os.environ["MLX_RUNTIME_UDS_PATH"] = previous

    def test_media_type_for_path_supports_current_image_and_audio_fixtures(
        self,
    ) -> None:
        self.assertEqual(
            _media_type_for_path(Path("conditioning.ppm"), "image"),
            "image/x-portable-pixmap",
        )
        self.assertEqual(
            _media_type_for_path(Path("bark.wav"), "audio"),
            "audio/wav",
        )
        self.assertEqual(
            _media_type_for_path(Path("conditioning.mp4"), "video"),
            "video/mp4",
        )
        self.assertEqual(
            _media_type_for_path(Path("control.safetensors"), "lora"),
            "application/x-safetensors",
        )
        self.assertEqual(
            _media_type_for_path(Path("unknown.bin"), "image"),
            "application/octet-stream",
        )

    def test_generation_params_only_emits_explicit_values(self) -> None:
        args = argparse.Namespace(
            width=384,
            height=224,
            fps=24,
            seed=1234,
            num_frames=17,
            num_inference_steps=8,
            guidance_scale=1.25,
            window_start_seconds=None,
            window_end_seconds=None,
            no_regenerate_video=False,
            no_regenerate_audio=False,
        )
        self.assertEqual(
            _generation_params(args),
            {
                "width": 384,
                "height": 224,
                "fps": 24,
                "seed": 1234,
                "num_frames": 17,
                "num_inference_steps": 8,
                "guidance_scale": 1.25,
            },
        )

    def test_generation_params_include_retake_window_and_regeneration_toggles(
        self,
    ) -> None:
        args = argparse.Namespace(
            width=None,
            height=None,
            fps=None,
            seed=None,
            num_frames=None,
            num_inference_steps=None,
            guidance_scale=None,
            window_start_seconds=1.25,
            window_end_seconds=2.75,
            no_regenerate_video=True,
            no_regenerate_audio=False,
        )
        self.assertEqual(
            _generation_params(args),
            {
                "window_start_seconds": 1.25,
                "window_end_seconds": 2.75,
                "regenerate_video": False,
            },
        )

    def test_parse_extensions_json_accepts_inline_json(self) -> None:
        self.assertEqual(
            _parse_extensions_json(
                '{"ltx": {"workflow_variant": "distilled_two_stage"}}'
            ),
            {"ltx": {"workflow_variant": "distilled_two_stage"}},
        )

    def test_coerce_action_values_rejects_non_string_sequence_items(self) -> None:
        action = argparse.Action(
            option_strings=["--demo"],
            dest="demo",
            nargs=None,
        )
        with self.assertRaisesRegex(argparse.ArgumentError, "non-string argument"):
            _coerce_action_values(action, ["ok", 1], "--demo", expected="2")

    def test_references_from_args_plan_only_does_not_import_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "conditioning.png"
            audio_path = Path(tmp_dir) / "bark.wav"
            video_path = Path(tmp_dir) / "conditioning.mp4"
            lora_path = Path(tmp_dir) / "control.safetensors"
            image_path.write_bytes(b"png")
            audio_path.write_bytes(b"wav")
            video_path.write_bytes(b"mp4")
            lora_path.write_bytes(b"lora")
            client = _FakeClient()
            args = argparse.Namespace(
                image=[image_path],
                keyframe_image=[
                    type(
                        "Spec",
                        (),
                        {"path": image_path, "frame_index": 16, "strength": 0.9},
                    )()
                ],
                image_frame_index=8,
                image_strength=0.75,
                audio=audio_path,
                video=[video_path],
                lora=[type("Lora", (), {"path": lora_path, "strength": 0.6})()],
                audio_start_seconds=0.5,
                audio_max_duration_seconds=2.0,
                extensions_json=None,
                plan_only=True,
            )
            references = _references_from_args(client, args)
            self.assertEqual(client.calls, [])
            self.assertEqual(
                [
                    (
                        reference.kind,
                        reference.input_handle,
                        reference.metadata,
                    )
                    for reference in references
                ],
                [
                    (
                        "image",
                        None,
                        {"frame_index": 8, "strength": 0.75},
                    ),
                    (
                        "image",
                        None,
                        {"frame_index": 16, "strength": 0.9},
                    ),
                    (
                        "audio",
                        None,
                        {
                            "start_time_seconds": 0.5,
                            "max_duration_seconds": 2.0,
                        },
                    ),
                    ("video", None, {"strength": 1.0}),
                    ("lora", None, {"strength": 0.6}),
                ],
            )

    def test_references_from_args_binds_imported_handles_for_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "conditioning.png"
            audio_path = Path(tmp_dir) / "bark.wav"
            video_path = Path(tmp_dir) / "conditioning.mp4"
            lora_path = Path(tmp_dir) / "control.safetensors"
            image_path.write_bytes(b"png")
            audio_path.write_bytes(b"wav")
            video_path.write_bytes(b"mp4")
            lora_path.write_bytes(b"lora")
            client = _FakeClient()
            args = argparse.Namespace(
                image=[image_path],
                keyframe_image=[
                    type(
                        "Spec",
                        (),
                        {"path": image_path, "frame_index": 16, "strength": 0.9},
                    )()
                ],
                image_frame_index=8,
                image_strength=0.75,
                audio=audio_path,
                video=[video_path],
                lora=[type("Lora", (), {"path": lora_path, "strength": 0.6})()],
                audio_start_seconds=0.5,
                audio_max_duration_seconds=2.0,
                extensions_json=None,
                plan_only=False,
            )
            references = _references_from_args(client, args)
            self.assertEqual(
                client.calls,
                [
                    (image_path, "image"),
                    (image_path, "image"),
                    (audio_path, "audio"),
                    (video_path, "video"),
                    (lora_path, "lora"),
                ],
            )
            self.assertEqual(
                [
                    (
                        reference.kind,
                        reference.input_handle,
                        reference.metadata,
                    )
                    for reference in references
                ],
                [
                    (
                        "image",
                        "image-handle",
                        {"frame_index": 8, "strength": 0.75},
                    ),
                    (
                        "image",
                        "image-handle",
                        {"frame_index": 16, "strength": 0.9},
                    ),
                    (
                        "audio",
                        "audio-handle",
                        {
                            "start_time_seconds": 0.5,
                            "max_duration_seconds": 2.0,
                        },
                    ),
                    ("video", "video-handle", {"strength": 1.0}),
                    ("lora", "lora-handle", {"strength": 0.6}),
                ],
            )

    def test_wait_for_terminal_job_polls_until_completed(self) -> None:
        client = _FakeClient()
        client.jobs = [
            JobRecord(
                job_id="job_1",
                request=JobRequest(model_id="m", task="video.generate"),
                state=JobState.RUNNING,
            ),
            JobRecord(
                job_id="job_1",
                request=JobRequest(model_id="m", task="video.generate"),
                state=JobState.COMPLETED,
            ),
        ]

        record = _wait_for_terminal_job(
            client,
            "job_1",
            timeout_seconds=0.1,
            poll_interval_seconds=0.0,
        )

        self.assertEqual(record.state, JobState.COMPLETED)

    def test_run_generate_command_waits_and_exports_first_artifact(self) -> None:
        client = _FakeClient()
        client.jobs = [
            JobRecord(
                job_id="job_1",
                request=JobRequest(model_id="m", task="video.generate"),
                state=JobState.COMPLETED,
                artifacts=[
                    OutputArtifactRecord(
                        artifact_id="out_1",
                        artifact_format="mp4",
                        job_id="job_1",
                        storage_key="jobs/job_1/out_1.mp4",
                    )
                ],
            )
        ]
        args = argparse.Namespace(
            model_id="ltx-2.3-fast-local",
            prompt="golden retriever in a park",
            task=None,
            negative_prompt=None,
            image=None,
            keyframe_image=[],
            image_frame_index=0,
            image_strength=1.0,
            audio=None,
            video=None,
            lora=[],
            audio_start_seconds=0.0,
            audio_max_duration_seconds=None,
            width=None,
            height=None,
            num_frames=None,
            fps=None,
            seed=None,
            num_inference_steps=None,
            guidance_scale=None,
            window_start_seconds=None,
            window_end_seconds=None,
            no_regenerate_video=False,
            no_regenerate_audio=False,
            artifact_format="mp4",
            quality="auto",
            extensions_json=None,
            plan_only=False,
            wait=True,
            timeout_seconds=0.1,
            poll_interval_seconds=0.0,
            export_path=Path("/tmp/result.mp4"),
            overwrite_export=True,
        )

        import io
        import sys

        previous_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exit_code = _run_generate_command(client, args)
            payload = json.loads(sys.stdout.getvalue())
        finally:
            sys.stdout = previous_stdout

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["state"], "completed")
        self.assertEqual(payload["artifact_ids"], ["out_1"])
        resolved_export_path = str(Path("/tmp/result.mp4").resolve())
        self.assertEqual(payload["export_path"], resolved_export_path)
        self.assertEqual(
            client.export_calls,
            [("out_1", Path(resolved_export_path), True)],
        )

    def test_wait_for_terminal_job_times_out(self) -> None:
        client = _FakeClient()
        client.jobs = [
            JobRecord(
                job_id="job_1",
                request=JobRequest(model_id="m", task="video.generate"),
                state=JobState.RUNNING,
            )
        ]

        with self.assertRaisesRegex(TimeoutError, "Timed out waiting for job"):
            _wait_for_terminal_job(
                client,
                "job_1",
                timeout_seconds=0.0,
                poll_interval_seconds=0.0,
            )

    def test_runtime_client_uses_http_auth_header_for_loopback_mode(self) -> None:
        captured: dict[str, object] = {}

        class _FakeHttpxClient:
            def __init__(
                self,
                *,
                base_url: str,
                headers: dict[str, str],
                transport: object,
                timeout: object = None,
            ) -> None:
                captured["base_url"] = base_url
                captured["headers"] = headers
                captured["transport"] = transport
                captured["timeout"] = timeout

            def close(self) -> None:
                return None

        with patch("mlxr.clients.cli.runtime.httpx.Client", _FakeHttpxClient):
            client = RuntimeClient(
                base_url="http://127.0.0.1:8000/",
                uds_path=None,
                http_token="secret-token",
            )
            client.close()

        self.assertEqual(captured["base_url"], "http://127.0.0.1:8000")
        self.assertEqual(
            captured["headers"],
            {"Authorization": "Bearer secret-token"},
        )
        self.assertIsNone(captured["transport"])
        self.assertIsNone(captured["timeout"])

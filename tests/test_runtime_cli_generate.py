from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from mlxr.clients.cli.cli import (
    RuntimeClient,
    _default_uds_path,
    _generation_params,
    _media_type_for_path,
    _references_from_args,
    _run_generate_command,
    _wait_for_terminal_job,
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
            ]
        )
        self.assertEqual(parsed.command, "generate")
        self.assertEqual(parsed.model_id, "ltx-2.3-fast-local")
        self.assertEqual(parsed.prompt, "golden retriever in a park")
        self.assertEqual(parsed.width, 96)
        self.assertEqual(parsed.num_frames, 9)

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
                    "--plan-only",
                ]
            )
            self.assertEqual(parsed.image, image_path)
            self.assertTrue(parsed.plan_only)

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
        )
        self.assertEqual(
            _generation_params(args),
            {
                "width": 384,
                "height": 224,
                "fps": 24,
                "seed": 1234,
                "num_frames": 17,
            },
        )

    def test_references_from_args_plan_only_does_not_import_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "conditioning.png"
            audio_path = Path(tmp_dir) / "bark.wav"
            image_path.write_bytes(b"png")
            audio_path.write_bytes(b"wav")
            client = _FakeClient()
            args = argparse.Namespace(
                image=image_path,
                audio=audio_path,
                plan_only=True,
            )
            references = _references_from_args(client, args)
            self.assertEqual(client.calls, [])
            self.assertEqual(
                [(reference.kind, reference.input_handle) for reference in references],
                [("image", None), ("audio", None)],
            )

    def test_references_from_args_binds_imported_handles_for_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "conditioning.png"
            audio_path = Path(tmp_dir) / "bark.wav"
            image_path.write_bytes(b"png")
            audio_path.write_bytes(b"wav")
            client = _FakeClient()
            args = argparse.Namespace(
                image=image_path,
                audio=audio_path,
                plan_only=False,
            )
            references = _references_from_args(client, args)
            self.assertEqual(
                client.calls,
                [(image_path, "image"), (audio_path, "audio")],
            )
            self.assertEqual(
                [(reference.kind, reference.input_handle) for reference in references],
                [("image", "image-handle"), ("audio", "audio-handle")],
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
            video_prompt=None,
            audio_prompt=None,
            image=None,
            audio=None,
            width=None,
            height=None,
            num_frames=None,
            fps=None,
            seed=None,
            artifact_format="mp4",
            natural_audio=False,
            no_music=False,
            enhance_prompt=False,
            quality="auto",
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

from __future__ import annotations

import unittest

from mlxr.core.runtime import TraceRecorder


class TraceRecorderTests(unittest.TestCase):
    def test_span_records_duration_and_attributes(self) -> None:
        recorder = TraceRecorder(enabled=True)

        with recorder.span(
            "decode_video",
            description="Decode final video frames",
            attributes={"stage": "decode"},
        ) as span:
            span.set_attribute("tiling_mode", "auto")

        metadata = recorder.to_metadata()
        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["event_count"], 1)
        event = metadata["events"][0]
        self.assertEqual(event["name"], "decode_video")
        self.assertEqual(event["description"], "Decode final video frames")
        self.assertEqual(event["attributes"]["stage"], "decode")
        self.assertEqual(event["attributes"]["tiling_mode"], "auto")
        self.assertGreaterEqual(event["duration_ms"], 0.0)

    def test_span_calls_sync_when_requested(self) -> None:
        recorder = TraceRecorder(enabled=True)
        calls: list[str] = []

        with recorder.span("denoise_step", sync=lambda: calls.append("sync")):
            pass

        self.assertEqual(calls, ["sync"])
        event = recorder.to_metadata()["events"][0]
        self.assertTrue(event["synchronized"])

    def test_disabled_recorder_yields_span_without_recording(self) -> None:
        recorder = TraceRecorder(enabled=False)

        with recorder.span("ignored") as span:
            span.set_attribute("key", "value")

        metadata = recorder.to_metadata()
        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["event_count"], 0)

    def test_span_preserves_body_exception_when_sync_also_fails(self) -> None:
        recorder = TraceRecorder(enabled=True)

        def failing_sync() -> None:
            raise ValueError("sync failed")

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with recorder.span("denoise_step", sync=failing_sync):
                raise RuntimeError("body failed")

        event = recorder.to_metadata()["events"][0]
        self.assertTrue(event["attributes"]["failed"])
        self.assertEqual(event["attributes"]["error_type"], "RuntimeError")
        self.assertEqual(event["attributes"]["sync_error_type"], "ValueError")

    def test_metadata_includes_summary(self) -> None:
        recorder = TraceRecorder(enabled=True)

        with recorder.span("decode"):
            pass
        with recorder.span("decode"):
            pass

        summary = recorder.to_metadata()["summary"]["decode"]
        self.assertEqual(summary["count"], 2)
        self.assertGreaterEqual(
            summary["total_duration_ms"], summary["max_duration_ms"]
        )

    def test_span_records_snapshot_before_and_after(self) -> None:
        recorder = TraceRecorder(enabled=True)
        calls = iter(({"active_bytes": 1}, {"active_bytes": 2}))

        with recorder.span("generate", snapshot=lambda: next(calls)):
            pass

        event = recorder.to_metadata()["events"][0]
        self.assertEqual(event["attributes"]["snapshot_before"]["active_bytes"], 1)
        self.assertEqual(event["attributes"]["snapshot_after"]["active_bytes"], 2)


if __name__ == "__main__":
    unittest.main()

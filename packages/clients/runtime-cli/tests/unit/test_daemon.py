from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from mlxr.clients.cli.daemon import (
    daemon_health,
    daemon_metadata_path,
    daemon_status,
    ensure_runtime_daemon,
    read_daemon_metadata,
    stop_runtime_daemon,
)
from mlxr.core.runtime import RuntimeHome


class RuntimeDaemonLifecycleTests(unittest.TestCase):
    def test_ensure_runtime_daemon_starts_reuses_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            socket_path = Path("/tmp") / f"mlxr-{uuid.uuid4().hex[:12]}.sock"
            with patch.dict(
                os.environ,
                {"MLX_RUNTIME_JOB_EXECUTION_MODE": "thread"},
                clear=False,
            ):
                try:
                    started = ensure_runtime_daemon(
                        runtime_home=runtime_home,
                        socket_path=socket_path,
                        startup_timeout_seconds=20.0,
                    )
                    self.assertEqual(started.status, "started")
                    self.assertTrue(started.healthy)
                    self.assertTrue(Path(started.stdio_log_path).exists())
                    metadata = read_daemon_metadata(runtime_home)
                    self.assertIsNotNone(metadata)
                    running = daemon_status(
                        runtime_home=runtime_home,
                        socket_path=socket_path,
                    )
                    self.assertEqual(running.status, "running")
                    self.assertTrue(daemon_health(socket_path=socket_path))
                    reused = ensure_runtime_daemon(
                        runtime_home=runtime_home,
                        socket_path=socket_path,
                        startup_timeout_seconds=20.0,
                    )
                    self.assertEqual(reused.status, "running")
                    self.assertTrue(reused.healthy)
                finally:
                    stopped = stop_runtime_daemon(
                        runtime_home=runtime_home,
                        socket_path=socket_path,
                    )
            self.assertEqual(stopped.status, "stopped")
            self.assertFalse(socket_path.exists())
            self.assertFalse(daemon_metadata_path(runtime_home).exists())

    def test_ensure_runtime_daemon_rejects_overlong_socket_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            socket_path = Path("/tmp") / ("x" * 120)
            with self.assertRaisesRegex(RuntimeError, "socket path is too long"):
                ensure_runtime_daemon(
                    runtime_home=runtime_home,
                    socket_path=socket_path,
                    startup_timeout_seconds=1.0,
                )

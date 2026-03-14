from __future__ import annotations

import queue
import unittest

from mlxr.core.server.jobs import SpawnManagedQueue, ThreadMessageQueue


class _FakeSpawnQueue:
    def __init__(self) -> None:
        self._items: list[object] = []

    def put(self, item: object) -> None:
        self._items.append(item)

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        del block, timeout
        return self._items.pop(0)

    def get_nowait(self) -> object:
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)

    def close(self) -> None:
        return None


class ManagedQueueTests(unittest.TestCase):
    def test_spawn_managed_queue_get_nowait_returns_worker_message(self) -> None:
        queue_wrapper = SpawnManagedQueue(_FakeSpawnQueue())
        queue_wrapper.put({"command": "cancel"})

        self.assertEqual(queue_wrapper.get_nowait(), {"command": "cancel"})

    def test_thread_message_queue_get_nowait_returns_worker_message(self) -> None:
        queue_wrapper = ThreadMessageQueue()
        queue_wrapper.put({"command": "cancel"})

        self.assertEqual(queue_wrapper.get_nowait(), {"command": "cancel"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

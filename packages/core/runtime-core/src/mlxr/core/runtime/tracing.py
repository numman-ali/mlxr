from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping, TypeAlias

import mlx.core as mx

TraceScalar: TypeAlias = str | int | float | bool | None
TraceValue: TypeAlias = TraceScalar | list["TraceValue"] | dict[str, "TraceValue"]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    offset_ms: float
    duration_ms: float
    synchronized: bool
    description: str | None = None
    attributes: dict[str, TraceValue] = field(default_factory=dict)


class TraceSpan:
    __slots__ = ("_attributes",)

    def __init__(
        self, initial_attributes: Mapping[str, TraceValue] | None = None
    ) -> None:
        self._attributes: dict[str, TraceValue] = (
            dict(initial_attributes) if initial_attributes is not None else {}
        )

    @property
    def attributes(self) -> dict[str, TraceValue]:
        return self._attributes

    def set_attribute(self, key: str, value: TraceValue) -> None:
        self._attributes[key] = value

    def add_attributes(self, values: Mapping[str, TraceValue]) -> None:
        self._attributes.update(values)


class TraceRecorder:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._started_at = time.perf_counter()
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        description: str | None = None,
        attributes: Mapping[str, TraceValue] | None = None,
        sync: Callable[[], None] | None = None,
        snapshot: Callable[[], Mapping[str, TraceValue]] | None = None,
    ) -> Iterator[TraceSpan]:
        span = TraceSpan(attributes)
        if not self.enabled:
            yield span
            return

        started_at = time.perf_counter()
        offset_ms = (started_at - self._started_at) * 1000.0
        body_error: BaseException | None = None
        if snapshot is not None:
            span.set_attribute("snapshot_before", dict(snapshot()))
        try:
            yield span
        except BaseException as exc:
            body_error = exc
            span.set_attribute("failed", True)
            span.set_attribute("error_type", type(exc).__name__)
            span.set_attribute("error_message", str(exc))
            raise
        finally:
            if sync is not None:
                try:
                    sync()
                except Exception as exc:
                    if body_error is None:
                        raise
                    span.set_attribute("sync_error_type", type(exc).__name__)
                    span.set_attribute("sync_error_message", str(exc))
            if snapshot is not None:
                span.set_attribute("snapshot_after", dict(snapshot()))
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            self._events.append(
                TraceEvent(
                    name=name,
                    description=description,
                    offset_ms=round(offset_ms, 3),
                    duration_ms=round(duration_ms, 3),
                    synchronized=sync is not None,
                    attributes=dict(span.attributes),
                )
            )

    def to_metadata(self) -> dict[str, object]:
        summaries: dict[str, dict[str, float | int]] = {}
        grouped: dict[str, list[float]] = defaultdict(list)
        for event in self._events:
            grouped[event.name].append(event.duration_ms)
        for name, durations in grouped.items():
            summaries[name] = {
                "count": len(durations),
                "total_duration_ms": round(sum(durations), 3),
                "max_duration_ms": round(max(durations), 3),
                "mean_duration_ms": round(sum(durations) / len(durations), 3),
            }
        return {
            "enabled": self.enabled,
            "elapsed_ms": round((time.perf_counter() - self._started_at) * 1000.0, 3),
            "event_count": len(self._events),
            "events": [
                {
                    "name": event.name,
                    "description": event.description,
                    "offset_ms": event.offset_ms,
                    "duration_ms": event.duration_ms,
                    "synchronized": event.synchronized,
                    "attributes": event.attributes,
                }
                for event in self._events
            ],
            "summary": summaries,
        }


def mlx_memory_snapshot() -> dict[str, TraceValue]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
    }

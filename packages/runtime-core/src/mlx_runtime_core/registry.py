from __future__ import annotations

from collections.abc import Iterable

from .contracts import FamilyAdapter


class RuntimeRegistry:
    def __init__(self) -> None:
        self._families: dict[str, FamilyAdapter] = {}

    def register(self, adapter: FamilyAdapter) -> None:
        self._families[adapter.family_id] = adapter

    def get(self, family_id: str) -> FamilyAdapter:
        return self._families[family_id]

    def families(self) -> Iterable[str]:
        return self._families.keys()

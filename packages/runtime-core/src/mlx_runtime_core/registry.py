from __future__ import annotations

from collections.abc import Iterable

from .contracts import ModelFamilyAdapter, SourceProviderAdapter


class RuntimeRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, SourceProviderAdapter] = {}
        self._families: dict[str, ModelFamilyAdapter] = {}

    def register_provider(self, adapter: SourceProviderAdapter) -> None:
        self._providers[adapter.provider_id] = adapter

    def register_family(self, adapter: ModelFamilyAdapter) -> None:
        self._families[adapter.family_id] = adapter

    def get_provider(self, provider_id: str) -> SourceProviderAdapter:
        return self._providers[provider_id]

    def get_family(self, family_id: str) -> ModelFamilyAdapter:
        return self._families[family_id]

    def has_provider(self, provider_id: str) -> bool:
        return provider_id in self._providers

    def has_family(self, family_id: str) -> bool:
        return family_id in self._families

    def providers(self) -> Iterable[str]:
        return self._providers.keys()

    def families(self) -> Iterable[str]:
        return self._families.keys()

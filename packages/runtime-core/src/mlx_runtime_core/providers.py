from __future__ import annotations

import hashlib
from pathlib import Path

from mlx_runtime_schemas import (
    AuthRequirements,
    ProvenanceRecord,
    ResolvedSource,
    SourceFileRecord,
    SourceRef,
)

from .contracts import FetchPolicy, ProviderInspection, SourceMaterialization


class LocalFileProviderAdapter:
    provider_id = "local"

    def resolve(self, source_ref: SourceRef) -> ResolvedSource:
        path_value = source_ref.locator.get("path")
        if not path_value:
            raise ValueError("Local provider requires locator.path")

        path = Path(path_value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local source '{path}' does not exist")

        files = self._collect_files(path)
        pinned_ref = self._structural_digest(path, files)
        return ResolvedSource(
            provider=self.provider_id,
            locator={"path": str(path)},
            pinned_ref=f"structural-sha256:{pinned_ref}",
            access_state="local-only",
            license=source_ref.locator.get("license"),
            remote_code_required=False,
            auth_requirements=self.auth_requirements(source_ref),
            files=files,
            metadata={
                "source_kind": "directory" if path.is_dir() else "file",
                "digest_basis": "relative_paths_and_sizes",
                "remote_code_approved": source_ref.policy.allow_remote_code,
                "family_hint": source_ref.family_hint,
            },
        )

    def inspect(self, resolved: ResolvedSource) -> ProviderInspection:
        bytes_total = sum(file.size_bytes or 0 for file in resolved.files) or None
        return ProviderInspection(
            resolved=resolved,
            bytes_total=bytes_total,
            metadata={"file_count": len(resolved.files)},
        )

    def auth_requirements(self, source_ref: SourceRef) -> AuthRequirements:
        return AuthRequirements(
            required=False,
            supported=[],
            message="Trusted local bundles require no provider auth",
        )

    def fetch(
        self, resolved: ResolvedSource, policy: FetchPolicy
    ) -> SourceMaterialization:
        path = Path(resolved.locator["path"])
        return SourceMaterialization(
            resolved=resolved,
            provenance=self.provenance(resolved),
            materialization_mode="local-bundle",
            local_path=path,
            local_refs=(str(path),),
            metadata={
                "fetch_policy": policy.options,
                "materialized_from": "local-filesystem",
            },
        )

    def provenance(self, resolved: ResolvedSource) -> ProvenanceRecord:
        return ProvenanceRecord(
            provider=resolved.provider,
            locator=resolved.locator,
            resolved_ref=resolved.pinned_ref,
            license=resolved.license,
            access_state=resolved.access_state,
            remote_code_required=resolved.remote_code_required,
            remote_code_approved=bool(
                resolved.metadata.get("remote_code_approved", False)
            ),
            metadata={"digest_basis": resolved.metadata.get("digest_basis")},
        )

    def _collect_files(self, path: Path) -> list[SourceFileRecord]:
        if path.is_file():
            stat = path.stat()
            return [SourceFileRecord(path=path.name, size_bytes=stat.st_size)]

        records: list[SourceFileRecord] = []
        for file_path in sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        ):
            stat = file_path.stat()
            records.append(
                SourceFileRecord(
                    path=str(file_path.relative_to(path)), size_bytes=stat.st_size
                )
            )
        return records

    def _structural_digest(self, path: Path, files: list[SourceFileRecord]) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(path).encode("utf-8"))
        for file in files:
            hasher.update(file.path.encode("utf-8"))
            hasher.update(str(file.size_bytes or 0).encode("utf-8"))
        return hasher.hexdigest()

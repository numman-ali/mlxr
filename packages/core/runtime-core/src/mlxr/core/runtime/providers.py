from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from mlxr.core.schemas import (
    AuthRequirements,
    ProvenanceRecord,
    ResolvedSource,
    SourceFileRecord,
    SourceRef,
)

from .contracts import FetchPolicy, ProviderInspection, SourceMaterialization


def _license_from_card_data(card_data: Any) -> str | None:
    if not isinstance(card_data, dict):
        return None
    license_value = card_data.get("license")
    return license_value if isinstance(license_value, str) else None


def _remote_code_metadata(
    files: list[SourceFileRecord], card_data: Any
) -> tuple[bool, dict[str, Any]]:
    auto_map_present = isinstance(card_data, dict) and "auto_map" in card_data
    remote_code_files = [
        file.path
        for file in files
        if file.path.endswith(".py")
        and (
            file.path.startswith("modeling_")
            or file.path.startswith("configuration_")
            or file.path.startswith("processing_")
            or file.path.startswith("tokenization_")
        )
    ]
    remote_code_required = auto_map_present or bool(remote_code_files)
    return (
        remote_code_required,
        {
            "remote_code_detection": "best_effort",
            "remote_code_detection_confidence": "low",
            "remote_code_signal": {
                "auto_map_present": auto_map_present,
                "python_entrypoints": remote_code_files,
            },
        },
    )


class ModelInfoClient(Protocol):
    def model_info(self, **kwargs: object) -> Any: ...


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


class HuggingFaceProviderAdapter:
    provider_id = "huggingface"
    _default_auth_message = (
        "Use `hf auth login` on this machine or set `HF_TOKEN`; "
        "`auth.token_ref='hf-default'` resolves the default Hugging Face login."
    )

    def __init__(self, api: ModelInfoClient | None = None) -> None:
        self._api = api or HfApi()

    def resolve(self, source_ref: SourceRef) -> ResolvedSource:
        repo_id = source_ref.locator.get("repo")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("Hugging Face provider requires locator.repo")

        revision = source_ref.locator.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise ValueError("Hugging Face locator.revision must be a string")

        token = self._token_for_ref(source_ref)
        try:
            model_info = self._api.model_info(
                repo_id=repo_id,
                revision=revision,
                files_metadata=True,
                token=token,
            )
        except RepositoryNotFoundError as exc:
            raise FileNotFoundError(
                f"Hugging Face repo '{repo_id}' was not found"
            ) from exc
        except RevisionNotFoundError as exc:
            raise FileNotFoundError(
                f"Hugging Face revision '{revision}' was not found for '{repo_id}'"
            ) from exc
        except GatedRepoError as exc:
            raise PermissionError(
                f"Hugging Face repo '{repo_id}' requires authenticated access"
            ) from exc
        except HfHubHTTPError as exc:
            raise ValueError(
                f"Hugging Face request failed for '{repo_id}': {exc}"
            ) from exc
        files = self._files_from_info(model_info)
        remote_code_required, remote_code_metadata = _remote_code_metadata(
            files, getattr(model_info, "cardData", None)
        )
        access_state = self._access_state(model_info)

        return ResolvedSource(
            provider=self.provider_id,
            locator={"repo": repo_id, "revision": revision},
            pinned_ref=getattr(model_info, "sha", None) or revision,
            access_state=access_state,
            license=_license_from_card_data(getattr(model_info, "cardData", None)),
            remote_code_required=remote_code_required,
            auth_requirements=AuthRequirements(
                required=access_state in {"gated", "private"},
                supported=["hf-default"],
                message=self._default_auth_message,
            ),
            files=files,
            metadata={
                "requested_revision": revision,
                "repo_id": repo_id,
                "pipeline_tag": getattr(model_info, "pipeline_tag", None),
                "private": bool(getattr(model_info, "private", False)),
                "gated": bool(getattr(model_info, "gated", False)),
                "auth_token_ref": source_ref.auth.token_ref,
                "remote_code_approved": source_ref.policy.allow_remote_code,
                **remote_code_metadata,
            },
        )

    def inspect(self, resolved: ResolvedSource) -> ProviderInspection:
        bytes_total = sum(file.size_bytes or 0 for file in resolved.files) or None
        return ProviderInspection(
            resolved=resolved,
            bytes_total=bytes_total,
            metadata={
                "file_count": len(resolved.files),
                "revision": resolved.pinned_ref,
                "repo_id": resolved.locator.get("repo"),
            },
        )

    def auth_requirements(self, source_ref: SourceRef) -> AuthRequirements:
        token_ref = source_ref.auth.token_ref
        if token_ref not in (None, "hf-default"):
            raise ValueError(
                "Hugging Face provider only supports auth.token_ref='hf-default'"
            )
        return AuthRequirements(
            required=token_ref is not None,
            supported=["hf-default"],
            message=self._default_auth_message,
        )

    def fetch(
        self, resolved: ResolvedSource, policy: FetchPolicy
    ) -> SourceMaterialization:
        repo_id = resolved.locator.get("repo")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("Resolved Hugging Face source is missing locator.repo")

        requested_revision = resolved.locator.get("revision")
        pinned_ref = resolved.pinned_ref or requested_revision
        token = self._token_from_token_ref(
            resolved.metadata.get("auth_token_ref")
            if isinstance(resolved.metadata.get("auth_token_ref"), str)
            else None
        )
        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                revision=pinned_ref,
                token=token,
                allow_patterns=list(policy.allow_patterns) or None,
            )
        except RepositoryNotFoundError as exc:
            raise FileNotFoundError(
                f"Hugging Face repo '{repo_id}' was not found"
            ) from exc
        except RevisionNotFoundError as exc:
            raise FileNotFoundError(
                f"Hugging Face revision '{pinned_ref}' was not found for '{repo_id}'"
            ) from exc
        except GatedRepoError as exc:
            raise PermissionError(
                f"Hugging Face repo '{repo_id}' requires authenticated access"
            ) from exc
        except HfHubHTTPError as exc:
            raise ValueError(
                f"Hugging Face fetch failed for '{repo_id}': {exc}"
            ) from exc
        local_path = Path(snapshot_path)
        return SourceMaterialization(
            resolved=resolved,
            provenance=self.provenance(resolved),
            materialization_mode="provider-cache-ref",
            local_path=local_path,
            local_refs=(str(local_path),),
            metadata={
                "fetch_policy": policy.options,
                "allow_patterns": list(policy.allow_patterns),
                "snapshot_path": str(local_path),
                "materialized_from": "huggingface-provider-cache",
            },
        )

    def provenance(self, resolved: ResolvedSource) -> ProvenanceRecord:
        metadata: dict[str, Any] = {
            "repo_id": resolved.locator.get("repo"),
            "requested_revision": resolved.locator.get("revision"),
            "remote_code_detection": resolved.metadata.get("remote_code_detection"),
            "remote_code_detection_confidence": resolved.metadata.get(
                "remote_code_detection_confidence"
            ),
            "remote_code_signal": resolved.metadata.get("remote_code_signal"),
        }
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
            metadata=metadata,
        )

    def _access_state(self, model_info: Any) -> str:
        if bool(getattr(model_info, "private", False)):
            return "private"
        if bool(getattr(model_info, "gated", False)):
            return "gated"
        return "public"

    def _files_from_info(self, model_info: Any) -> list[SourceFileRecord]:
        files: list[SourceFileRecord] = []
        for sibling in getattr(model_info, "siblings", []) or []:
            path = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
            if not isinstance(path, str) or not path:
                continue
            size_bytes = getattr(sibling, "size", None)
            files.append(SourceFileRecord(path=path, size_bytes=size_bytes))
        return files

    def _token_for_ref(self, source_ref: SourceRef) -> str | None:
        return self._token_from_token_ref(source_ref.auth.token_ref)

    def _token_from_token_ref(self, token_ref: str | None) -> str | None:
        if token_ref is None:
            return None
        if token_ref != "hf-default":
            raise ValueError(
                "Hugging Face provider only supports auth.token_ref='hf-default'"
            )
        token = get_token()
        if not token:
            raise ValueError(
                "No default Hugging Face login is available. "
                "Run `hf auth login` on this machine or set HF_TOKEN."
            )
        return token

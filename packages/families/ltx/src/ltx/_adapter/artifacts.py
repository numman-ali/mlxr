# mypy: ignore-errors
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mlx_runtime_core import ArtifactPayloadItem, ConversionPlan, ConversionSource
from mlx_runtime_schemas import (
    PolicyDescriptor,
    PortableArtifactComponentRecord,
    ResolvedSource,
)

from .state import PreparedComponent


def _role_candidates(self, source: ResolvedSource) -> list[str]:
    file_paths = {record.path for record in source.files}
    candidates: list[str] = []
    if self._checkpoint_filename in file_paths:
        candidates.append("checkpoint")
    if self._spatial_upsampler_filename in file_paths:
        candidates.append("spatial_upsampler")
    if self._looks_like_text_encoder_snapshot(file_paths):
        candidates.append("text_encoder")
    if (
        self._checkpoint_filename in file_paths
        and self._spatial_upsampler_filename in file_paths
        and any(
            path.startswith(f"{self._text_encoder_dirname}/")
            or path == self._text_encoder_dirname
            for path in file_paths
        )
    ):
        candidates.append("bundle")
    return candidates


def _prepare_components(
    self, sources: dict[str, ConversionSource]
) -> dict[str, PreparedComponent]:
    if set(sources) == {"bundle"}:
        return self._prepare_bundle_components(sources["bundle"])

    unexpected_roles = sorted(
        role for role in sources if role not in self._required_roles
    )
    if unexpected_roles:
        raise ValueError(
            f"LTX conversion does not support source roles: {', '.join(unexpected_roles)}"
        )
    missing_roles = [role for role in self._required_roles if role not in sources]
    if missing_roles:
        raise ValueError(
            f"LTX conversion requires source roles: {', '.join(missing_roles)}"
        )

    checkpoint_source = sources["checkpoint"]
    upsampler_source = sources["spatial_upsampler"]
    text_encoder_source = sources["text_encoder"]
    return {
        "checkpoint": PreparedComponent(
            role="checkpoint",
            kind="file",
            source_id=checkpoint_source.source_id,
            source_path=self._resolve_required_file(
                checkpoint_source.materialization.local_path,
                self._checkpoint_filename,
                role="checkpoint",
            ),
            provenance=checkpoint_source.materialization.provenance,
            resolved_ref=checkpoint_source.materialization.provenance.resolved_ref,
        ),
        "spatial_upsampler": PreparedComponent(
            role="spatial_upsampler",
            kind="file",
            source_id=upsampler_source.source_id,
            source_path=self._resolve_required_file(
                upsampler_source.materialization.local_path,
                self._spatial_upsampler_filename,
                role="spatial_upsampler",
            ),
            provenance=upsampler_source.materialization.provenance,
            resolved_ref=upsampler_source.materialization.provenance.resolved_ref,
        ),
        "text_encoder": PreparedComponent(
            role="text_encoder",
            kind="directory",
            source_id=text_encoder_source.source_id,
            source_path=self._resolve_text_encoder_dir(
                text_encoder_source.materialization.local_path,
                role="text_encoder",
            ),
            provenance=text_encoder_source.materialization.provenance,
            resolved_ref=text_encoder_source.materialization.provenance.resolved_ref,
        ),
    }


def _prepare_bundle_components(
    self, bundle_source: ConversionSource
) -> dict[str, PreparedComponent]:
    bundle_root = bundle_source.materialization.local_path
    if bundle_root is None or not bundle_root.is_dir():
        raise ValueError(
            "LTX bundle conversion requires a directory source containing the fast-path assets"
        )
    text_encoder_root = bundle_root / self._text_encoder_dirname
    if not text_encoder_root.exists():
        raise ValueError(f"LTX bundle source is missing '{self._text_encoder_dirname}'")
    self._validate_text_encoder_dir(text_encoder_root, role="bundle")
    return {
        "checkpoint": PreparedComponent(
            role="checkpoint",
            kind="file",
            source_id=bundle_source.source_id,
            source_path=self._resolve_required_file(
                bundle_root, self._checkpoint_filename, role="bundle"
            ),
            provenance=bundle_source.materialization.provenance,
            resolved_ref=bundle_source.materialization.provenance.resolved_ref,
        ),
        "spatial_upsampler": PreparedComponent(
            role="spatial_upsampler",
            kind="file",
            source_id=bundle_source.source_id,
            source_path=self._resolve_required_file(
                bundle_root, self._spatial_upsampler_filename, role="bundle"
            ),
            provenance=bundle_source.materialization.provenance,
            resolved_ref=bundle_source.materialization.provenance.resolved_ref,
        ),
        "text_encoder": PreparedComponent(
            role="text_encoder",
            kind="directory",
            source_id=bundle_source.source_id,
            source_path=text_encoder_root,
            provenance=bundle_source.materialization.provenance,
            resolved_ref=bundle_source.materialization.provenance.resolved_ref,
        ),
    }


def _artifact_components(
    self, prepared: dict[str, PreparedComponent]
) -> tuple[list[PortableArtifactComponentRecord], list[ArtifactPayloadItem]]:
    components: list[PortableArtifactComponentRecord] = []
    payload_items: list[ArtifactPayloadItem] = []
    for role in self._required_roles:
        component = prepared[role]
        if component.kind == "file":
            relative_path = Path("payload") / role / component.source_path.name
            payload_items.append(
                ArtifactPayloadItem(
                    source_path=component.source_path,
                    relative_path=relative_path,
                )
            )
            components.append(
                PortableArtifactComponentRecord(
                    role=role,
                    kind="file",
                    relative_path=relative_path.as_posix(),
                    source_id=component.source_id,
                    resolved_ref=component.resolved_ref,
                    size_bytes=component.source_path.stat().st_size,
                    component_digest=self._file_digest(component.source_path),
                    provenance=component.provenance,
                    metadata={"filename": component.source_path.name},
                )
            )
            continue

        relative_root = Path("payload") / role
        for file_path in self._directory_files(component.source_path):
            payload_items.append(
                ArtifactPayloadItem(
                    source_path=file_path,
                    relative_path=relative_root
                    / file_path.relative_to(component.source_path),
                )
            )
        components.append(
            PortableArtifactComponentRecord(
                role=role,
                kind="directory",
                relative_path=relative_root.as_posix(),
                source_id=component.source_id,
                resolved_ref=component.resolved_ref,
                size_bytes=self._directory_size(component.source_path),
                component_digest=self._directory_digest(component.source_path),
                provenance=component.provenance,
                metadata={"dirname": component.source_path.name},
            )
        )
    return components, payload_items


def _artifact_digest(
    self,
    components: list[PortableArtifactComponentRecord],
    plan: ConversionPlan,
) -> str:
    payload = {
        "family": self.family_id,
        "model_id": plan.model_id,
        "precision": plan.precision,
        "target_format": plan.target_format,
        "components": [
            {
                "role": component.role,
                "kind": component.kind,
                "relative_path": component.relative_path,
                "component_digest": component.component_digest,
            }
            for component in components
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _combined_policy(self, prepared: dict[str, PreparedComponent]) -> PolicyDescriptor:
    provenances = [component.provenance for component in prepared.values()]
    remote_code_required = any(
        provenance.remote_code_required for provenance in provenances
    )
    remote_code_approved = remote_code_required and all(
        (not provenance.remote_code_required or provenance.remote_code_approved)
        for provenance in provenances
    )
    return PolicyDescriptor(
        license=prepared["checkpoint"].provenance.license,
        access_state=prepared["checkpoint"].provenance.access_state,
        remote_code_required=remote_code_required,
        remote_code_approved=remote_code_approved,
    )


def _component_paths(
    self,
    artifact_root: Path,
    components: list[PortableArtifactComponentRecord],
) -> dict[str, Path]:
    component_paths: dict[str, Path] = {}
    for component in components:
        component_path = artifact_root / component.relative_path
        if component.kind == "file":
            if not component_path.is_file():
                raise ValueError(
                    f"LTX artifact is missing required file component '{component.role}'"
                )
        elif component.kind == "directory":
            if not component_path.is_dir():
                raise ValueError(
                    f"LTX artifact is missing required directory component '{component.role}'"
                )
        component_paths[component.role] = component_path
    missing_roles = [
        role for role in self._required_roles if role not in component_paths
    ]
    if missing_roles:
        raise ValueError(
            f"LTX artifact is missing required component roles: {', '.join(missing_roles)}"
        )
    self._validate_text_encoder_dir(component_paths["text_encoder"], role="artifact")
    return component_paths


def _resolve_required_file(
    self, local_path: Path | None, filename: str, *, role: str
) -> Path:
    if local_path is None:
        raise ValueError(f"LTX {role} source is missing a local materialization path")
    if local_path.is_file():
        if local_path.name != filename:
            raise ValueError(
                f"LTX {role} source must point to '{filename}', got '{local_path.name}'"
            )
        return local_path
    candidate = local_path / filename
    if candidate.is_file():
        return candidate
    raise ValueError(f"LTX {role} source is missing '{filename}'")


def _resolve_text_encoder_dir(self, local_path: Path | None, *, role: str) -> Path:
    if local_path is None:
        raise ValueError(f"LTX {role} source is missing a local text-encoder directory")
    if local_path.is_file():
        raise ValueError(f"LTX {role} source must be a directory")

    candidates = [local_path]
    named_child = local_path / self._text_encoder_dirname
    if named_child.is_dir():
        candidates.insert(0, named_child)

    for candidate in candidates:
        try:
            self._validate_text_encoder_dir(candidate, role=role)
        except ValueError:
            continue
        return candidate
    raise ValueError(
        f"LTX {role} source is missing a valid Gemma text-encoder directory"
    )


def _validate_text_encoder_dir(self, text_encoder_root: Path, *, role: str) -> None:
    if not text_encoder_root.is_dir():
        raise ValueError(f"LTX {role} text encoder source must be a directory")
    config_path = text_encoder_root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"LTX {role} text encoder source is missing 'config.json'")
    tokenizer_candidates = (
        text_encoder_root / "tokenizer.json",
        text_encoder_root / "tokenizer.model",
        text_encoder_root / "tokenizer_config.json",
    )
    if not any(candidate.is_file() for candidate in tokenizer_candidates):
        raise ValueError("LTX text encoder source must include tokenizer metadata")
    if not any(
        file_path.suffix == ".safetensors"
        for file_path in self._directory_files(text_encoder_root)
    ):
        raise ValueError(
            "LTX text encoder source must include at least one safetensors weight file"
        )


def _looks_like_text_encoder_snapshot(self, file_paths: set[str]) -> bool:
    has_config = "config.json" in file_paths
    has_tokenizer = any(
        name in file_paths
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    has_weights = any(path.endswith(".safetensors") for path in file_paths)
    return has_config and has_tokenizer and has_weights


def _directory_files(self, root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def _directory_size(self, root: Path) -> int:
    return sum(path.stat().st_size for path in self._directory_files(root))


def _file_digest(self, source_path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(source_path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def _directory_digest(self, root: Path) -> str:
    hasher = hashlib.sha256()
    for file_path in self._directory_files(root):
        relative_path = file_path.relative_to(root).as_posix()
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(file_path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def _media_type_for_format(self, artifact_format: str) -> str:
    if artifact_format == "mp4":
        return "video/mp4"
    if artifact_format == "mov":
        return "video/quicktime"
    if artifact_format == "wav":
        return "audio/wav"
    return "application/octet-stream"


def _require_str(self, value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"LTX artifact metadata '{name}' must be a non-empty string")
    return value

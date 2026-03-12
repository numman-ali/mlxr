from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from mlxr.core.runtime import ArtifactPayloadItem, ConversionPlan, ConversionSource
from mlxr.core.schemas import (
    PolicyDescriptor,
    PortableArtifactComponentRecord,
    ResolvedSource,
)

from .state import PreparedComponent

if TYPE_CHECKING:
    from ..adapter import LTXFamilyAdapter


def _role_candidates(self: LTXFamilyAdapter, source: ResolvedSource) -> list[str]:
    file_paths = {record.path for record in source.files}
    candidates: list[str] = []
    checkpoint_variant = _checkpoint_variant_for_file_paths(self, file_paths)
    if checkpoint_variant is not None:
        candidates.append("checkpoint")
    if self._spatial_upsampler_filename in file_paths:
        candidates.append("spatial_upsampler")
    if self._distilled_lora_filename in file_paths:
        candidates.append("distilled_lora")
    if self._looks_like_text_encoder_snapshot(file_paths):
        candidates.append("text_encoder")
    text_encoder_ready = any(
        path.startswith(f"{self._text_encoder_dirname}/")
        or path == self._text_encoder_dirname
        for path in file_paths
    )
    bundle_ready = (
        checkpoint_variant == "fast"
        and self._spatial_upsampler_filename in file_paths
        and text_encoder_ready
    ) or (checkpoint_variant == "dev" and text_encoder_ready)
    if bundle_ready:
        candidates.append("bundle")
    return candidates


def _prepare_components(
    self: LTXFamilyAdapter, sources: dict[str, ConversionSource]
) -> dict[str, PreparedComponent]:
    if set(sources) == {"bundle"}:
        return self._prepare_bundle_components(sources["bundle"])

    unexpected_roles = sorted(
        role
        for role in sources
        if role not in (*self._required_roles, *self._optional_roles)
    )
    if unexpected_roles:
        raise ValueError(
            f"LTX conversion does not support source roles: {', '.join(unexpected_roles)}"
        )
    checkpoint_source = sources.get("checkpoint")
    if checkpoint_source is None:
        raise ValueError("LTX conversion requires a checkpoint source role")
    checkpoint_path = self._resolve_checkpoint_file(
        checkpoint_source.materialization.local_path,
        role="checkpoint",
    )
    required_roles = _required_roles_for_variant(
        self, _checkpoint_variant_for_path(self, checkpoint_path)
    )
    missing_roles = [role for role in required_roles if role not in sources]
    if missing_roles:
        raise ValueError(
            f"LTX conversion requires source roles: {', '.join(missing_roles)}"
        )

    text_encoder_source = sources["text_encoder"]
    prepared: dict[str, PreparedComponent] = {
        "checkpoint": PreparedComponent(
            role="checkpoint",
            kind="file",
            source_id=checkpoint_source.source_id,
            source_path=checkpoint_path,
            provenance=checkpoint_source.materialization.provenance,
            resolved_ref=checkpoint_source.materialization.provenance.resolved_ref,
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
    upsampler_source = sources.get("spatial_upsampler")
    if upsampler_source is not None:
        prepared["spatial_upsampler"] = PreparedComponent(
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
        )
    distilled_lora_source = sources.get("distilled_lora")
    if distilled_lora_source is not None:
        prepared["distilled_lora"] = PreparedComponent(
            role="distilled_lora",
            kind="file",
            source_id=distilled_lora_source.source_id,
            source_path=self._resolve_required_file(
                distilled_lora_source.materialization.local_path,
                self._distilled_lora_filename,
                role="distilled_lora",
            ),
            provenance=distilled_lora_source.materialization.provenance,
            resolved_ref=distilled_lora_source.materialization.provenance.resolved_ref,
        )
    return prepared


def _prepare_bundle_components(
    self: LTXFamilyAdapter, bundle_source: ConversionSource
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
    checkpoint_path = self._resolve_checkpoint_file(bundle_root, role="bundle")
    checkpoint_variant = _checkpoint_variant_for_path(self, checkpoint_path)
    prepared: dict[str, PreparedComponent] = {
        "checkpoint": PreparedComponent(
            role="checkpoint",
            kind="file",
            source_id=bundle_source.source_id,
            source_path=checkpoint_path,
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
    upsampler_path = bundle_root / self._spatial_upsampler_filename
    if checkpoint_variant == "fast":
        prepared["spatial_upsampler"] = PreparedComponent(
            role="spatial_upsampler",
            kind="file",
            source_id=bundle_source.source_id,
            source_path=self._resolve_required_file(
                bundle_root, self._spatial_upsampler_filename, role="bundle"
            ),
            provenance=bundle_source.materialization.provenance,
            resolved_ref=bundle_source.materialization.provenance.resolved_ref,
        )
    elif upsampler_path.is_file():
        prepared["spatial_upsampler"] = PreparedComponent(
            role="spatial_upsampler",
            kind="file",
            source_id=bundle_source.source_id,
            source_path=upsampler_path,
            provenance=bundle_source.materialization.provenance,
            resolved_ref=bundle_source.materialization.provenance.resolved_ref,
        )
    distilled_lora_path = bundle_root / self._distilled_lora_filename
    if distilled_lora_path.is_file():
        prepared["distilled_lora"] = PreparedComponent(
            role="distilled_lora",
            kind="file",
            source_id=bundle_source.source_id,
            source_path=distilled_lora_path,
            provenance=bundle_source.materialization.provenance,
            resolved_ref=bundle_source.materialization.provenance.resolved_ref,
        )
    return prepared


def _artifact_components(
    self: LTXFamilyAdapter, prepared: dict[str, PreparedComponent]
) -> tuple[list[PortableArtifactComponentRecord], list[ArtifactPayloadItem]]:
    checkpoint_variant = _checkpoint_variant_for_path(
        self, prepared["checkpoint"].source_path
    )
    components: list[PortableArtifactComponentRecord] = []
    payload_items: list[ArtifactPayloadItem] = []
    required_roles = _required_roles_for_variant(self, checkpoint_variant)
    component_roles = [
        *required_roles,
        *tuple(
            role
            for role in prepared
            if role not in required_roles and role != "checkpoint"
        ),
    ]
    for role in component_roles:
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
    self: LTXFamilyAdapter,
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


def _combined_policy(
    self: LTXFamilyAdapter, prepared: dict[str, PreparedComponent]
) -> PolicyDescriptor:
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
    self: LTXFamilyAdapter,
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
    checkpoint_path = component_paths.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("LTX artifact is missing required component role 'checkpoint'")
    required_roles = _required_roles_for_variant(
        self, _checkpoint_variant_for_path(self, checkpoint_path)
    )
    missing_roles = [role for role in required_roles if role not in component_paths]
    if missing_roles:
        raise ValueError(
            f"LTX artifact is missing required component roles: {', '.join(missing_roles)}"
        )
    self._validate_text_encoder_dir(component_paths["text_encoder"], role="artifact")
    return component_paths


def _checkpoint_variant_for_file_paths(
    self: LTXFamilyAdapter, file_paths: set[str]
) -> str | None:
    if self._checkpoint_filename in file_paths:
        return "fast"
    if self._dev_checkpoint_filename in file_paths:
        return "dev"
    return None


def _checkpoint_variant_for_path(self: LTXFamilyAdapter, checkpoint_path: Path) -> str:
    if checkpoint_path.name == self._checkpoint_filename:
        return "fast"
    if checkpoint_path.name == self._dev_checkpoint_filename:
        return "dev"
    raise ValueError(
        f"LTX checkpoint '{checkpoint_path.name}' does not match a known checkpoint variant"
    )


def _required_roles_for_variant(
    self: LTXFamilyAdapter, checkpoint_variant: str
) -> tuple[str, ...]:
    if checkpoint_variant == "fast":
        return ("checkpoint", "spatial_upsampler", "text_encoder")
    if checkpoint_variant == "dev":
        return ("checkpoint", "text_encoder")
    raise ValueError(f"Unknown LTX checkpoint variant '{checkpoint_variant}'")


def _resolve_required_file(
    self: LTXFamilyAdapter, local_path: Path | None, filename: str, *, role: str
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


def _resolve_checkpoint_file(
    self: LTXFamilyAdapter, local_path: Path | None, *, role: str
) -> Path:
    if local_path is None:
        raise ValueError(f"LTX {role} source is missing a local materialization path")
    if local_path.is_file():
        if local_path.name not in self._checkpoint_filenames:
            expected = ", ".join(self._checkpoint_filenames)
            raise ValueError(
                f"LTX {role} source must point to one of {expected}, got '{local_path.name}'"
            )
        return local_path

    for filename in self._checkpoint_filenames:
        candidate = local_path / filename
        if candidate.is_file():
            return candidate

    expected = ", ".join(self._checkpoint_filenames)
    raise ValueError(f"LTX {role} source is missing one of: {expected}")


def _resolve_text_encoder_dir(
    self: LTXFamilyAdapter, local_path: Path | None, *, role: str
) -> Path:
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


def _validate_text_encoder_dir(
    self: LTXFamilyAdapter, text_encoder_root: Path, *, role: str
) -> None:
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


def _looks_like_text_encoder_snapshot(
    self: LTXFamilyAdapter, file_paths: set[str]
) -> bool:
    has_config = "config.json" in file_paths
    has_tokenizer = any(
        name in file_paths
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    has_weights = any(path.endswith(".safetensors") for path in file_paths)
    return has_config and has_tokenizer and has_weights


def _directory_files(self: LTXFamilyAdapter, root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def _directory_size(self: LTXFamilyAdapter, root: Path) -> int:
    return sum(path.stat().st_size for path in self._directory_files(root))


def _file_digest(self: LTXFamilyAdapter, source_path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(source_path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def _directory_digest(self: LTXFamilyAdapter, root: Path) -> str:
    hasher = hashlib.sha256()
    for file_path in self._directory_files(root):
        relative_path = file_path.relative_to(root).as_posix()
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(file_path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def _media_type_for_format(self: LTXFamilyAdapter, artifact_format: str) -> str:
    if artifact_format == "mp4":
        return "video/mp4"
    if artifact_format == "mov":
        return "video/quicktime"
    if artifact_format == "wav":
        return "audio/wav"
    return "application/octet-stream"


def _require_str(self: LTXFamilyAdapter, value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"LTX artifact metadata '{name}' must be a non-empty string")
    return value

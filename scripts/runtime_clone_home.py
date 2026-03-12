from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from mlxr.core.runtime import ConversionPlan, ConversionSource, RuntimeHome
from mlxr.core.runtime.manifests import (
    ArtifactManifestStore,
    ModelManifestStore,
    SourceManifestStore,
    write_json_atomic,
)
from mlxr.core.schemas import (
    ModelRecord,
    PortableArtifactComponentRecord,
    PortableArtifactRecord,
    SourceRegistrationRecord,
)
from mlxr.core.server.registry import default_runtime_registry


@dataclass(frozen=True, slots=True)
class _CloneSelection:
    model: ModelRecord
    artifact: PortableArtifactRecord
    source_records: dict[str, SourceRegistrationRecord]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an isolated MLXR runtime home by reusing installed model "
            "artifacts and source manifests from another runtime home."
        )
    )
    parser.add_argument(
        "--source-runtime-home",
        type=Path,
        default=Path.home() / ".mlx-runtime",
        help="Existing runtime home that already contains the model artifacts.",
    )
    parser.add_argument(
        "--target-runtime-home",
        type=Path,
        required=True,
        help="New runtime home to create for isolated validation or benchmarking.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        required=True,
        help="Model id to clone into the new runtime home. Repeat for multiple models.",
    )
    parser.add_argument(
        "--refresh-manifests",
        action="store_true",
        help=(
            "Rebuild the model/artifact manifests from the current family adapter "
            "using the stored source manifests, while reusing the existing payloads."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the target runtime home first if it already exists.",
    )
    return parser


def _load_selection(
    *,
    source_runtime: RuntimeHome,
    model_id: str,
) -> _CloneSelection:
    model_store = ModelManifestStore(source_runtime)
    artifact_store = ArtifactManifestStore(source_runtime)
    source_store = SourceManifestStore(source_runtime)

    model = model_store.get(model_id)
    if model is None:
        raise FileNotFoundError(
            f"Model '{model_id}' was not found under {source_runtime.models_dir}"
        )
    if model.artifact is None:
        raise ValueError(f"Model '{model_id}' has no portable artifact record")

    artifact = artifact_store.get(model.artifact.artifact_digest)
    if artifact is None:
        raise FileNotFoundError(
            f"Artifact '{model.artifact.artifact_digest}' for model '{model_id}' "
            f"was not found under {source_runtime.artifacts_portable_dir}"
        )

    source_records: dict[str, SourceRegistrationRecord] = {}
    for component in artifact.components:
        if component.source_id in source_records:
            continue
        source_record = source_store.get(component.source_id)
        if source_record is None:
            raise FileNotFoundError(
                f"Source manifest '{component.source_id}' required by model '{model_id}' "
                f"was not found under {source_runtime.sources_ref_dir}"
            )
        source_records[component.source_id] = source_record

    return _CloneSelection(
        model=model,
        artifact=artifact,
        source_records=source_records,
    )


def _prepare_target_runtime_home(
    *, target_runtime: RuntimeHome, reset: bool
) -> RuntimeHome:
    if target_runtime.root.exists():
        if not reset:
            raise FileExistsError(
                f"Target runtime home already exists: {target_runtime.root}. "
                "Pass --reset to replace it."
            )
        shutil.rmtree(target_runtime.root)
    target_runtime.ensure_layout()
    return target_runtime


def _precision_for_model(
    model: ModelRecord, artifact: PortableArtifactRecord
) -> str:
    if model.capability is not None:
        precision = model.capability.metadata.get("precision")
        if isinstance(precision, str) and precision:
            return precision
    precision = artifact.capability.metadata.get("precision")
    if isinstance(precision, str) and precision:
        return precision
    return "bf16"


def _refreshed_artifact_record(
    *,
    selection: _CloneSelection,
) -> PortableArtifactRecord:
    registry = default_runtime_registry()
    family = registry.get_family(selection.model.family)

    role_to_source_record: dict[str, SourceRegistrationRecord] = {}
    for component in selection.artifact.components:
        if component.role in role_to_source_record:
            continue
        role_to_source_record[component.role] = selection.source_records[component.source_id]

    conversion_sources: dict[str, ConversionSource] = {}
    for role, source_record in role_to_source_record.items():
        provider = registry.get_provider(source_record.source.provider)
        fetch_policy = family.fetch_policy_for_conversion(
            role, source_record.resolved_source
        )
        materialization = provider.fetch(source_record.resolved_source, fetch_policy)
        conversion_sources[role] = ConversionSource(
            role=role,
            source_id=source_record.source_id,
            source=source_record.source,
            materialization=materialization,
        )

    refreshed = family.convert(
        conversion_sources,
        ConversionPlan(
            model_id=selection.model.model_id,
            precision=_precision_for_model(selection.model, selection.artifact),
        ),
    ).record

    if refreshed.artifact_digest != selection.artifact.artifact_digest:
        raise ValueError(
            "Manifest refresh changed the artifact digest for "
            f"'{selection.model.model_id}'. This runtime clone only supports "
            "refreshing metadata for already-materialized artifacts."
        )

    existing_components = {
        (component.role, component.relative_path, component.kind): component
        for component in selection.artifact.components
    }
    refreshed_components: list[PortableArtifactComponentRecord] = []
    for component in refreshed.components:
        key = (component.role, component.relative_path, component.kind)
        existing = existing_components.get(key)
        if existing is None:
            raise ValueError(
                "Manifest refresh changed the artifact component layout for "
                f"'{selection.model.model_id}' at {component.relative_path}"
            )
        refreshed_components.append(
            component.model_copy(
                update={
                    "storage_key": existing.storage_key,
                }
            )
        )

    return refreshed.model_copy(
        update={
            "storage_key": selection.artifact.storage_key,
            "components": refreshed_components,
        }
    )


def _link_existing_artifact_payload(
    *,
    source_runtime: RuntimeHome,
    target_runtime: RuntimeHome,
    artifact: PortableArtifactRecord,
) -> None:
    source_artifact_dir = source_runtime.artifact_dir(
        artifact.family, artifact.model_id, artifact.artifact_digest
    )
    if not source_artifact_dir.exists():
        raise FileNotFoundError(
            f"Source artifact directory is missing: {source_artifact_dir}"
        )

    target_artifact_dir = target_runtime.artifact_dir(
        artifact.family, artifact.model_id, artifact.artifact_digest
    )
    target_artifact_dir.mkdir(parents=True, exist_ok=True)

    for child in source_artifact_dir.iterdir():
        if child.name == "artifact.json":
            continue
        target_child = target_artifact_dir / child.name
        if target_child.exists() or target_child.is_symlink():
            if target_child.is_dir() and not target_child.is_symlink():
                shutil.rmtree(target_child)
            else:
                target_child.unlink()
        target_child.symlink_to(child, target_is_directory=child.is_dir())


def _persist_source_manifests(
    *,
    target_runtime: RuntimeHome,
    source_records: dict[str, SourceRegistrationRecord],
) -> None:
    for source_record in source_records.values():
        path = target_runtime.source_manifest_path(
            source_record.source.provider, source_record.source_id
        )
        write_json_atomic(path, source_record)


def _persist_model_and_artifact(
    *,
    target_runtime: RuntimeHome,
    model: ModelRecord,
    artifact: PortableArtifactRecord,
) -> None:
    artifact_path = target_runtime.artifact_manifest_path(
        artifact.family, artifact.model_id, artifact.artifact_digest
    )
    write_json_atomic(artifact_path, artifact)

    refreshed_model = model.model_copy(
        update={
            "artifact": artifact,
            "capability": artifact.capability,
            "loaded": False,
        }
    )
    write_json_atomic(target_runtime.model_manifest_path(model.model_id), refreshed_model)


def clone_runtime_home(
    *,
    source_runtime: RuntimeHome,
    target_runtime: RuntimeHome,
    model_ids: list[str],
    refresh_manifests: bool,
    reset: bool,
) -> None:
    _prepare_target_runtime_home(target_runtime=target_runtime, reset=reset)
    for model_id in model_ids:
        selection = _load_selection(source_runtime=source_runtime, model_id=model_id)
        artifact = (
            _refreshed_artifact_record(
                selection=selection,
            )
            if refresh_manifests
            else selection.artifact
        )
        _persist_source_manifests(
            target_runtime=target_runtime,
            source_records=selection.source_records,
        )
        _link_existing_artifact_payload(
            source_runtime=source_runtime,
            target_runtime=target_runtime,
            artifact=selection.artifact,
        )
        _persist_model_and_artifact(
            target_runtime=target_runtime,
            model=selection.model,
            artifact=artifact,
        )


def main() -> int:
    args = build_parser().parse_args()
    clone_runtime_home(
        source_runtime=RuntimeHome(args.source_runtime_home.expanduser().resolve()),
        target_runtime=RuntimeHome(args.target_runtime_home.expanduser().resolve()),
        model_ids=list(dict.fromkeys(args.model_id)),
        refresh_manifests=bool(args.refresh_manifests),
        reset=bool(args.reset),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

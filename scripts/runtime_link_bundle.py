"""Register a trusted local bundle by linking payload files into a runtime home."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

from mlxr.core.runtime import (
    ArtifactPayloadItem,
    ConversionPlan,
    ConversionSource,
    RuntimeHome,
)
from mlxr.core.runtime.manifests import write_json_atomic
from mlxr.core.schemas import (
    ModelRecord,
    SourceRef,
    SourceRegistrationRecord,
)
from mlxr.core.server.registry import default_runtime_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Register a trusted local bundle into an MLXR runtime home by "
            "linking payload files instead of copying them."
        )
    )
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--family-variant",
        default=None,
        help=(
            "Optional family-variant hint for local bundles whose path does not "
            "already imply the canonical upstream row."
        ),
    )
    parser.add_argument(
        "--runtime-home",
        type=Path,
        required=True,
        help="Target runtime home to populate.",
    )
    parser.add_argument("--precision", default="bf16")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the target runtime home first if it already exists.",
    )
    return parser


def _prepare_runtime_home(runtime_home: RuntimeHome, *, reset: bool) -> RuntimeHome:
    if runtime_home.root.exists():
        if reset:
            shutil.rmtree(runtime_home.root)
    runtime_home.ensure_layout()
    return runtime_home


def _link_payload_items(
    *,
    runtime_home: RuntimeHome,
    family: str,
    model_id: str,
    artifact_digest: str,
    payload_items: Sequence[ArtifactPayloadItem],
) -> None:
    artifact_root = runtime_home.artifact_dir(family, model_id, artifact_digest)
    artifact_root.mkdir(parents=True, exist_ok=True)
    for item in payload_items:
        destination = artifact_root / item.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(item.source_path)


def main() -> int:
    args = build_parser().parse_args()
    bundle_path = args.bundle_path.expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle path does not exist: {bundle_path}")

    runtime_home = _prepare_runtime_home(
        RuntimeHome(args.runtime_home.expanduser().resolve()),
        reset=bool(args.reset),
    )
    registry = default_runtime_registry()
    family = registry.get_family(str(args.family))
    provider = registry.get_provider("local")

    locator: dict[str, str] = {"path": str(bundle_path)}
    if args.family_variant:
        locator["variant"] = str(args.family_variant)
    source_ref = SourceRef(
        provider="local", locator=locator, family_hint=str(args.family)
    )

    resolved = provider.resolve(source_ref)
    provenance = provider.provenance(resolved)
    source_record = SourceRegistrationRecord(
        source_id=f"src_linked_{args.model_id}",
        source=source_ref,
        resolved_source=resolved,
        provenance=provenance,
        family_hint=str(args.family),
    )
    write_json_atomic(
        runtime_home.source_manifest_path(source_ref.provider, source_record.source_id),
        source_record,
    )

    materialization = provider.fetch(
        resolved,
        family.fetch_policy_for_conversion("bundle", resolved),
    )
    artifact = family.convert(
        {
            "bundle": ConversionSource(
                role="bundle",
                source_id=source_record.source_id,
                source=source_ref,
                materialization=materialization,
            )
        },
        ConversionPlan(model_id=str(args.model_id), precision=str(args.precision)),
    )
    storage_key = runtime_home.artifact_storage_key(
        artifact.record.family,
        artifact.record.model_id,
        artifact.record.artifact_digest,
    )
    linked_record = artifact.record.model_copy(
        update={
            "storage_key": storage_key,
            "components": [
                component.model_copy(
                    update={"storage_key": f"{storage_key}/{component.relative_path}"}
                )
                for component in artifact.record.components
            ],
        }
    )
    _link_payload_items(
        runtime_home=runtime_home,
        family=linked_record.family,
        model_id=linked_record.model_id,
        artifact_digest=linked_record.artifact_digest,
        payload_items=artifact.payload_items,
    )
    write_json_atomic(
        runtime_home.artifact_manifest_path(
            linked_record.family,
            linked_record.model_id,
            linked_record.artifact_digest,
        ),
        linked_record,
    )
    write_json_atomic(
        runtime_home.model_manifest_path(str(args.model_id)),
        ModelRecord(
            model_id=str(args.model_id),
            family=linked_record.family,
            source=source_ref,
            artifact=linked_record,
            loaded=False,
            capability=linked_record.capability,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

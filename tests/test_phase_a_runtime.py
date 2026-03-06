from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault(
    "MLX_RUNTIME_HOME", str(Path(tempfile.gettempdir()) / "mlxr-test-import-home")
)

from fastapi.testclient import TestClient
from mlx_runtime_core import (
    ExecutionProfile,
    FetchPolicy,
    LocalFileProviderAdapter,
    PortableArtifact,
    ProviderInspection,
    RuntimeCatalog,
    RuntimeHome,
    RuntimeRegistry,
    SourceMaterialization,
    source_id_for_ref,
)
from mlx_runtime_family_ltx import LTXFamilyAdapter
from mlx_runtime_schemas import (
    ArtifactConversionRequest,
    AuthRequirements,
    ProvenanceRecord,
    ResolvedSource,
    SourceFileRecord,
    SourceRef,
)
from mlx_runtime_server.app import create_app
from mlx_runtime_server.state import RuntimeState
from pydantic import ValidationError

from tests.runtime_test_support import (
    LTX_CHECKPOINT_FILENAME,
    LTX_SPATIAL_UPSAMPLER_FILENAME,
    LTX_TEXT_ENCODER_DIRNAME,
    make_local_bundle,
    make_split_local_ltx_sources,
)


def make_registry() -> RuntimeRegistry:
    registry = RuntimeRegistry()
    registry.register_provider(LocalFileProviderAdapter())
    registry.register_family(LTXFamilyAdapter())
    return registry


def make_state(tmp_path: Path) -> RuntimeState:
    return RuntimeState(
        registry=make_registry(),
        runtime_home=RuntimeHome(root=tmp_path / "runtime-home"),
    )


class RecordingProvider:
    provider_id = "fake"

    def __init__(self, materialization_root: Path, *, allow_fetch: bool = True) -> None:
        self.materialization_root = materialization_root
        self.allow_fetch = allow_fetch
        self.fetch_calls: list[FetchPolicy] = []

    def resolve(self, source_ref: SourceRef) -> ResolvedSource:
        role = str(source_ref.locator.get("role", "bundle"))
        return ResolvedSource(
            provider=self.provider_id,
            locator=source_ref.locator,
            pinned_ref=f"fake-revision-{role}",
            access_state="public",
            auth_requirements=AuthRequirements(),
            files=self._files_for_role(role),
            metadata={"remote_code_approved": source_ref.policy.allow_remote_code},
        )

    def inspect(self, resolved: ResolvedSource) -> ProviderInspection:
        return ProviderInspection(
            resolved=resolved,
            bytes_total=sum(file.size_bytes or 0 for file in resolved.files),
            metadata={"file_count": len(resolved.files)},
        )

    def auth_requirements(self, source_ref: SourceRef) -> AuthRequirements:
        return AuthRequirements()

    def fetch(
        self, resolved: ResolvedSource, policy: FetchPolicy
    ) -> SourceMaterialization:
        if not self.allow_fetch:
            raise AssertionError("inspect_source should not fetch provider contents")
        self.fetch_calls.append(policy)
        local_path = self._materialize_role(str(resolved.locator.get("role", "bundle")))
        return SourceMaterialization(
            resolved=resolved,
            provenance=self.provenance(resolved),
            materialization_mode="provider-cache-ref",
            local_path=local_path,
            local_refs=(str(local_path),),
            metadata={"allow_patterns": list(policy.allow_patterns)},
        )

    def provenance(self, resolved: ResolvedSource) -> ProvenanceRecord:
        return ProvenanceRecord(
            provider=resolved.provider,
            locator=resolved.locator,
            resolved_ref=resolved.pinned_ref,
            access_state=resolved.access_state,
            remote_code_required=resolved.remote_code_required,
            remote_code_approved=bool(
                resolved.metadata.get("remote_code_approved", False)
            ),
        )

    def _files_for_role(self, role: str) -> list[SourceFileRecord]:
        if role == "checkpoint":
            return [
                SourceFileRecord(path=LTX_CHECKPOINT_FILENAME, size_bytes=128),
                SourceFileRecord(path="config.json", size_bytes=12),
            ]
        if role == "spatial_upsampler":
            return [
                SourceFileRecord(path=LTX_SPATIAL_UPSAMPLER_FILENAME, size_bytes=96),
                SourceFileRecord(path="config.json", size_bytes=12),
            ]
        if role == "text_encoder":
            return [
                SourceFileRecord(path="config.json", size_bytes=12),
                SourceFileRecord(path="tokenizer.json", size_bytes=24),
                SourceFileRecord(
                    path="model-00001-of-00001.safetensors", size_bytes=512
                ),
            ]
        if role == "bundle":
            return [
                SourceFileRecord(path=LTX_CHECKPOINT_FILENAME, size_bytes=128),
                SourceFileRecord(path=LTX_SPATIAL_UPSAMPLER_FILENAME, size_bytes=96),
                SourceFileRecord(
                    path=f"{LTX_TEXT_ENCODER_DIRNAME}/config.json", size_bytes=12
                ),
                SourceFileRecord(
                    path=f"{LTX_TEXT_ENCODER_DIRNAME}/tokenizer.json", size_bytes=24
                ),
                SourceFileRecord(
                    path=(
                        f"{LTX_TEXT_ENCODER_DIRNAME}/model-00001-of-00001.safetensors"
                    ),
                    size_bytes=512,
                ),
            ]
        raise ValueError(f"Unsupported test role '{role}'")

    def _materialize_role(self, role: str) -> Path:
        if role == "checkpoint":
            directory = self.materialization_root / role
            directory.mkdir(parents=True, exist_ok=True)
            (directory / LTX_CHECKPOINT_FILENAME).write_text(
                "checkpoint", encoding="utf-8"
            )
            return directory
        if role == "spatial_upsampler":
            directory = self.materialization_root / role
            directory.mkdir(parents=True, exist_ok=True)
            (directory / LTX_SPATIAL_UPSAMPLER_FILENAME).write_text(
                "upsampler", encoding="utf-8"
            )
            return directory
        if role == "text_encoder":
            directory = self.materialization_root / role
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
            (directory / "model-00001-of-00001.safetensors").write_text(
                "weights", encoding="utf-8"
            )
            return directory
        if role == "bundle":
            return make_local_bundle(
                self.materialization_root, directory_name="provider-bundle"
            )
        raise ValueError(f"Unsupported test role '{role}'")


class PhaseARuntimeTests(unittest.TestCase):
    def test_source_id_is_deterministic(self) -> None:
        source_ref = SourceRef(
            provider="local",
            locator={"path": "/tmp/example", "license": "test-license"},
            family_hint="ltx",
        )
        self.assertEqual(
            source_id_for_ref(source_ref), source_id_for_ref(source_ref.model_copy())
        )

    def test_runtime_home_creates_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = RuntimeHome(root=Path(tmp_dir) / "runtime-home")
            runtime_home.ensure_layout()

            self.assertTrue(runtime_home.config_dir.is_dir())
            self.assertTrue(runtime_home.logs_dir.is_dir())
            self.assertTrue(runtime_home.jobs_dir.is_dir())
            self.assertTrue(runtime_home.temp_dir.is_dir())
            self.assertTrue(runtime_home.sources_ref_dir.is_dir())
            self.assertTrue(runtime_home.artifacts_portable_dir.is_dir())
            self.assertTrue(runtime_home.build_cache_local_dir.is_dir())
            self.assertTrue(runtime_home.models_dir.is_dir())

            artifact_path = runtime_home.artifact_manifest_path(
                "ltx", "ltx-2.3-fast-local", "sha256:abc123"
            )
            self.assertEqual(artifact_path.name, "artifact.json")
            self.assertIn("sha256_abc123", artifact_path.as_posix())

    def test_local_provider_handles_file_and_directory_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            file_path = root / "bundle.safetensors"
            file_path.write_text("weights", encoding="utf-8")

            directory = root / "bundle"
            directory.mkdir()
            (directory / "a.txt").write_text("a", encoding="utf-8")
            (directory / "b.txt").write_text("bb", encoding="utf-8")

            provider = LocalFileProviderAdapter()
            file_resolved = provider.resolve(
                SourceRef(provider="local", locator={"path": str(file_path)})
            )
            dir_resolved = provider.resolve(
                SourceRef(provider="local", locator={"path": str(directory)})
            )

            self.assertEqual(len(file_resolved.files), 1)
            self.assertEqual(file_resolved.files[0].path, "bundle.safetensors")
            self.assertEqual(len(dir_resolved.files), 2)
            self.assertEqual(
                {record.path for record in dir_resolved.files}, {"a.txt", "b.txt"}
            )

    def test_source_registration_is_idempotent_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)

            state = make_state(root)
            client = TestClient(create_app(state))
            payload = {
                "provider": "local",
                "locator": {"path": str(source_dir)},
                "family_hint": "ltx",
            }

            first = client.post("/v1/sources/register", json=payload)
            second = client.post("/v1/sources/register", json=payload)

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(first.json()["source_id"], second.json()["source_id"])

            listed = client.get("/v1/sources")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(len(listed.json()), 1)

            manifest_path = state.runtime_home.source_manifest_path(
                "local", first.json()["source_id"]
            )
            self.assertTrue(manifest_path.exists())

    def test_convert_persists_artifact_model_and_capability_across_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)

            state = make_state(root)
            client = TestClient(create_app(state))
            register_response = client.post(
                "/v1/sources/register",
                json={
                    "provider": "local",
                    "locator": {
                        "path": str(source_dir),
                        "license": "ltx-2-community-license-agreement",
                    },
                    "family_hint": "ltx",
                },
            )
            self.assertEqual(register_response.status_code, 200)
            source_id = register_response.json()["source_id"]

            convert_response = client.post(
                "/v1/artifacts/convert",
                json={"source_id": source_id, "model_id": "ltx-2.3-fast-local"},
            )
            self.assertEqual(convert_response.status_code, 200)
            body = convert_response.json()
            artifact_digest = body["artifact"]["artifact_digest"]

            self.assertEqual(body["model"]["model_id"], "ltx-2.3-fast-local")
            self.assertEqual(
                body["artifact"]["capability"]["scheduler_class"], "media_video_dit"
            )
            self.assertIn(
                "artifacts-portable/ltx/ltx-2.3-fast-local/",
                body["artifact"]["storage_key"],
            )
            components = {
                component["role"]: component
                for component in body["artifact"]["components"]
            }
            self.assertEqual(
                set(components), {"checkpoint", "spatial_upsampler", "text_encoder"}
            )
            self.assertEqual(
                body["artifact"]["capability"]["dependencies"]["text_encoder"]["mode"],
                "strict-local",
            )
            artifact_root = state.runtime_home.artifact_dir(
                "ltx", "ltx-2.3-fast-local", artifact_digest
            )
            self.assertTrue(
                (artifact_root / components["checkpoint"]["relative_path"]).is_file()
            )
            self.assertTrue(
                (
                    artifact_root / components["spatial_upsampler"]["relative_path"]
                ).is_file()
            )
            self.assertTrue(
                (artifact_root / components["text_encoder"]["relative_path"]).is_dir()
            )
            self.assertTrue(
                (
                    artifact_root
                    / components["text_encoder"]["relative_path"]
                    / "config.json"
                ).is_file()
            )

            capabilities = client.get("/v1/capabilities")
            self.assertEqual(capabilities.status_code, 200)
            self.assertEqual(len(capabilities.json()), 1)
            self.assertEqual(
                capabilities.json()[0]["tasks"],
                ["video.generate", "video.condition.image"],
            )

            restarted = make_state(root)
            restarted_client = TestClient(create_app(restarted))
            self.assertEqual(len(restarted_client.get("/v1/sources").json()), 1)
            self.assertEqual(len(restarted_client.get("/v1/artifacts").json()), 1)
            self.assertEqual(len(restarted_client.get("/v1/models").json()), 1)
            self.assertEqual(
                restarted_client.get(f"/v1/artifacts/{artifact_digest}").json()[
                    "artifact_digest"
                ],
                artifact_digest,
            )
            self.assertEqual(
                restarted_client.get("/v1/models/ltx-2.3-fast-local").json()[
                    "artifact"
                ]["artifact_digest"],
                artifact_digest,
            )
            self.assertEqual(
                len(restarted_client.get("/v1/artifacts").json()[0]["components"]), 3
            )

    def test_convert_conflict_and_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)

            state = make_state(root)
            client = TestClient(create_app(state))

            missing_source = client.post(
                "/v1/artifacts/convert",
                json={"source_id": "src_missing", "model_id": "ltx-2.3-fast-local"},
            )
            self.assertEqual(missing_source.status_code, 404)

            missing_provider = client.post(
                "/v1/sources/inspect", json={"provider": "unknown", "locator": {}}
            )
            self.assertEqual(missing_provider.status_code, 404)

            no_family_register = client.post(
                "/v1/sources/register",
                json={"provider": "local", "locator": {"path": str(source_dir)}},
            )
            self.assertEqual(no_family_register.status_code, 200)
            no_family_convert = client.post(
                "/v1/artifacts/convert",
                json={
                    "source_id": no_family_register.json()["source_id"],
                    "model_id": "family-missing",
                },
            )
            self.assertEqual(no_family_convert.status_code, 400)

            register_response = client.post(
                "/v1/sources/register",
                json={
                    "provider": "local",
                    "locator": {"path": str(source_dir)},
                    "family_hint": "ltx",
                },
            )
            self.assertEqual(register_response.status_code, 200)
            source_id = register_response.json()["source_id"]

            first_convert = client.post(
                "/v1/artifacts/convert",
                json={
                    "source_id": source_id,
                    "model_id": "ltx-2.3-fast-local",
                    "precision": "bf16",
                },
            )
            second_convert = client.post(
                "/v1/artifacts/convert",
                json={
                    "source_id": source_id,
                    "model_id": "ltx-2.3-fast-local",
                    "precision": "q8",
                },
            )

            self.assertEqual(first_convert.status_code, 200)
            self.assertEqual(second_convert.status_code, 409)

    def test_artifact_conversion_request_validates_source_selection(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "Artifact conversion requires exactly one of source_id or source_bindings",
        ):
            ArtifactConversionRequest(model_id="ltx-2.3-fast-local")

        with self.assertRaisesRegex(
            ValidationError,
            "Artifact conversion requires exactly one of source_id or source_bindings",
        ):
            ArtifactConversionRequest(
                source_id="src_bundle",
                source_bindings={"checkpoint": "src_bundle"},
                family="ltx",
                model_id="ltx-2.3-fast-local",
            )

        request = ArtifactConversionRequest(
            source_bindings={
                "checkpoint": "src_bundle",
                "spatial_upsampler": "src_bundle",
                "text_encoder": "src_text",
            },
            family="ltx",
            model_id="ltx-2.3-fast-local",
        )
        source_bindings = request.source_bindings
        assert source_bindings is not None
        self.assertEqual(source_bindings["checkpoint"], "src_bundle")

    def test_family_inspection_does_not_fetch_provider_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry = RuntimeRegistry()
            provider = RecordingProvider(
                Path(tmp_dir) / "provider-cache", allow_fetch=False
            )
            registry.register_provider(provider)
            registry.register_family(LTXFamilyAdapter())
            catalog = RuntimeCatalog(
                registry=registry,
                runtime_home=RuntimeHome(root=Path(tmp_dir) / "runtime-home"),
            )

            inspection = catalog.inspect_source(
                SourceRef(
                    provider="fake",
                    locator={"repo": "example/model"},
                    family_hint="ltx",
                )
            )

            self.assertIsNotNone(inspection.family_inspection)
            self.assertEqual(provider.fetch_calls, [])

    def test_convert_uses_family_fetch_policy_for_selective_materialization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry = RuntimeRegistry()
            provider = RecordingProvider(Path(tmp_dir) / "provider-cache")
            registry.register_provider(provider)
            registry.register_family(LTXFamilyAdapter())
            catalog = RuntimeCatalog(
                registry=registry,
                runtime_home=RuntimeHome(root=Path(tmp_dir) / "runtime-home"),
            )

            checkpoint_source = catalog.register_source(
                SourceRef(
                    provider="fake",
                    locator={"repo": "example/checkpoint", "role": "checkpoint"},
                    family_hint="ltx",
                )
            )
            upsampler_source = catalog.register_source(
                SourceRef(
                    provider="fake",
                    locator={"repo": "example/upsampler", "role": "spatial_upsampler"},
                    family_hint="ltx",
                )
            )
            text_encoder_source = catalog.register_source(
                SourceRef(
                    provider="fake",
                    locator={"repo": "example/text-encoder", "role": "text_encoder"},
                    family_hint="ltx",
                )
            )
            catalog.convert_artifact(
                ArtifactConversionRequest(
                    source_bindings={
                        "checkpoint": checkpoint_source.source_id,
                        "spatial_upsampler": upsampler_source.source_id,
                        "text_encoder": text_encoder_source.source_id,
                    },
                    family="ltx",
                    model_id="ltx-2.3-fast-local",
                )
            )

            self.assertEqual(len(provider.fetch_calls), 3)
            policies_by_role = {
                str(fetch_policy.options["role"]): fetch_policy
                for fetch_policy in provider.fetch_calls
            }
            self.assertEqual(
                set(policies_by_role),
                {"checkpoint", "spatial_upsampler", "text_encoder"},
            )
            self.assertIn(
                LTX_CHECKPOINT_FILENAME,
                policies_by_role["checkpoint"].allow_patterns,
            )
            self.assertIn(
                LTX_SPATIAL_UPSAMPLER_FILENAME,
                policies_by_role["spatial_upsampler"].allow_patterns,
            )
            self.assertIn("*.tiktoken", policies_by_role["text_encoder"].allow_patterns)
            self.assertTrue(
                all(
                    fetch_policy.options["strict_local_text_encoding"]
                    for fetch_policy in provider.fetch_calls
                )
            )

    def test_convert_supports_multi_source_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dirs = make_split_local_ltx_sources(root)

            state = make_state(root)
            client = TestClient(create_app(state))
            source_ids: dict[str, str] = {}
            for role, source_dir in source_dirs.items():
                response = client.post(
                    "/v1/sources/register",
                    json={
                        "provider": "local",
                        "locator": {"path": str(source_dir)},
                        "family_hint": "ltx",
                    },
                )
                self.assertEqual(response.status_code, 200)
                source_ids[role] = response.json()["source_id"]

            convert_response = client.post(
                "/v1/artifacts/convert",
                json={
                    "family": "ltx",
                    "source_bindings": source_ids,
                    "model_id": "ltx-2.3-fast-split",
                },
            )
            self.assertEqual(convert_response.status_code, 200)
            body = convert_response.json()
            components = {
                component["role"]: component
                for component in body["artifact"]["components"]
            }
            self.assertEqual(
                components["checkpoint"]["source_id"], source_ids["checkpoint"]
            )
            self.assertEqual(
                components["spatial_upsampler"]["source_id"],
                source_ids["spatial_upsampler"],
            )
            self.assertEqual(
                components["text_encoder"]["source_id"], source_ids["text_encoder"]
            )
            artifact_root = state.runtime_home.artifact_dir(
                "ltx",
                "ltx-2.3-fast-split",
                body["artifact"]["artifact_digest"],
            )
            self.assertTrue(
                (
                    artifact_root / "payload" / "checkpoint" / LTX_CHECKPOINT_FILENAME
                ).is_file()
            )
            self.assertTrue(
                (
                    artifact_root
                    / "payload"
                    / "spatial_upsampler"
                    / LTX_SPATIAL_UPSAMPLER_FILENAME
                ).is_file()
            )
            self.assertTrue(
                (
                    artifact_root / "payload" / "text_encoder" / "tokenizer.json"
                ).is_file()
            )

    def test_convert_allows_reusing_single_source_in_source_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root, directory_name="ltx-shared-bundle")

            state = make_state(root)
            client = TestClient(create_app(state))
            register_response = client.post(
                "/v1/sources/register",
                json={
                    "provider": "local",
                    "locator": {"path": str(source_dir)},
                    "family_hint": "ltx",
                },
            )
            self.assertEqual(register_response.status_code, 200)
            source_id = register_response.json()["source_id"]

            convert_response = client.post(
                "/v1/artifacts/convert",
                json={
                    "family": "ltx",
                    "source_bindings": {
                        "checkpoint": source_id,
                        "spatial_upsampler": source_id,
                        "text_encoder": source_id,
                    },
                    "model_id": "ltx-2.3-fast-shared-bindings",
                },
            )

            self.assertEqual(convert_response.status_code, 200)
            body = convert_response.json()
            self.assertEqual(len(body["artifact"]["components"]), 3)
            self.assertTrue(
                all(
                    component["source_id"] == source_id
                    for component in body["artifact"]["components"]
                )
            )

    def test_convert_requires_all_ltx_roles_for_multi_source_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dirs = make_split_local_ltx_sources(root)
            source_dirs.pop("text_encoder")

            state = make_state(root)
            client = TestClient(create_app(state))
            source_ids: dict[str, str] = {}
            for role, source_dir in source_dirs.items():
                response = client.post(
                    "/v1/sources/register",
                    json={
                        "provider": "local",
                        "locator": {"path": str(source_dir)},
                        "family_hint": "ltx",
                    },
                )
                self.assertEqual(response.status_code, 200)
                source_ids[role] = response.json()["source_id"]

            convert_response = client.post(
                "/v1/artifacts/convert",
                json={
                    "family": "ltx",
                    "source_bindings": source_ids,
                    "model_id": "ltx-2.3-fast-missing-text",
                },
            )

            self.assertEqual(convert_response.status_code, 400)
            self.assertIn(
                "LTX conversion requires source roles: text_encoder",
                convert_response.json()["detail"],
            )

    def test_load_fails_when_required_artifact_payload_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)

            state = make_state(root)
            client = TestClient(create_app(state))
            register_response = client.post(
                "/v1/sources/register",
                json={
                    "provider": "local",
                    "locator": {"path": str(source_dir)},
                    "family_hint": "ltx",
                },
            )
            self.assertEqual(register_response.status_code, 200)
            source_id = register_response.json()["source_id"]

            convert_response = client.post(
                "/v1/artifacts/convert",
                json={"source_id": source_id, "model_id": "ltx-2.3-fast-local"},
            )
            self.assertEqual(convert_response.status_code, 200)
            artifact_digest = convert_response.json()["artifact"]["artifact_digest"]
            artifact_record = state.catalog.get_artifact(artifact_digest)
            artifact_root = state.runtime_home.artifact_dir(
                "ltx", "ltx-2.3-fast-local", artifact_digest
            )
            (
                artifact_root / "payload" / "checkpoint" / LTX_CHECKPOINT_FILENAME
            ).unlink()

            with self.assertRaisesRegex(
                ValueError, "missing required file component 'checkpoint'"
            ):
                state.registry.get_family("ltx").load(
                    PortableArtifact(
                        record=artifact_record, storage_path=artifact_root
                    ),
                    ExecutionProfile(task="video.generate", profile="bf16"),
                )

    def test_control_plane_logging_writes_runtime_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)

            state = make_state(root)
            with TestClient(create_app(state)) as client:
                response = client.post(
                    "/v1/sources/register",
                    json={
                        "provider": "local",
                        "locator": {"path": str(source_dir)},
                        "family_hint": "ltx",
                    },
                )
                self.assertEqual(response.status_code, 200)

            log_path = state.runtime_home.logs_dir / "control-plane.log"
            self.assertTrue(log_path.exists())
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("Control-plane app startup", content)
            self.assertIn("Source registered", content)


if __name__ == "__main__":
    unittest.main()

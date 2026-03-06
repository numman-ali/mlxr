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
    FetchPolicy,
    LocalFileProviderAdapter,
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

    def __init__(self, *, allow_fetch: bool = True) -> None:
        self.allow_fetch = allow_fetch
        self.fetch_calls: list[FetchPolicy] = []

    def resolve(self, source_ref: SourceRef) -> ResolvedSource:
        return ResolvedSource(
            provider=self.provider_id,
            locator=source_ref.locator,
            pinned_ref="fake-revision",
            access_state="public",
            auth_requirements=AuthRequirements(),
            files=[
                SourceFileRecord(path="config.json", size_bytes=12),
                SourceFileRecord(path="weights.safetensors", size_bytes=128),
            ],
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
        return SourceMaterialization(
            resolved=resolved,
            provenance=self.provenance(resolved),
            materialization_mode="provider-cache-ref",
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
            source_dir = root / "ltx-bundle"
            source_dir.mkdir()
            (source_dir / "weights.safetensors").write_text("bundle", encoding="utf-8")

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
            source_dir = root / "ltx-bundle"
            source_dir.mkdir()
            (source_dir / "weights.safetensors").write_text("bundle", encoding="utf-8")

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

    def test_convert_conflict_and_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "ltx-bundle"
            source_dir.mkdir()
            (source_dir / "weights.safetensors").write_text("bundle", encoding="utf-8")

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

    def test_family_inspection_does_not_fetch_provider_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry = RuntimeRegistry()
            provider = RecordingProvider(allow_fetch=False)
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
            provider = RecordingProvider()
            registry.register_provider(provider)
            registry.register_family(LTXFamilyAdapter())
            catalog = RuntimeCatalog(
                registry=registry,
                runtime_home=RuntimeHome(root=Path(tmp_dir) / "runtime-home"),
            )

            source_record = catalog.register_source(
                SourceRef(
                    provider="fake",
                    locator={"repo": "example/model"},
                    family_hint="ltx",
                )
            )
            catalog.convert_artifact(
                ArtifactConversionRequest(
                    source_id=source_record.source_id,
                    model_id="ltx-2.3-fast-local",
                )
            )

            self.assertEqual(len(provider.fetch_calls), 1)
            fetch_policy = provider.fetch_calls[0]
            self.assertIn("*.json", fetch_policy.allow_patterns)
            self.assertIn("*.safetensors", fetch_policy.allow_patterns)
            self.assertTrue(fetch_policy.options["strict_local_text_encoding"])

    def test_control_plane_logging_writes_runtime_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "ltx-bundle"
            source_dir.mkdir()
            (source_dir / "weights.safetensors").write_text("bundle", encoding="utf-8")

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

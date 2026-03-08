from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from mlxr.core.server.app import create_app
from mlxr.core.server.settings import ServerSettings

from tests.runtime_test_support import http_headers, make_local_bundle, make_state


class RuntimeSecurityTests(unittest.TestCase):
    def test_http_settings_require_token(self) -> None:
        settings = ServerSettings(http_enabled=True, http_host="127.0.0.1")
        with self.assertRaises(RuntimeError):
            settings.validate_startup()

    def test_http_read_routes_do_not_require_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = make_state(
                Path(tmp_dir),
                settings=ServerSettings(
                    http_enabled=True,
                    http_host="127.0.0.1",
                    http_bearer_token="secret-token",
                ),
            )
            with TestClient(create_app(state)) as client:
                response = client.get("/v1/providers")
                self.assertEqual(response.status_code, 200)
                self.assertIn("local", response.json()["providers"])
                self.assertIn("huggingface", response.json()["providers"])

    def test_http_mutations_require_bearer_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(
                root,
                settings=ServerSettings(
                    http_enabled=True,
                    http_host="127.0.0.1",
                    http_bearer_token="secret-token",
                ),
            )
            with TestClient(create_app(state)) as client:
                missing_auth = client.post(
                    "/v1/sources/register",
                    json={
                        "provider": "local",
                        "locator": {"path": str(source_dir)},
                        "family_hint": "ltx",
                    },
                )
                wrong_auth = client.post(
                    "/v1/sources/register",
                    json={
                        "provider": "local",
                        "locator": {"path": str(source_dir)},
                        "family_hint": "ltx",
                    },
                    headers=http_headers(token="wrong-token"),
                )
                self.assertEqual(missing_auth.status_code, 401)
                self.assertEqual(wrong_auth.status_code, 401)

    def test_http_mutations_enforce_origin_and_fetch_site_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = make_local_bundle(root)
            state = make_state(
                root,
                settings=ServerSettings(
                    http_enabled=True,
                    http_host="127.0.0.1",
                    http_bearer_token="secret-token",
                    allowed_origins=("http://localhost:3000",),
                ),
            )
            with TestClient(create_app(state)) as client:
                bad_origin = client.post(
                    "/v1/sources/register",
                    json={
                        "provider": "local",
                        "locator": {"path": str(source_dir)},
                        "family_hint": "ltx",
                    },
                    headers=http_headers(origin="http://evil.example"),
                )
                cross_site = client.post(
                    "/v1/sources/register",
                    json={
                        "provider": "local",
                        "locator": {"path": str(source_dir)},
                        "family_hint": "ltx",
                    },
                    headers=http_headers(fetch_site="cross-site"),
                )
                allowed = client.post(
                    "/v1/sources/register",
                    json={
                        "provider": "local",
                        "locator": {"path": str(source_dir)},
                        "family_hint": "ltx",
                    },
                    headers=http_headers(origin="http://localhost:3000"),
                )
                self.assertEqual(bad_origin.status_code, 403)
                self.assertEqual(cross_site.status_code, 403)
                self.assertEqual(allowed.status_code, 200)

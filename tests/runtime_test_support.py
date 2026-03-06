from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import TypeVar

from fastapi.testclient import TestClient
from httpx import Response
from mlx_runtime_core import RuntimeHome
from mlx_runtime_schemas import (
    ArtifactConversionResult,
    InputHandleRecord,
    JobRecord,
    SourceRegistrationRecord,
)
from mlx_runtime_server.registry import default_runtime_registry
from mlx_runtime_server.settings import ServerSettings
from mlx_runtime_server.state import RuntimeState
from pydantic import BaseModel

os.environ.setdefault(
    "MLX_RUNTIME_HOME", str(Path(tempfile.gettempdir()) / "mlxr-test-import-home")
)

ModelT = TypeVar("ModelT", bound=BaseModel)
LTX_CHECKPOINT_FILENAME = "ltx-2.3-22b-distilled.safetensors"
LTX_SPATIAL_UPSAMPLER_FILENAME = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
LTX_TEXT_ENCODER_DIRNAME = "gemma-3-12b-it-qat-q4_0-unquantized"


def make_state(tmp_path: Path, settings: ServerSettings | None = None) -> RuntimeState:
    return RuntimeState(
        registry=default_runtime_registry(),
        runtime_home=RuntimeHome(root=tmp_path / "runtime-home"),
        settings=settings or ServerSettings(),
    )


def make_local_bundle(root: Path, *, directory_name: str = "ltx-bundle") -> Path:
    source_dir = root / directory_name
    source_dir.mkdir()
    (source_dir / LTX_CHECKPOINT_FILENAME).write_text("bundle", encoding="utf-8")
    (source_dir / LTX_SPATIAL_UPSAMPLER_FILENAME).write_text(
        "upsampler", encoding="utf-8"
    )
    text_encoder_dir = source_dir / LTX_TEXT_ENCODER_DIRNAME
    _write_fake_text_encoder(text_encoder_dir)
    return source_dir


def make_split_local_ltx_sources(root: Path) -> dict[str, Path]:
    checkpoint_dir = root / "ltx-checkpoint-source"
    checkpoint_dir.mkdir()
    (checkpoint_dir / LTX_CHECKPOINT_FILENAME).write_text("bundle", encoding="utf-8")

    upsampler_dir = root / "ltx-upsampler-source"
    upsampler_dir.mkdir()
    (upsampler_dir / LTX_SPATIAL_UPSAMPLER_FILENAME).write_text(
        "upsampler", encoding="utf-8"
    )

    text_encoder_dir = root / "ltx-text-encoder-source"
    text_encoder_dir.mkdir()
    _write_fake_text_encoder(text_encoder_dir)

    return {
        "checkpoint": checkpoint_dir,
        "spatial_upsampler": upsampler_dir,
        "text_encoder": text_encoder_dir,
    }


def _write_fake_text_encoder(text_encoder_dir: Path) -> None:
    text_encoder_dir.mkdir(exist_ok=True)
    (text_encoder_dir / "config.json").write_text("{}", encoding="utf-8")
    (text_encoder_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (text_encoder_dir / "model-00001-of-00001.safetensors").write_text(
        "weights", encoding="utf-8"
    )


def response_model(response: Response, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(response.json())


def register_local_ltx_model(
    client: TestClient,
    source_dir: Path,
    *,
    headers: dict[str, str] | None = None,
    model_id: str = "ltx-2.3-fast-local",
) -> str:
    register_response = client.post(
        "/v1/sources/register",
        json={
            "provider": "local",
            "locator": {"path": str(source_dir)},
            "family_hint": "ltx",
        },
        headers=headers,
    )
    assert register_response.status_code == 200, register_response.text
    source_record = response_model(register_response, SourceRegistrationRecord)

    convert_response = client.post(
        "/v1/artifacts/convert",
        json={"source_id": source_record.source_id, "model_id": model_id},
        headers=headers,
    )
    assert convert_response.status_code == 200, convert_response.text
    response_model(convert_response, ArtifactConversionResult)
    return source_record.source_id


def import_input_handle(
    client: TestClient,
    payload: bytes,
    *,
    headers: dict[str, str] | None = None,
    media_type: str = "image/png",
) -> str:
    response = client.post(
        "/v1/inputs/import",
        json={
            "content_base64": base64.b64encode(payload).decode("ascii"),
            "media_type": media_type,
            "filename": "conditioning.png",
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response_model(response, InputHandleRecord).handle_id


def wait_for_job_terminal_state(
    client: TestClient,
    job_id: str,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        record = response_model(response, JobRecord)
        if record.state.value in {"completed", "failed", "cancelled"}:
            return record.model_dump(mode="json")
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for job '{job_id}' to finish")


def http_headers(
    *,
    token: str = "secret-token",
    origin: str | None = None,
    referer: str | None = None,
    fetch_site: str | None = None,
) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if origin is not None:
        headers["Origin"] = origin
    if referer is not None:
        headers["Referer"] = referer
    if fetch_site is not None:
        headers["Sec-Fetch-Site"] = fetch_site
    return headers

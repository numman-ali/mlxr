"""Inspect or register a trusted local model bundle through the runtime API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
from mlxr.core.runtime import RuntimeHome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register a trusted local model bundle and convert it into an MLXR portable artifact."
    )
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--family-variant",
        default=None,
        help=(
            "Optional explicit family-variant hint for local bundles whose parent "
            "directory name is not the canonical upstream row name."
        ),
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--uds-path", type=Path, default=None)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect the bundle source without registering or converting it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle_path = args.bundle_path.expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle path does not exist: {bundle_path}")

    uds_path = (
        args.uds_path.expanduser().resolve()
        if args.uds_path is not None
        else (RuntimeHome.from_env().temp_dir / "control-plane.sock")
    )
    transport = httpx.HTTPTransport(uds=str(uds_path))
    source_ref: dict[str, object] = {
        "provider": "local",
        "locator": {"path": str(bundle_path)},
        "family_hint": str(args.family),
    }
    if args.family_variant:
        locator = source_ref["locator"]
        if not isinstance(locator, dict):
            raise TypeError("source_ref locator must be a dictionary")
        locator["variant"] = str(args.family_variant)

    with httpx.Client(
        base_url="http://mlxr", transport=transport, timeout=None
    ) as client:
        if args.inspect_only:
            response = client.post("/v1/sources/inspect", json=source_ref)
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
            return 0

        register_response = client.post("/v1/sources/register", json=source_ref)
        register_response.raise_for_status()
        register_payload = register_response.json()

        convert_response = client.post(
            "/v1/artifacts/convert",
            json={
                "source_id": register_payload["source_id"],
                "model_id": str(args.model_id),
                "precision": str(args.precision),
            },
        )
        convert_response.raise_for_status()
        print(
            json.dumps(
                {
                    "source": register_payload,
                    "conversion": convert_response.json(),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

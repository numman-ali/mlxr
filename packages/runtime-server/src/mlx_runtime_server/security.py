from __future__ import annotations

from fastapi import HTTPException, Request

from .settings import ServerSettings

SAFE_FETCH_SITES = {"same-origin", "same-site", "none"}


def enforce_mutating_request_policy(request: Request, settings: ServerSettings) -> None:
    if not settings.http_enabled:
        return

    expected_auth = f"Bearer {settings.http_bearer_token}"
    provided_auth = request.headers.get("authorization")
    if provided_auth != expected_auth:
        raise HTTPException(
            status_code=401, detail="Mutating HTTP requests require bearer auth"
        )

    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site not in SAFE_FETCH_SITES:
        raise HTTPException(
            status_code=403, detail="Cross-site browser requests are not allowed"
        )

    origin = request.headers.get("origin")
    if origin is not None and not settings.origin_allowed(origin):
        raise HTTPException(status_code=403, detail="Origin is not allowed")

    referer = request.headers.get("referer")
    if referer is not None and not settings.referer_allowed(referer):
        raise HTTPException(status_code=403, detail="Referer is not allowed")

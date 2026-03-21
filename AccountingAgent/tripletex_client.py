"""
Tripletex API client — thin wrapper with auth, error handling, and logging.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class TripletexClient:
    """Authenticated client for the Tripletex v2 REST API (via proxy)."""

    def __init__(self, base_url: str, session_token: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self.timeout = timeout
        self._client = httpx.Client(
            auth=self.auth,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        # Tracking for efficiency scoring
        self.call_count = 0
        self.error_count = 0
        self.call_log: list[dict] = []

    # ── Core HTTP ────────────────────────────────────────────────────

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        self.call_count += 1
        logger.info(f"[{self.call_count}] {method} {url} params={kwargs.get('params')} body={str(kwargs.get('json',''))[:500]}")

        resp = self._client.request(method, url, **kwargs)
        log_entry = {
            "n": self.call_count,
            "method": method,
            "url": url,
            "status": resp.status_code,
        }
        self.call_log.append(log_entry)

        if resp.status_code >= 400:
            self.error_count += 1
            body = resp.text[:1000]
            logger.warning(f"  → {resp.status_code} {body}")
            log_entry["error"] = body
            # Return the error so the agent can reason about it
            return {"_error": True, "status_code": resp.status_code, "detail": body}

        if resp.status_code == 204 or not resp.content:
            return {"_ok": True}

        data = resp.json()
        logger.info(f"  → {resp.status_code} OK")
        return data

    def get(self, path: str, params: dict | None = None, **kw: Any) -> dict:
        return self._request("GET", path, params=params, **kw)

    def post(self, path: str, json: Any = None, params: dict | None = None, **kw: Any) -> dict:
        return self._request("POST", path, json=json, params=params, **kw)

    def put(self, path: str, json: Any = None, params: dict | None = None, **kw: Any) -> dict:
        return self._request("PUT", path, json=json, params=params, **kw)

    def delete(self, path: str, params: dict | None = None, **kw: Any) -> dict:
        return self._request("DELETE", path, params=params, **kw)

    # ── Convenience helpers ──────────────────────────────────────────

    def list_values(self, path: str, params: dict | None = None) -> list[dict]:
        """GET a list endpoint and return the values array."""
        data = self.get(path, params=params)
        if "_error" in data:
            return data  # pass error through
        return data.get("values", [])

    def create(self, path: str, body: dict, params: dict | None = None) -> dict:
        """POST to create an entity and return the created value."""
        data = self.post(path, json=body, params=params)
        if "_error" in data:
            return data
        return data.get("value", data)

    def update(self, path: str, body: dict, params: dict | None = None) -> dict:
        """PUT to update an entity and return the updated value."""
        data = self.put(path, json=body, params=params)
        if "_error" in data:
            return data
        return data.get("value", data)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

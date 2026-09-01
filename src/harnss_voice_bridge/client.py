from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_SETTINGS_PATH = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Harnss"
    / "openacpui-data"
    / "settings.json"
)


class HarnssRemoteError(RuntimeError):
    """Raised when Harnss Remote Control rejects or cannot process a request."""


def load_pairing_token(settings_path: Path = DEFAULT_SETTINGS_PATH) -> str:
    """Load a local pairing token without ever logging it."""
    env_token = os.environ.get("HARNSS_REMOTE_TOKEN", "").strip()
    if env_token:
        return env_token

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnssRemoteError(f"Cannot read Harnss settings: {settings_path}") from exc

    token = settings.get("remoteToken")
    if not isinstance(token, str) or not token:
        raise HarnssRemoteError("Harnss remote pairing token is not configured")
    return token


def load_remote_endpoint(
    settings_path: Path = DEFAULT_SETTINGS_PATH,
) -> tuple[str, int]:
    """Return the loopback endpoint configured by Harnss."""
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnssRemoteError(f"Cannot read Harnss settings: {settings_path}") from exc

    enabled = settings.get("remoteEnabled", False)
    if enabled is not True:
        raise HarnssRemoteError("Harnss Remote Control is disabled")

    port = settings.get("remotePort", 8787)
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        port = 8787
    return "127.0.0.1", port


class HarnssClient:
    """Small stdlib-only client for the existing Harnss Remote API."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8787,
        token: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.token = token or load_pairing_token()
        self.timeout = timeout

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=payload,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HarnssRemoteError(
                f"Harnss Remote API HTTP {exc.code}: {detail[:300]}"
            ) from exc
        except URLError as exc:
            raise HarnssRemoteError(f"Cannot reach Harnss Remote API: {exc.reason}") from exc

        try:
            result = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise HarnssRemoteError("Harnss Remote API returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise HarnssRemoteError("Harnss Remote API returned a non-object response")
        return result

    def health(self) -> dict[str, Any]:
        return self._request("/api/health")

    def list_sessions(self) -> list[dict[str, Any]]:
        result = self._request("/api/sessions")
        sessions = result.get("sessions", [])
        if not isinstance(sessions, list):
            raise HarnssRemoteError("Harnss returned an invalid sessions list")
        return [session for session in sessions if isinstance(session, dict)]

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request(f"/api/sessions/{session_id}")

    def send_message(self, session_id: str, text: str) -> dict[str, Any]:
        """Send text through Harnss's normal session path.

        Callers must apply their own risk policy before invoking this method.
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        return self._request(
            f"/api/sessions/{session_id}/messages",
            method="POST",
            body={"text": text},
        )


def session_line(session: dict[str, Any]) -> str:
    session_id = str(session.get("id", "?"))
    title = str(session.get("title") or "Untitled")
    engine = str(session.get("engine") or "?")
    model = str(session.get("model") or "default")
    state = (
        "approval"
        if session.get("hasPendingPermission")
        else "working"
        if session.get("isProcessing")
        else "idle"
    )
    return f"{session_id} | {engine} | {model} | {state} | {title}"

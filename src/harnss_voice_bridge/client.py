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


APPROVAL_BEHAVIORS = frozenset({"allow", "allowForSession", "deny"})
SESSION_ACTIONS = frozenset({"activate", "interrupt", "retry"})
SESSION_ENGINES = frozenset({"claude", "codex", "acp"})


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

    def respond_approval(
        self,
        session_id: str,
        request_id: str,
        behavior: str,
        *,
        answers_by_question_id: dict[str, list[str]] | None = None,
        deny_message: str | None = None,
    ) -> dict[str, Any]:
        """Answer a pending permission request through Harnss.

        Callers must apply their own risk policy before invoking this method.
        """
        if behavior not in APPROVAL_BEHAVIORS:
            raise ValueError("behavior must be one of: allow, allowForSession, deny")
        body: dict[str, Any] = {"behavior": behavior}
        if answers_by_question_id:
            body["answersByQuestionId"] = {
                str(question_id): [str(answer) for answer in answers]
                for question_id, answers in answers_by_question_id.items()
            }
        if deny_message:
            body["denyMessage"] = deny_message
        return self._request(
            f"/api/sessions/{session_id}/approvals/{request_id}",
            method="POST",
            body=body,
        )

    def session_action(self, session_id: str, action: str) -> dict[str, Any]:
        """Run a session action (activate, interrupt, or retry)."""
        if action not in SESSION_ACTIONS:
            raise ValueError("action must be one of: activate, interrupt, retry")
        return self._request(
            f"/api/sessions/{session_id}/actions",
            method="POST",
            body={"action": action},
        )

    def update_settings(
        self,
        session_id: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        permission_mode: str | None = None,
        plan_mode: bool | None = None,
        acp_permission_behavior: str | None = None,
    ) -> dict[str, Any]:
        """Update session settings; at least one field must be provided."""
        body: dict[str, Any] = {}
        if model is not None:
            body["model"] = model
        if effort is not None:
            body["effort"] = effort
        if permission_mode is not None:
            body["permissionMode"] = permission_mode
        if plan_mode is not None:
            body["planMode"] = plan_mode
        if acp_permission_behavior is not None:
            body["acpPermissionBehavior"] = acp_permission_behavior
        if not body:
            raise ValueError("at least one setting must be provided")
        return self._request(
            f"/api/sessions/{session_id}/settings",
            method="POST",
            body=body,
        )

    def create_session(
        self,
        project_id: str,
        engine: str,
        *,
        model: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new Harnss session in the given project.

        Requires the POST /api/sessions route on the Harnss side.
        """
        if engine not in SESSION_ENGINES:
            raise ValueError("engine must be one of: claude, codex, acp")
        body: dict[str, Any] = {"projectId": project_id, "engine": engine}
        if model:
            body["model"] = model
        if agent_id:
            body["agentId"] = agent_id
        return self._request("/api/sessions", method="POST", body=body)


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

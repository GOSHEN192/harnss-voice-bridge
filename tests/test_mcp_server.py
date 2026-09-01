"""Tests for the Harnss MCP service layer (no transport involved)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from harnss_voice_bridge.mcp_server import (
    MAX_RESULT_BYTES,
    HarnssMcpService,
    _tool,
    build_server,
    truncate_json,
)


class FakeClient:
    def __init__(self, snapshot: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.snapshot = snapshot or {
            "session": {"id": "session-1", "engine": "codex", "title": "T"},
            "messages": [{"role": "user", "content": "hi"}],
            "isProcessing": False,
            "pendingPermission": {
                "requestId": "request-9",
                "toolName": "Bash",
                "kind": "approval",
                "summary": "run tests",
            },
        }

    def _record(self, name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((name, args, kwargs))
        return {"accepted": True}

    def list_sessions(self) -> list[dict[str, Any]]:
        self.calls.append(("list_sessions", (), {}))
        return [
            {
                "id": "session-1",
                "projectId": "proj-1",
                "projectName": "Harnss",
                "title": "T",
                "engine": "codex",
                "isProcessing": True,
                "hasPendingPermission": False,
                "lastMessageAt": 20,
            },
            {"id": "session-2", "title": "Old", "lastMessageAt": 10},
        ]

    def get_session(self, session_id: str) -> dict[str, Any]:
        self.calls.append(("get_session", (session_id,), {}))
        return self.snapshot

    def send_message(self, session_id: str, text: str) -> dict[str, Any]:
        return self._record("send_message", session_id, text)

    def respond_approval(
        self,
        session_id: str,
        request_id: str,
        behavior: str,
        *,
        answers_by_question_id=None,
        deny_message=None,
    ) -> dict[str, Any]:
        return self._record(
            "respond_approval",
            session_id,
            request_id,
            behavior=behavior,
            answers_by_question_id=answers_by_question_id,
            deny_message=deny_message,
        )

    def session_action(self, session_id: str, action: str) -> dict[str, Any]:
        return self._record("session_action", session_id, action=action)

    def update_settings(self, session_id: str, **payload: Any) -> dict[str, Any]:
        return self._record("update_settings", session_id, **payload)

    def create_session(
        self,
        project_id: str,
        engine: str,
        *,
        model=None,
        agent_id=None,
    ) -> dict[str, Any]:
        return self._record(
            "create_session",
            project_id,
            engine,
            model=model,
            agent_id=agent_id,
        )


def service_with(snapshot: dict[str, Any] | None = None) -> tuple[HarnssMcpService, FakeClient]:
    client = FakeClient(snapshot)
    return HarnssMcpService(client), client


def test_list_sessions_sorts_and_trims() -> None:
    service, _ = service_with()
    result = service.list_sessions()
    assert [session["id"] for session in result["sessions"]] == ["session-1", "session-2"]
    assert set(result["sessions"][0]) <= {
        "id",
        "projectId",
        "projectName",
        "title",
        "engine",
        "agentId",
        "model",
        "effort",
        "permissionMode",
        "planMode",
        "acpPermissionBehavior",
        "isProcessing",
        "hasPendingPermission",
        "lastMessageAt",
    }


def test_get_session_hides_request_ids_and_caps_messages() -> None:
    snapshot = {
        "session": {"id": "session-1", "engine": "claude", "title": "T"},
        "isProcessing": True,
        "totalCost": 0.5,
        "pendingPermission": {
            "requestId": "request-9",
            "toolUseId": "tool-1",
            "toolName": "Bash",
            "kind": "approval",
            "summary": "run tests",
        },
        "messages": [
            {"role": "user", "content": f"m{i}", "secretField": "x"}
            for i in range(30)
        ],
    }
    service, _ = service_with(snapshot)

    result = service.get_session("session-1")

    assert result["pendingPermission"] == {
        "toolName": "Bash",
        "kind": "approval",
        "decisionReason": None,
        "summary": "run tests",
        "questions": [],
    }
    assert len(result["recentMessages"]) == 20
    assert result["recentMessages"][-1]["content"] == "m29"
    assert all("secretField" not in message for message in result["recentMessages"])


def test_respond_approval_resolves_request_id_from_snapshot() -> None:
    service, client = service_with()
    result = service.respond_approval("session-1", "allow")
    assert result["ok"] is True
    assert client.calls[-1] == (
        "respond_approval",
        ("session-1", "request-9"),
        {"behavior": "allow", "answers_by_question_id": None, "deny_message": None},
    )


def test_respond_approval_without_pending_permission_fails() -> None:
    snapshot = {"session": {"id": "s"}, "pendingPermission": None, "messages": []}
    service, _ = service_with(snapshot)
    result = service.respond_approval("s", "allow")
    assert result["ok"] is False
    assert "no pending permission" in result["error"]


def test_allow_for_session_blocked_for_write_tool() -> None:
    service, client = service_with()  # pending tool is Bash
    result = service.respond_approval("session-1", "allowForSession")
    assert result["ok"] is False
    assert "read-only tools" in result["error"]
    assert client.calls, "snapshot must still have been read"
    assert all(name != "respond_approval" for name, _, _ in client.calls)


def test_allow_for_session_allowed_for_read_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {
        "session": {"id": "s"},
        "pendingPermission": {
            "requestId": "request-9",
            "toolName": "Read",
            "kind": "approval",
        },
        "messages": [],
    }
    service, client = service_with(snapshot)
    result = service.respond_approval("session-1", "allowForSession")
    assert result["ok"] is True
    assert client.calls[-1][1] == ("session-1", "request-9")


def test_allow_for_session_opt_out_env_lifts_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNSS_VOICE_SESSION_SCOPE_APPROVAL", "1")
    service, client = service_with()
    result = service.respond_approval("session-1", "allowForSession")
    assert result["ok"] is True
    assert client.calls[-1][1] == ("session-1", "request-9")


def test_allow_for_session_rejected_for_user_input() -> None:
    snapshot = {
        "session": {"id": "s"},
        "pendingPermission": {
            "requestId": "request-9",
            "kind": "user-input",
            "questions": [{"id": "q1", "question": "Which?"}],
        },
        "messages": [],
    }
    service, client = service_with(snapshot)
    result = service.respond_approval("session-1", "allowForSession")
    assert result["ok"] is False
    assert "user-input" in result["error"]
    result = service.respond_approval(
        "session-1",
        "allow",
        answers_by_question_id={"q1": ["left"]},
    )
    assert result["ok"] is True


def test_send_message_rejects_empty_and_forwards_text() -> None:
    service, client = service_with()
    with pytest.raises(ValueError, match="non-empty"):
        service.send_message("session-1", "   ")
    wrapped = _tool(service.send_message)
    assert json.loads(wrapped("session-1", " ")) == {
        "ok": False,
        "error": "text must be a non-empty string",
    }
    result = service.send_message("session-1", "run the tests")
    assert result["ok"] is True
    assert client.calls[-1][:2] == ("send_message", ("session-1", "run the tests"))


def test_interrupt_and_settings_forward_payloads() -> None:
    service, client = service_with()
    assert service.interrupt("session-1")["ok"] is True
    assert client.calls[-1] == (
        "session_action",
        ("session-1",),
        {"action": "interrupt"},
    )
    assert service.set_session_settings("session-1", model="m1")["ok"] is True
    assert client.calls[-1][:2] == ("update_settings", ("session-1",))
    assert client.calls[-1][2] == {"model": "m1"}


def test_create_session_forwards_fields_and_rejects_bad_engine() -> None:
    service, client = service_with()
    result = service.create_session("proj-1", "codex", model="m", agent_id="a")
    assert result["ok"] is True
    assert client.calls[-1][1] == ("proj-1", "codex")
    assert client.calls[-1][2] == {"model": "m", "agent_id": "a"}
    with pytest.raises(ValueError, match="engine"):
        service.create_session("proj-1", "shell")
    assert json.loads(_tool(service.create_session)("proj-1", "shell")) == {
        "ok": False,
        "error": "engine must be one of: claude, codex, acp",
    }


def test_truncate_json_caps_output() -> None:
    small = {"a": "x"}
    assert json.loads(truncate_json(small)) == small
    huge = {"blob": "x" * (MAX_RESULT_BYTES * 2)}
    text = truncate_json(huge)
    assert len(text.encode("utf-8")) <= MAX_RESULT_BYTES
    assert text.endswith("…[truncated]")


def test_build_server_registers_seven_tools() -> None:
    import asyncio

    service, _ = service_with()
    server = build_server(service)
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert names == {
        "list_sessions",
        "get_session",
        "send_message",
        "respond_approval",
        "interrupt",
        "create_session",
        "set_session_settings",
    }

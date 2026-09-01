import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from harnss_voice_bridge.client import (
    HarnssClient,
    HarnssRemoteError,
    load_pairing_token,
    load_remote_endpoint,
    session_line,
)


def write_settings(path: Path, **values: object) -> None:
    import json

    path.write_text(json.dumps(values), encoding="utf-8")


def test_load_endpoint_and_token(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    write_settings(settings, remoteEnabled=True, remotePort=9876, remoteToken="secret")

    assert load_remote_endpoint(settings) == ("127.0.0.1", 9876)
    assert load_pairing_token(settings) == "secret"


def test_disabled_remote_control_is_rejected(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    write_settings(settings, remoteEnabled=False, remoteToken="secret")

    with pytest.raises(HarnssRemoteError, match="disabled"):
        load_remote_endpoint(settings)


def test_session_line_is_non_sensitive() -> None:
    line = session_line(
        {
            "id": "session-1",
            "title": "Read-only test",
            "engine": "codex",
            "model": "gpt-5.6-luna",
            "isProcessing": False,
        }
    )
    assert line == "session-1 | codex | gpt-5.6-luna | idle | Read-only test"


def test_empty_message_is_rejected() -> None:
    client = HarnssClient(token="test")
    try:
        client.send_message("session-1", "   ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected empty message rejection")


class _RecordingHandler(BaseHTTPRequestHandler):
    captured: ClassVar[list[dict[str, Any]]] = []
    status: ClassVar[int] = 200
    body: ClassVar[dict[str, Any]] = {}

    def _record(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        _RecordingHandler.captured.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "json": json.loads(raw) if raw else None,
            }
        )
        payload = json.dumps(_RecordingHandler.body).encode("utf-8")
        self.send_response(_RecordingHandler.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _record
    do_POST = _record

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture()
def fake_harnss():
    _RecordingHandler.captured = []
    _RecordingHandler.status = 200
    _RecordingHandler.body = {"accepted": True}
    httpd = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    yield f"http://127.0.0.1:{port}", _RecordingHandler.captured
    httpd.shutdown()


def _client_at(base_url: str) -> HarnssClient:
    host, port = base_url.replace("http://", "").split(":")
    return HarnssClient(host=host, port=int(port), token="test-token")


def test_respond_approval_posts_behavior_and_answers(fake_harnss) -> None:
    base_url, captured = fake_harnss
    client = _client_at(base_url)

    result = client.respond_approval(
        "session-1",
        "request-9",
        "allowForSession",
        answers_by_question_id={"q1": ["yes"]},
    )

    assert result == {"accepted": True}
    record = captured[0]
    assert record["method"] == "POST"
    assert record["path"] == "/api/sessions/session-1/approvals/request-9"
    assert record["authorization"] == "Bearer test-token"
    assert record["json"] == {
        "behavior": "allowForSession",
        "answersByQuestionId": {"q1": ["yes"]},
    }


def test_respond_approval_rejects_unknown_behavior(fake_harnss) -> None:
    base_url, _ = fake_harnss
    with pytest.raises(ValueError, match="behavior"):
        _client_at(base_url).respond_approval("session-1", "request-9", "maybe")


def test_session_action_posts_action(fake_harnss) -> None:
    base_url, captured = fake_harnss
    _client_at(base_url).session_action("session-1", "interrupt")
    assert captured[0]["json"] == {"action": "interrupt"}
    assert captured[0]["path"] == "/api/sessions/session-1/actions"


def test_session_action_rejects_unknown_action(fake_harnss) -> None:
    base_url, _ = fake_harnss
    with pytest.raises(ValueError, match="action"):
        _client_at(base_url).session_action("session-1", "restart")


def test_update_settings_maps_to_camel_case(fake_harnss) -> None:
    base_url, captured = fake_harnss
    _client_at(base_url).update_settings(
        "session-1",
        model="m1",
        plan_mode=True,
        permission_mode="default",
    )
    assert captured[0]["json"] == {
        "model": "m1",
        "permissionMode": "default",
        "planMode": True,
    }


def test_update_settings_requires_a_field(fake_harnss) -> None:
    base_url, captured = fake_harnss
    with pytest.raises(ValueError, match="at least one"):
        _client_at(base_url).update_settings("session-1")
    assert not captured


def test_create_session_posts_project_and_engine(fake_harnss) -> None:
    base_url, captured = fake_harnss
    _client_at(base_url).create_session("proj-1", "codex", model="m", agent_id="a")
    assert captured[0]["path"] == "/api/sessions"
    assert captured[0]["json"] == {
        "projectId": "proj-1",
        "engine": "codex",
        "model": "m",
        "agentId": "a",
    }


def test_create_session_rejects_unknown_engine(fake_harnss) -> None:
    base_url, _ = fake_harnss
    with pytest.raises(ValueError, match="engine"):
        _client_at(base_url).create_session("proj-1", "shell")


def test_http_error_surfaces_status_and_detail(fake_harnss) -> None:
    base_url, _ = fake_harnss
    _RecordingHandler.status = 404
    _RecordingHandler.body = {"error": "Session not found"}
    with pytest.raises(HarnssRemoteError, match="HTTP 404.*Session not found"):
        _client_at(base_url).session_action("session-1", "interrupt")

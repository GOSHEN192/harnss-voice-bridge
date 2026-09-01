import json
from typing import Any

from harnss_voice_bridge.tools import HarnssToolExecutor, explicit_confirmation


class FakeHarnss:
    def list_sessions(self) -> list[dict[str, Any]]:
        return [{"id": "s1", "title": "Safe test"}]

    def get_session(self, session_id: str) -> dict[str, Any]:
        return {"id": session_id, "isProcessing": False}

    def send_message(self, session_id: str, text: str) -> dict[str, Any]:
        return {"accepted": True, "sessionId": session_id, "text": text}


def test_confirmation_phrases_are_explicit() -> None:
    assert explicit_confirmation("确认发送") is True
    assert explicit_confirmation("Please confirm send") is True
    assert explicit_confirmation("确认一下") is False


def test_read_only_tool_does_not_require_send_permission() -> None:
    executor = HarnssToolExecutor(FakeHarnss())
    result = json.loads(executor.execute("harnss_list_sessions", "{}"))
    assert result == {"ok": True, "sessions": [{"id": "s1", "title": "Safe test"}]}


def test_send_requires_runtime_flag_and_spoken_confirmation() -> None:
    executor = HarnssToolExecutor(FakeHarnss(), allow_send=True)
    denied = json.loads(
        executor.execute(
            "harnss_send_message",
            {"session_id": "s1", "text": "read-only check"},
        )
    )
    assert denied["confirmation_required"] is True

    executor.set_transcript("确认发送")
    accepted = json.loads(
        executor.execute(
            "harnss_send_message",
            {"session_id": "s1", "text": "read-only check"},
        )
    )
    assert accepted["ok"] is True
    assert accepted["accepted"]["accepted"] is True


def test_send_is_disabled_by_default_even_after_confirmation() -> None:
    executor = HarnssToolExecutor(FakeHarnss())
    executor.set_transcript("确认发送")
    result = json.loads(
        executor.execute(
            "harnss_send_message",
            {"session_id": "s1", "text": "read-only check"},
        )
    )
    assert result["ok"] is False
    assert "disabled" in result["error"]

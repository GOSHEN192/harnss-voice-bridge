from pathlib import Path

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

import json
import sys
import types

from openai.types.realtime import ConversationItemCreateEvent

from harnss_voice_bridge.client import HarnssClient
from harnss_voice_bridge.realtime import (
    AudioDevice,
    RealtimeVoiceClient,
    format_audio_device,
    is_stop_voice_phrase,
    list_audio_devices,
    parse_device,
)
from harnss_voice_bridge.speech_runtime import split_realtime_text


def test_session_update_exposes_only_guarded_harnss_tools() -> None:
    voice = RealtimeVoiceClient(
        endpoint="ws://127.0.0.1:8765/v1/realtime",
        harnss=HarnssClient(token="test"),
    )
    session = voice._session_update()["session"]
    names = [tool["name"] for tool in session["tools"]]
    assert names == [
        "harnss_list_sessions",
        "harnss_get_session",
        "harnss_send_message",
    ]
    assert session["tool_choice"] == "auto"
    assert "confirm send" in session["instructions"]


def test_voice_instructions_prioritize_direct_concise_replies() -> None:
    voice = RealtimeVoiceClient(
        endpoint="ws://127.0.0.1:8765/v1/realtime",
        harnss=HarnssClient(token="test"),
    )
    instructions = voice._session_update()["session"]["instructions"]
    assert "直接回答用户当前的问题" in instructions
    assert "不要重复寒暄" in instructions
    assert "每次只说一到两句短句" in instructions


def test_chinese_streaming_text_flushes_before_punctuation() -> None:
    assert split_realtime_text("听得见我在") == ["听得见我在"]
    assert split_realtime_text("我先帮你查看相关会话") == [
        "我先帮你查看相关会话",
    ]
    assert split_realtime_text("好的。现在处理") == ["好的。", "现在处理"]
    assert split_realtime_text("一" * 29) == ["一" * 28, "一"]


def test_explicit_voice_shutdown_phrases_are_local() -> None:
    assert is_stop_voice_phrase("关闭语音")
    assert is_stop_voice_phrase("请停止实时语音")
    assert is_stop_voice_phrase(" stop   voice ")
    assert not is_stop_voice_phrase("关闭 Harnss 会话")


def test_function_call_output_is_valid_realtime_event() -> None:
    event = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": json.dumps({"ok": True}),
        },
    }
    parsed = ConversationItemCreateEvent.model_validate(event)
    assert parsed.item.type == "function_call_output"


def test_parse_device_accepts_index_or_name() -> None:
    assert parse_device(None) is None
    assert parse_device(" 2 ") == 2
    assert parse_device("External Headphones") == "External Headphones"


def test_parse_device_rejects_empty_value() -> None:
    import pytest

    with pytest.raises(ValueError, match="must not be empty"):
        parse_device(" ")


def test_list_audio_devices_returns_display_safe_fields(monkeypatch) -> None:
    fake_sounddevice = types.SimpleNamespace(
        query_devices=lambda: [
            {
                "name": "Built-in Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    assert list_audio_devices() == [
        {
            "index": 0,
            "name": "Built-in Microphone",
            "max_input_channels": 2,
            "max_output_channels": 0,
        }
    ]
    assert format_audio_device(list_audio_devices()[0]) == (
        "0 | Built-in Microphone | in=2 out=0"
    )


def test_audio_device_selection_is_passed_to_streams(monkeypatch) -> None:
    created: list[dict[str, object]] = []

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def close(self):
            pass

    voice = RealtimeVoiceClient(
        endpoint="ws://unused",
        harnss=HarnssClient(token="test"),
        device=AudioDevice(input_device=3, output_device="USB Speaker"),
    )
    voice._open_audio_streams(
        types.SimpleNamespace(RawOutputStream=FakeStream, RawInputStream=FakeStream),
        lambda *_args: None,
    )
    assert created[0]["device"] == "USB Speaker"
    assert created[1]["device"] == 3

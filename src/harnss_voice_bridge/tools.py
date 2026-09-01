from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .client import HarnssClient


HARNSS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "harnss_list_sessions",
        "description": "List available Harnss Agent sessions. Read-only.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "harnss_get_session",
        "description": "Inspect one Harnss session. Read-only.",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The Harnss session ID.",
                }
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "harnss_send_message",
        "description": (
            "Send a message to a Harnss Agent session. This has an external side effect. "
            "Only call it after the user has explicitly said 确认发送 or confirm send "
            "in the current turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The Harnss session ID.",
                },
                "text": {
                    "type": "string",
                    "description": "The exact message to send to the Agent.",
                },
            },
            "required": ["session_id", "text"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class HarnssToolExecutor:
    client: HarnssClient
    allow_send: bool = False
    last_transcript: str = ""

    def set_transcript(self, transcript: str) -> None:
        self.last_transcript = transcript.strip()

    def execute(self, name: str, arguments: str | dict[str, Any]) -> str:
        try:
            payload = (
                json.loads(arguments)
                if isinstance(arguments, str)
                else arguments
            )
        except json.JSONDecodeError:
            return json.dumps(
                {"ok": False, "error": "Tool arguments were not valid JSON"},
                ensure_ascii=False,
            )
        if not isinstance(payload, dict):
            return json.dumps(
                {"ok": False, "error": "Tool arguments must be an object"},
                ensure_ascii=False,
            )

        try:
            if name == "harnss_list_sessions":
                return json.dumps(
                    {
                        "ok": True,
                        "sessions": self.client.list_sessions(),
                    },
                    ensure_ascii=False,
                )
            if name == "harnss_get_session":
                session_id = self._required_string(payload, "session_id")
                return json.dumps(
                    {
                        "ok": True,
                        "session": self.client.get_session(session_id),
                    },
                    ensure_ascii=False,
                )
            if name == "harnss_send_message":
                return self._send_message(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            return json.dumps(
                {"ok": False, "error": f"Harnss request failed: {exc}"},
                ensure_ascii=False,
            )

        return json.dumps(
            {"ok": False, "error": f"Unknown tool: {name}"},
            ensure_ascii=False,
        )

    def _send_message(self, payload: dict[str, Any]) -> str:
        if not self.allow_send:
            return json.dumps(
                {
                    "ok": False,
                    "error": "Sending is disabled. Start the bridge with --allow-send.",
                },
                ensure_ascii=False,
            )
        if not explicit_confirmation(self.last_transcript):
            return json.dumps(
                {
                    "ok": False,
                    "confirmation_required": True,
                    "error": "Say 确认发送 or confirm send in the current turn.",
                },
                ensure_ascii=False,
            )
        session_id = self._required_string(payload, "session_id")
        text = self._required_string(payload, "text")
        return json.dumps(
            {
                "ok": True,
                "accepted": self.client.send_message(session_id, text),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _required_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
        return value.strip()


def explicit_confirmation(transcript: str) -> bool:
    normalized = " ".join(transcript.lower().split())
    return "确认发送" in normalized or "confirm send" in normalized

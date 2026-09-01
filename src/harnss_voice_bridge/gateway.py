from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any

from .client import HarnssClient
from .realtime import RealtimeVoiceClient


DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 8766


class MobileVoiceGateway:
    """Proxy browser Realtime JSON events to one speech runtime per client.

    The gateway owns the Realtime session configuration.  In particular, a
    browser cannot replace the instructions or tools with a less restrictive
    session.update event.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        harnss: HarnssClient,
        host: str = DEFAULT_GATEWAY_HOST,
        port: int = DEFAULT_GATEWAY_PORT,
        allow_send: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.harnss = harnss
        self.host = host
        self.port = port
        self.allow_send = allow_send

    async def run(self) -> None:
        """Serve until cancelled or interrupted."""
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("websockets is required for gateway mode") from exc

        try:
            server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            )
        except OSError as exc:
            if exc.errno in {48, 98}:
                raise RuntimeError(
                    f"gateway address already in use: {self.host}:{self.port}; "
                    "reuse the existing gateway or choose another port"
                ) from exc
            raise
        async with server:
            print(f"mobile voice gateway listening on {self.host}:{self.port}")
            await asyncio.Future()

    async def handle_client(self, client_socket: Any) -> None:
        """Handle one browser connection and its private upstream connection."""
        voice = RealtimeVoiceClient(
            endpoint=self.endpoint,
            harnss=self.harnss,
            allow_send=self.allow_send,
        )
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - environment dependent
            await self._close(client_socket, code=1011)
            raise RuntimeError("websockets is required for gateway mode") from exc

        try:
            async with websockets.connect(
                self.endpoint,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            ) as upstream_socket:
                await self._send_upstream(
                    upstream_socket,
                    json.dumps(voice._session_update(), ensure_ascii=False),
                    asyncio.Lock(),
                )
                await self._proxy(client_socket, upstream_socket, voice)
        except Exception:
            # Do not disclose endpoint details, credentials, or upstream
            # exceptions to a browser client.
            await self._send_error(client_socket, "speech runtime unavailable")
            await self._close(client_socket, code=1011)
        finally:
            await self._close(client_socket)

    async def _proxy(
        self,
        client_socket: Any,
        upstream_socket: Any,
        voice: RealtimeVoiceClient,
    ) -> None:
        send_lock = asyncio.Lock()
        client_to_upstream = asyncio.create_task(
            self._forward_client_events(
                client_socket,
                upstream_socket,
                voice,
                send_lock,
            )
        )
        upstream_to_client = asyncio.create_task(
            self._forward_upstream_events(
                client_socket,
                upstream_socket,
                voice,
                send_lock,
            )
        )
        done, pending = await asyncio.wait(
            (client_to_upstream, upstream_to_client),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)

    async def _forward_client_events(
        self,
        client_socket: Any,
        upstream_socket: Any,
        voice: RealtimeVoiceClient,
        send_lock: asyncio.Lock,
    ) -> None:
        async for raw in client_socket:
            event = self._decode_json_object(raw)
            if event is None:
                await self._send_error(client_socket, "invalid JSON event")
                continue

            # The browser may request a new session, but safety-critical
            # instructions and tools are always supplied by this bridge.
            if event.get("type") == "session.update":
                payload = json.dumps(voice._session_update(), ensure_ascii=False)
            else:
                payload = raw if isinstance(raw, str) else json.dumps(event)
            await self._send_upstream(upstream_socket, payload, send_lock)

    async def _forward_upstream_events(
        self,
        client_socket: Any,
        upstream_socket: Any,
        voice: RealtimeVoiceClient,
        send_lock: asyncio.Lock,
    ) -> None:
        async for raw in upstream_socket:
            event = self._decode_json_object(raw)
            if event is None:
                await client_socket.send(raw)
                continue
            # Forward every valid or opaque upstream frame, including audio
            # deltas and runtime events, before handling local tool events.
            await client_socket.send(raw)

            event_type = event.get("type")
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = str(event.get("transcript") or "").strip()
                if transcript:
                    voice.tools.set_transcript(transcript)
            elif event_type == "response.function_call_arguments.done":
                await voice._handle_tool_call(upstream_socket, event)
            elif event_type == "response.done":
                await voice._continue_after_tool_calls(upstream_socket, send_lock)

    @staticmethod
    def _decode_json_object(raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if not isinstance(raw, str):
            return None
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return event if isinstance(event, dict) else None

    @staticmethod
    async def _send_upstream(
        upstream_socket: Any,
        payload: str,
        send_lock: asyncio.Lock,
    ) -> None:
        async with send_lock:
            await upstream_socket.send(payload)

    @staticmethod
    async def _send_error(client_socket: Any, message: str) -> None:
        with suppress(Exception):
            await client_socket.send(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": message,
                        },
                    },
                    ensure_ascii=False,
                )
            )

    @staticmethod
    async def _close(socket: Any, *, code: int = 1000) -> None:
        with suppress(Exception):
            await socket.close(code=code)


VoiceGateway = MobileVoiceGateway

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .client import HarnssClient, HarnssRemoteError, load_remote_endpoint, session_line
from .realtime import (
    AudioDevice,
    RealtimeVoiceClient,
    format_audio_device,
    list_audio_devices,
    parse_device,
)
from .gateway import (
    DEFAULT_GATEWAY_HOST,
    DEFAULT_GATEWAY_PORT,
    MobileVoiceGateway,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe Harnss voice bridge probe")
    parser.add_argument("--host", help="override Harnss host")
    parser.add_argument("--port", type=int, help="override Harnss port")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", help="check Harnss Remote Control")
    subparsers.add_parser("list-sessions", help="list available Harnss sessions")
    subparsers.add_parser("list-devices", help="list local audio input/output devices")
    send = subparsers.add_parser("send", help="send text through a Harnss session")
    send.add_argument("session_id")
    send.add_argument("text")
    send.add_argument(
        "--confirm",
        action="store_true",
        help="required safety acknowledgement before sending",
    )
    watch = subparsers.add_parser("watch", help="watch Harnss Remote WebSocket events")
    watch.add_argument(
        "--session-id",
        help="only print events for one session",
    )
    voice = subparsers.add_parser(
        "voice",
        help="connect local microphone and speaker to speech-to-speech",
    )
    voice.add_argument(
        "--endpoint",
        default="ws://127.0.0.1:8765/v1/realtime",
        help="speech-to-speech Realtime WebSocket endpoint",
    )
    voice.add_argument(
        "--allow-send",
        action="store_true",
        help="allow Harnss sends after explicit spoken confirmation",
    )
    voice.add_argument(
        "--input-device",
        type=parse_device,
        help="microphone device index or name (see list-devices)",
    )
    voice.add_argument(
        "--output-device",
        type=parse_device,
        help="speaker device index or name (see list-devices)",
    )
    gateway = subparsers.add_parser(
        "gateway",
        aliases=["mobile-gateway"],
        help="proxy browser JSON audio to speech-to-speech",
    )
    gateway.add_argument(
        "--endpoint",
        default="ws://127.0.0.1:8765/v1/realtime",
        help="speech-to-speech Realtime WebSocket endpoint",
    )
    gateway.add_argument(
        "--listen-host",
        "--gateway-host",
        "--host",
        dest="listen_host",
        default=DEFAULT_GATEWAY_HOST,
        help=f"gateway bind host (default: {DEFAULT_GATEWAY_HOST})",
    )
    gateway.add_argument(
        "--listen-port",
        "--gateway-port",
        "--port",
        dest="listen_port",
        type=int,
        default=DEFAULT_GATEWAY_PORT,
        help=f"gateway bind port (default: {DEFAULT_GATEWAY_PORT})",
    )
    gateway.add_argument(
        "--allow-send",
        action="store_true",
        help="allow Harnss sends after explicit spoken confirmation",
    )
    return parser


async def watch_events(
    *,
    host: str,
    port: int,
    token: str,
    session_id: str | None,
) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise HarnssRemoteError("websockets is not installed") from exc

    uri = f"ws://{host}:{port}/?token={token}"
    async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as socket:
        async for raw in socket:
            try:
                event: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if session_id and event.get("sessionId") != session_id:
                continue
            # Never print token-bearing connection URLs or raw credentials.
            print(json.dumps(event, ensure_ascii=False, sort_keys=True))


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "list-devices":
            for device in list_audio_devices():
                print(format_audio_device(device))
            return 0

        configured_host, configured_port = load_remote_endpoint()
        client = HarnssClient(
            host=args.host or configured_host,
            port=args.port or configured_port,
        )
        if args.command == "health":
            print(client.health())
        elif args.command == "list-sessions":
            for session in client.list_sessions():
                print(session_line(session))
        elif args.command == "send":
            if not args.confirm:
                print(
                    "Refusing to send: add --confirm after applying the voice risk policy.",
                    file=sys.stderr,
                )
                return 2
            print(client.send_message(args.session_id, args.text))
        elif args.command == "watch":
            asyncio.run(
                watch_events(
                    host=args.host or configured_host,
                    port=args.port or configured_port,
                    token=client.token,
                    session_id=args.session_id,
                )
            )
        elif args.command == "voice":
            asyncio.run(
                RealtimeVoiceClient(
                    endpoint=args.endpoint,
                    harnss=client,
                    allow_send=args.allow_send,
                    device=AudioDevice(
                        input_device=args.input_device,
                        output_device=args.output_device,
                    ),
                ).run()
            )
        elif args.command in {"gateway", "mobile-gateway"}:
            asyncio.run(
                MobileVoiceGateway(
                    endpoint=args.endpoint,
                    harnss=client,
                    host=args.listen_host,
                    port=args.listen_port,
                    allow_send=args.allow_send,
                ).run()
            )
        return 0
    except (HarnssRemoteError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print()
            return 0
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

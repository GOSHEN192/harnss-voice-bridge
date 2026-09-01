from __future__ import annotations

import asyncio
import base64
import json
import queue
from dataclasses import dataclass
from typing import Any

from .client import HarnssClient
from .tools import HARNSS_TOOLS, HarnssToolExecutor


REALTIME_INSTRUCTIONS = """你是 Harnss Voice Bridge 的中文语音助手，服务于 Harnss。
连接建立后的主动问候只能说一句：“你好，我在。请说。”不要再追加泛泛的提问。
正常对话先直接回答用户当前的问题，优先使用中文，每次只说一到两句短句；不要重复用户原话，不要重复寒暄，不要解释内部推理或网络细节。
只有用户明确询问 Harnss 会话、Agent 会话或工作安排时，才使用提供的 Harnss 工具；“查看今天的工作安排”也表示列出 Harnss 会话，必须调用 harnss_list_sessions；不要假装已经执行工具，也不要为了寒暄调用工具。
列出或查看会话只能使用只读工具，并根据工具返回的真实结果回答；结果过多时只报最相关的几项。
向 Agent 发送消息前，先用一句话复述目标会话和将发送的原文；只有用户在当前语音轮次明确说“确认发送”或“confirm send”后才调用发送工具。
发送工具返回 confirmation_required 时，只需简短请求用户说“确认发送”。
工具失败时只报告工具返回的失败，不要自行声称已重试或成功；用户说“关闭/停止语音”时只简短确认，不要把它理解为关闭 Agent 会话。
不要执行删除、部署、付款、发布或修改系统设置等高风险操作。
听不清时只说“我没听清，请再说一次。”并最多追问一个问题。"""

STOP_VOICE_PHRASES = (
    "关闭语音",
    "停止语音",
    "退出语音",
    "关闭实时语音",
    "停止实时语音",
    "stop voice",
    "close voice",
)


def is_stop_voice_phrase(transcript: str) -> bool:
    """Recognize explicit local voice shutdown commands."""
    normalized = " ".join(transcript.strip().lower().split())
    return any(phrase in normalized for phrase in STOP_VOICE_PHRASES)


@dataclass
class AudioDevice:
    input_rate: int = 16_000
    output_rate: int = 16_000
    channels: int = 1
    blocksize: int = 512
    input_device: int | str | None = None
    output_device: int | str | None = None


def parse_device(value: str | None) -> int | str | None:
    """Convert numeric CLI device identifiers while preserving device names."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError("audio device must not be empty")
    try:
        return int(value)
    except ValueError:
        return value


def list_audio_devices() -> list[dict[str, Any]]:
    """Return PortAudio devices, with a stable subset safe for display."""
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("sounddevice is required to list audio devices") from exc
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"unable to query audio devices: {exc}") from exc

    result: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        if not isinstance(device, dict):
            continue
        result.append(
            {
                "index": index,
                "name": str(device.get("name") or "Unnamed device"),
                "max_input_channels": int(device.get("max_input_channels", 0)),
                "max_output_channels": int(device.get("max_output_channels", 0)),
            }
        )
    return result


def format_audio_device(device: dict[str, Any]) -> str:
    return (
        f'{device["index"]} | {device["name"]} | '
        f'in={device["max_input_channels"]} out={device["max_output_channels"]}'
    )


class RealtimeVoiceClient:
    """Local microphone/speaker client for speech-to-speech's Realtime API."""

    def __init__(
        self,
        *,
        endpoint: str,
        harnss: HarnssClient,
        allow_send: bool = False,
        device: AudioDevice | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.device = device or AudioDevice()
        self.tools = HarnssToolExecutor(harnss, allow_send=allow_send)
        self._audio_out: queue.Queue[bytes | None] = queue.Queue(maxsize=32)
        self._pending_tool_outputs: list[dict[str, Any]] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("sounddevice is required for voice mode") from exc
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("websockets is required for voice mode") from exc

        input_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        loop = asyncio.get_running_loop()

        def enqueue_input(chunk: bytes) -> None:
            try:
                input_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

        def on_input(indata: Any, _frames: int, _time: Any, status: Any) -> None:
            if status:
                print(f"audio input: {status}")
            loop.call_soon_threadsafe(enqueue_input, bytes(indata))

        output_stream, input_stream = self._open_audio_streams(sd, on_input)

        async with websockets.connect(
            self.endpoint,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as socket:
            await socket.send(json.dumps(self._session_update(), ensure_ascii=False))
            print("voice bridge connected; speak naturally, press Ctrl-C to stop")
            with output_stream, input_stream:
                send_task = asyncio.create_task(self._send_audio(socket, input_queue))
                receive_task = asyncio.create_task(
                    self._receive_events(socket, output_stream)
                )
                done, pending = await asyncio.wait(
                    (send_task, receive_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*done, *pending, return_exceptions=True)

    def _open_audio_streams(self, sd: Any, callback: Any) -> tuple[Any, Any]:
        output_stream = None
        try:
            output_stream = sd.RawOutputStream(
                samplerate=self.device.output_rate,
                channels=self.device.channels,
                dtype="int16",
                blocksize=self.device.blocksize,
                device=self.device.output_device,
            )
            input_stream = sd.RawInputStream(
                samplerate=self.device.input_rate,
                channels=self.device.channels,
                dtype="int16",
                blocksize=self.device.blocksize,
                callback=callback,
                device=self.device.input_device,
            )
        except Exception as exc:
            if output_stream is not None:
                output_stream.close()
            selected = (
                f"input={self.device.input_device!r}, "
                f"output={self.device.output_device!r}"
            )
            raise RuntimeError(
                f"unable to open audio devices ({selected}); "
                "run 'harnss-voice-bridge list-devices' to inspect available devices: "
                f"{exc}"
            ) from exc
        return output_stream, input_stream

    async def _send_audio(self, socket: Any, input_queue: asyncio.Queue[bytes]) -> None:
        while not self._stop.is_set():
            chunk = await input_queue.get()
            await socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )

    async def _receive_events(self, socket: Any, output_stream: Any) -> None:
        async for raw in socket:
            event = json.loads(raw)
            event_type = event.get("type")
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = str(event.get("transcript") or "").strip()
                if transcript:
                    self.tools.set_transcript(transcript)
                    print(f"\n你：{transcript}")
                    if is_stop_voice_phrase(transcript):
                        # Stop locally instead of asking the LLM to interpret a
                        # shutdown request. Closing the websocket releases the
                        # runtime pipeline slot and the audio devices immediately.
                        self._stop.set()
                        try:
                            await socket.send(json.dumps({"type": "response.cancel"}))
                        except Exception:
                            pass
                        return
            elif event_type in {"response.output_audio.delta", "response.audio.delta"}:
                audio = base64.b64decode(event.get("delta", ""))
                if audio:
                    await asyncio.to_thread(output_stream.write, audio)
            elif event_type in {
                "response.output_audio_transcript.done",
                "response.audio_transcript.done",
            }:
                text = str(event.get("transcript") or "").strip()
                if text:
                    print(f"助手：{text}")
            elif event_type == "response.function_call_arguments.done":
                await self._handle_tool_call(socket, event)
            elif event_type == "response.done":
                await self._continue_after_tool_calls(socket)
            elif event_type == "error":
                print(f"voice runtime error: {event.get('error')}")
            elif event_type == "session.updated":
                print("voice session ready")

    async def _handle_tool_call(self, socket: Any, event: dict[str, Any]) -> None:
        name = str(event.get("name") or "")
        call_id = str(event.get("call_id") or "")
        result = await asyncio.to_thread(
            self.tools.execute,
            name,
            str(event.get("arguments") or "{}"),
        )
        self._pending_tool_outputs.append(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                },
            }
        )

    async def _continue_after_tool_calls(
        self,
        socket: Any,
        send_lock: asyncio.Lock | None = None,
    ) -> None:
        if not self._pending_tool_outputs:
            return
        if send_lock is None:
            for event in self._pending_tool_outputs:
                await socket.send(json.dumps(event, ensure_ascii=False))
        else:
            async with send_lock:
                for event in self._pending_tool_outputs:
                    await socket.send(json.dumps(event, ensure_ascii=False))
                await socket.send(json.dumps({"type": "response.create"}))
            self._pending_tool_outputs.clear()
            return
        self._pending_tool_outputs.clear()
        await socket.send(json.dumps({"type": "response.create"}))

    def _session_update(self) -> dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": REALTIME_INSTRUCTIONS,
                "tools": HARNSS_TOOLS,
                "tool_choice": "auto",
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": True,
                            "interrupt_response": True,
                        }
                    }
                },
            },
        }

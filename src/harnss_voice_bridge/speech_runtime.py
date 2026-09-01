from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _patch_local_paraformer() -> None:
    """Allow the upstream handler to use a verified local HF snapshot."""
    from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler

    original_setup = ParaformerSTTHandler.setup

    def setup(
        self: Any,
        model_name: str = "paraformer-zh",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not Path(model_name).expanduser().is_dir():
            original_setup(self, model_name, device, gen_kwargs or {})
            return

        from funasr import AutoModel

        # FunASR's Paraformer CIF decoder currently requests float64, which
        # PyTorch MPS does not support. Keep TTS on MLX/MPS, but run this STT
        # handler on the M4 CPU for correctness.
        actual_device = "cpu" if device == "mps" else device
        self.device = actual_device
        self.model = AutoModel(
            model=str(Path(model_name).expanduser()),
            device=actual_device,
            disable_update=True,
        )
        self.warmup()

    ParaformerSTTHandler.setup = setup


def split_realtime_text(text: str, *, chunk_size: int = 28) -> list[str]:
    """Split Chinese streaming text into speakable chunks before punctuation.

    The upstream sentence splitter waits for punctuation. Chinese voice
    responses often omit punctuation, so that turns a streamed response into
    one large batch and delays TTS until the LLM is completely finished.
    Twenty-eight CJK characters is a compromise between first-audio latency
    and preserving enough context for natural prosody.
    Keep punctuation with its chunk and flush long unfinished Chinese text
    incrementally.
    """
    if not text:
        return []
    chunks: list[str] = []
    current: list[str] = []
    sentence_endings = set("。！？；!?;\n")
    has_cjk = any("\u3400" <= char <= "\u9fff" for char in text)
    if not has_cjk:
        return [text]

    for char in text:
        current.append(char)
        if char in sentence_endings or len(current) >= chunk_size:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    return chunks


def _patch_chinese_streaming() -> None:
    """Make the upstream audio path flush unfinished Chinese text early."""
    from nltk import sent_tokenize as original_sent_tokenize
    from speech_to_speech.LLM import base_openai_compatible_language_model as base_llm

    def tokenize(text: str) -> list[str]:
        try:
            configured_size = int(os.environ.get("LOCAL_VOICE_CHINESE_CHUNK_SIZE", "28"))
        except ValueError:
            configured_size = 28
        chunk_size = max(20, min(64, configured_size))
        chunks = split_realtime_text(text, chunk_size=chunk_size)
        if any("\u3400" <= char <= "\u9fff" for char in text):
            return chunks
        return original_sent_tokenize(text)

    base_llm.sent_tokenize = tokenize


def _patch_realtime_stt_to_text() -> None:
    """Avoid sending the same turn to the LLM as both text and audio.

    RealtimeService currently queues one text request when STT finishes and a
    second audio request when the VAD audio arrives. With Paraformer enabled,
    the text request is sufficient and is much faster for the compatible
    language model backend.
    """
    args = sys.argv[1:]
    stt = None
    for index, value in enumerate(args):
        if value == "--stt" and index + 1 < len(args):
            stt = args[index + 1]
            break
        if value.startswith("--stt="):
            stt = value.split("=", 1)[1]
            break
    if stt != "paraformer":
        return

    from speech_to_speech.api.openai_realtime.service import RealtimeService

    def on_audio_input_completed(self: Any, conn_id: str, event: Any) -> list[Any]:
        state = self._state(conn_id)
        state.input_audio_duration_s = event.audio_duration_s
        state.response_usage.audio_duration_s += event.audio_duration_s
        if event.turn_id is not None:
            state.speculative_audio_duration_s = event.audio_duration_s
            state.speculative_user_turn_id = event.turn_id
            state.speculative_user_turn_revision = event.turn_revision
            state.speculative_user_speech_stopped_at_s = event.speech_stopped_at_s
        return []

    RealtimeService._on_audio_input_completed = on_audio_input_completed


def _patch_no_smart_turn_reopen() -> None:
    """Make --no_smart_turn disable speculative VAD turn reopening too.

    The upstream realtime pipeline always allocates a speculative-turn
    tracker, so its VAD can still reopen a turn even when the Smart Turn
    classifier is disabled. That causes duplicate Paraformer passes on short
    Chinese pauses.
    """
    if "--no_smart_turn" not in sys.argv[1:]:
        return

    from speech_to_speech.VAD.vad_handler import VADHandler

    original_should_reopen = VADHandler._should_reopen_current_turn

    def should_reopen_current_turn(self: Any, audio_start_ms: int) -> bool:
        if getattr(self, "smart_turn_analyzer", None) is None:
            return False
        return original_should_reopen(self, audio_start_ms)

    VADHandler._should_reopen_current_turn = should_reopen_current_turn


def _patch_llm_request_timeout() -> None:
    """Allow local OpenAI-compatible models time to produce first output."""
    value = os.environ.get("LOCAL_VOICE_LLM_REQUEST_TIMEOUT_S", "").strip()
    if not value:
        return
    try:
        timeout_s = max(20.0, float(value))
    except ValueError:
        return

    from speech_to_speech.LLM import base_openai_compatible_language_model as base_llm

    original_setup = base_llm.BaseOpenAICompatibleHandler.setup

    def setup(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("request_timeout_s", timeout_s)
        original_setup(self, *args, **kwargs)

    base_llm.BaseOpenAICompatibleHandler.setup = setup


def main() -> None:
    _patch_local_paraformer()
    _patch_chinese_streaming()
    _patch_realtime_stt_to_text()
    _patch_no_smart_turn_reopen()
    _patch_llm_request_timeout()
    from speech_to_speech.s2s_pipeline import main as speech_to_speech_main

    speech_to_speech_main()


if __name__ == "__main__":
    main()

#!/bin/sh
set -eu

BRIDGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON="$BRIDGE_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Missing bridge virtualenv: $PYTHON" >&2
  exit 1
fi

LLM_BASE_URL="${LOCAL_VOICE_LLM_BASE_URL:-https://proxy.logoshen.com/v1}"
export LOCAL_VOICE_LLM_REQUEST_TIMEOUT_S="${LOCAL_VOICE_LLM_REQUEST_TIMEOUT_S:-90}"

# Keep the Keychain fallback for the default remote OpenAI-compatible endpoint.
# Local Ollama remains available by overriding LOCAL_VOICE_LLM_BASE_URL and
# LOCAL_VOICE_LLM_MODEL.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  case "$LLM_BASE_URL" in
    http://127.0.0.1:11434/*|http://localhost:11434/*)
      OPENAI_API_KEY=ollama
      ;;
    *)
      # Store the value once with:
      # security add-generic-password -a "$USER" -s com.harnss.voice.llm-key -w
      OPENAI_API_KEY=$(security find-generic-password -a "$USER" -s com.harnss.voice.llm-key -w)
      ;;
  esac
fi
export OPENAI_API_KEY

exec "$PYTHON" -m harnss_voice_bridge.speech_runtime \
  --mode realtime \
  --stt mlx-audio-whisper \
  --mlx_audio_whisper_model_name "${MLX_AUDIO_WHISPER_MODEL:-mlx-community/whisper-large-v3-turbo}" \
  --language "${MLX_AUDIO_WHISPER_LANGUAGE:-zh}" \
  --min_silence_ms "${MLX_AUDIO_WHISPER_MIN_SILENCE_MS:-300}" \
  --min_speech_ms "${MLX_AUDIO_WHISPER_MIN_SPEECH_MS:-192}" \
  --no_enable_live_transcription \
  --llm_backend chat-completions \
  --tts qwen3 \
  --qwen3_tts_language "${QWEN3_TTS_LANGUAGE:-chinese}" \
  --qwen3_tts_mlx_quantization "${QWEN3_TTS_QUANTIZATION:-6bit}" \
  --model_name "${LOCAL_VOICE_LLM_MODEL:-gpt-5.6-luna}" \
  --responses_api_base_url "$LLM_BASE_URL" \
  --responses_api_stream \
  --responses_api_disable_thinking \
  --responses_api_audio_max_tokens "${LOCAL_VOICE_LLM_AUDIO_MAX_TOKENS:-128}" \
  --stream_batch_sentences 1 \
  --enable_lang_prompt \
  --speculative_reopen_ms "${LOCAL_VOICE_SPECULATIVE_REOPEN_MS:-400}" \
  --smart_turn_max_wait_ms "${LOCAL_VOICE_SMART_TURN_MAX_WAIT_MS:-500}" \
  --smart_turn_incomplete_delay_ms "${LOCAL_VOICE_SMART_TURN_INCOMPLETE_DELAY_MS:-180}" \
  --qwen3_tts_speaker "${QWEN3_TTS_SPEAKER:-Aiden}" \
  --qwen3_tts_streaming_chunk_size "${QWEN3_TTS_STREAMING_CHUNK_SIZE:-8}" \
  --compact_history \
  --ws_host "${SPEECH_TO_SPEECH_HOST:-127.0.0.1}" \
  --ws_port "${SPEECH_TO_SPEECH_PORT:-8765}"

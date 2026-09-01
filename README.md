# Harnss Voice Bridge

Isolated bridge for the Harnss Remote Control API. The project deliberately
does not modify Harnss or replace its Claude Code, Codex, or ACP engines.

## Current stage

Stage 0E local voice loop is implemented:

- reads the Harnss pairing token from the existing local settings file;
- checks Harnss health;
- lists existing sessions;
- sends text only with an explicit `--confirm`;
- watches Harnss Remote WebSocket events;
- connects a local microphone and speaker to the `speech-to-speech`
  OpenAI Realtime-compatible WebSocket;
- exposes read-only session tools and a guarded send tool to the voice model;
- returns Harnss tool results to the voice model for spoken confirmation;
- never prints the pairing token.

The voice client uses 16 kHz mono PCM input and expects 16 kHz mono PCM output.
The upstream speech runtime remains a separate process and is not embedded into
Harnss.

## Commands

```bash
uv run python -m harnss_voice_bridge health
uv run python -m harnss_voice_bridge list-sessions
uv run python -m harnss_voice_bridge watch --session-id <id>
uv run python -m harnss_voice_bridge list-devices
```

Start the local speech runtime in one terminal. MLX Whisper is the selected
Chinese STT backend for this stage. It runs on Apple Silicon through MLX rather
than CPU-only Paraformer. The default Ollama configuration needs no API key.
For a remote OpenAI-compatible endpoint, put its key in the current shell
without writing it into the project:

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
```

Use the bridge wrapper with the Apple-GPU Whisper model:

```bash
uv run harnss-voice-speech \
  --mode realtime \
  --stt mlx-audio-whisper \
  --mlx_audio_whisper_model_name "mlx-community/whisper-large-v3-turbo" \
  --language zh \
  --min_silence_ms 300 \
  --min_speech_ms 192 \
  --no_enable_live_transcription \
  --llm_backend chat-completions \
  --tts qwen3 \
  --qwen3_tts_language chinese \
  --qwen3_tts_mlx_quantization 6bit \
  --model_name "qwen3.8:27b-mlx" \
  --responses_api_base_url "http://127.0.0.1:11434/v1" \
  --responses_api_stream \
  --responses_api_disable_thinking \
  --responses_api_audio_max_tokens 128 \
  --stream_batch_sentences 1 \
  --enable_lang_prompt \
  --speculative_reopen_ms 400 \
  --smart_turn_max_wait_ms 500 \
  --smart_turn_incomplete_delay_ms 180 \
  --qwen3_tts_speaker Aiden \
  --qwen3_tts_streaming_chunk_size 8 \
  --ws_host 127.0.0.1 \
  --ws_port 8765
```

The Apple Silicon profile uses eight Qwen3-TTS codec tokens per streamed
audio packet (about 640 ms). This reduces MLX scheduling overhead on the
tested M4 Pro while keeping first-audio latency low. Set
`QWEN3_TTS_STREAMING_CHUNK_SIZE=4` if a shorter packet is preferred.

The default language model is the GPT-5.6-compatible endpoint
`gpt-5.6-luna` at `https://proxy.logoshen.com/v1`. The launcher reads its
Bearer token from the macOS Keychain entry `com.harnss.voice.llm-key`.
Override `LOCAL_VOICE_LLM_MODEL` and `LOCAL_VOICE_LLM_BASE_URL` when using
another OpenAI-compatible endpoint, including local Ollama:

```bash
LOCAL_VOICE_LLM_BASE_URL=http://127.0.0.1:11434/v1 \
LOCAL_VOICE_LLM_MODEL=qwen3.8:27b \
./scripts/run-speech-runtime.sh
```

Model responses are allowed up to 90 seconds for first output; override
`LOCAL_VOICE_LLM_REQUEST_TIMEOUT_S` when needed.

Then start the local microphone/speaker bridge in a second terminal:

```bash
uv run python -m harnss_voice_bridge voice
```

For a browser or mobile client, keep the speech runtime running and start the
JSON WebSocket gateway instead:

```bash
uv run python -m harnss_voice_bridge gateway
```

The gateway listens on `127.0.0.1:8766` by default. It accepts the same JSON
audio/event protocol as the upstream Realtime runtime, creates a separate
upstream connection for each browser, forwards audio and runtime events, and
executes function calls through the guarded Harnss tools. Safety instructions
and tools are applied by the gateway and cannot be replaced by a browser
`session.update`. To select another local bind address or port:

```bash
uv run python -m harnss_voice_bridge gateway \
  --listen-host 127.0.0.1 \
  --listen-port 8766 \
  --endpoint ws://127.0.0.1:8765/v1/realtime
```

The aliases `--host`/`--port` and `--gateway-host`/`--gateway-port` are also
accepted by the `gateway` command. Sending remains disabled unless
`--allow-send` is explicitly provided, and still requires the spoken
confirmation phrase.

Use `list-devices` to find PortAudio device indexes, then select a microphone
and speaker by index or exact device name:

```bash
uv run python -m harnss_voice_bridge voice \
  --input-device 2 \
  --output-device "External Headphones"
```

Sending is disabled by default. To enable the Harnss send tool, pass
`--allow-send`; the spoken turn must still contain the exact confirmation
phrase `确认发送` or `confirm send`:

```bash
uv run python -m harnss_voice_bridge voice --allow-send
```

The `send` command is intentionally guarded:

```bash
uv run python -m harnss_voice_bridge send <id> "text" --confirm
```

Before production use, the Voice Bridge must apply the voice risk policy and
route approval-tier work to Harnss's existing approval UI.

## Current safety boundary

The bridge does not create sessions, approve permissions, execute shell
commands, deploy, publish, or expose Harnss outside localhost by default. Both
the gateway and the `speech-to-speech` runtime have no authentication by
default; keep their WebSockets bound to localhost unless an independently
authenticated network transport is in place.

## Optional macOS startup

The upstream speech runtime can be kept warm with the example launchd service.
It reads the external LLM key from macOS Keychain at launch; the key is never
stored in the plist or passed as a command-line argument:

```bash
security add-generic-password -a "$USER" -s com.harnss.voice.llm-key -w
cp launchd/com.harnss.voice-runtime.plist.example \
  "$HOME/Library/LaunchAgents/com.harnss.voice-runtime.plist"
# Replace both /REPLACE/WITH/ABSOLUTE/PATH placeholders before loading.
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.harnss.voice-runtime.plist"
```

Harnss starts and stops the local browser gateway and microphone bridge when
Realtime Voice is enabled and the desktop button is used. The runtime remains
separate so stopping it cannot affect Harnss sessions.

For iPhone access, expose only Harnss Remote Control through an authenticated
Tailscale HTTPS/Serve setup. The browser uses the same-origin `/voice` proxy;
do not expose ports 8765 or 8766 directly because those local speech sockets
do not provide standalone authentication.
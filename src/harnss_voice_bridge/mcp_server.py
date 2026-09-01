"""Loopback MCP server that lets the QWEN floating-ball voice drive Harnss.

Exposes a small set of Harnss Remote API operations as MCP tools
(``mcp__harnss__*``). The pairing token never leaves this process: tools
receive session ids and plain answers, never credentials.

Safety model:
- Read-only tools (list/get) return facts only; wording is left to the voice
  model.
- Writable tools rely on the QWEN frontend MCP platform gate (natural-language
  confirmation per operation) as the primary gate.
- ``allowForSession`` is restricted to a whitelist of read-only agent tools
  unless ``HARNSS_VOICE_SESSION_SCOPE_APPROVAL=1`` is set, so one spoken
  approval can never silence future prompts for destructive tools.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from harnss_voice_bridge.client import (
    SESSION_ENGINES,
    HarnssClient,
    HarnssRemoteError,
    load_remote_endpoint,
)

MAX_RESULT_BYTES = 32 * 1024
MAX_MESSAGE_CHARS = 20_000
MAX_CONTEXT_MESSAGES = 20
TRUNCATION_MARKER = "…[truncated]"

#: Agent tools for which a single voice approval may be extended to the whole
#: session. Anything not listed here is treated as writable (fail-closed).
READ_ONLY_PERMISSION_TOOLS = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "LS",
        "WebFetch",
        "WebSearch",
        "TodoRead",
        "TodoWrite",
        "NotebookRead",
        "TaskOutput",
    }
)

SESSION_SCOPE_OPT_OUT_ENV = "HARNSS_VOICE_SESSION_SCOPE_APPROVAL"

SUMMARY_FIELDS = (
    "id",
    "projectId",
    "projectName",
    "title",
    "engine",
    "agentId",
    "model",
    "effort",
    "permissionMode",
    "planMode",
    "acpPermissionBehavior",
    "isProcessing",
    "hasPendingPermission",
    "lastMessageAt",
)

MESSAGE_FIELDS = (
    "role",
    "content",
    "displayContent",
    "toolName",
    "toolSummary",
    "isError",
    "timestamp",
)


def truncate_json(result: Any, limit: int = MAX_RESULT_BYTES) -> str:
    """Serialize a tool result and hard-cap the payload size."""
    text = json.dumps(result, ensure_ascii=False)
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    cut = text[: max(limit // 2, 0)]
    while len(cut.encode("utf-8")) > limit - len(TRUNCATION_MARKER) and cut:
        cut = cut[:-1]
    return cut + TRUNCATION_MARKER


class HarnssMcpService:
    """Plain-Python tool implementations, testable without a transport."""

    def __init__(self, client: HarnssClient) -> None:
        self._client = client

    # -- read-only ---------------------------------------------------------

    def list_sessions(self) -> dict[str, Any]:
        sessions = [
            {field: session.get(field) for field in SUMMARY_FIELDS}
            for session in self._client.list_sessions()
        ]
        sessions.sort(
            key=lambda session: session.get("lastMessageAt") or 0,
            reverse=True,
        )
        return {"ok": True, "sessions": sessions}

    def get_session(self, session_id: str) -> dict[str, Any]:
        snapshot = self._client.get_session(session_id)
        permission = snapshot.get("pendingPermission")
        safe_permission: dict[str, Any] | None = None
        if isinstance(permission, dict):
            safe_permission = {
                "toolName": permission.get("toolName"),
                "kind": permission.get("kind"),
                "decisionReason": permission.get("decisionReason"),
                "summary": permission.get("summary"),
                "questions": permission.get("questions") or [],
            }
        messages = [
            {field: message.get(field) for field in MESSAGE_FIELDS}
            for message in (snapshot.get("messages") or [])[-MAX_CONTEXT_MESSAGES:]
        ]
        return {
            "ok": True,
            "session": {
                field: snapshot.get("session", {}).get(field)
                for field in SUMMARY_FIELDS
            },
            "isProcessing": snapshot.get("isProcessing"),
            "isConnected": snapshot.get("isConnected"),
            "totalCost": snapshot.get("totalCost"),
            "contextUsage": snapshot.get("contextUsage"),
            "pendingPermission": safe_permission,
            "truncated": snapshot.get("truncated"),
            "recentMessages": messages,
        }

    # -- writable ----------------------------------------------------------

    def send_message(self, session_id: str, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if len(text) > MAX_MESSAGE_CHARS:
            raise ValueError("text must be at most 20000 characters")
        data = self._client.send_message(session_id, text)
        return {"ok": True, "result": data}

    def respond_approval(
        self,
        session_id: str,
        behavior: str,
        answers_by_question_id: dict[str, list[str]] | None = None,
        deny_message: str | None = None,
    ) -> dict[str, Any]:
        if behavior not in {"allow", "allowForSession", "deny"}:
            raise ValueError("behavior must be one of: allow, allowForSession, deny")

        snapshot = self._client.get_session(session_id)
        permission = snapshot.get("pendingPermission")
        if not isinstance(permission, dict) or not permission.get("requestId"):
            return {"ok": False, "error": "This session has no pending permission to answer."}

        if behavior == "allowForSession":
            blocked = self._session_scope_block_reason(permission)
            if blocked:
                return {"ok": False, "error": blocked}

        data = self._client.respond_approval(
            session_id,
            str(permission["requestId"]),
            behavior,
            answers_by_question_id=answers_by_question_id or None,
            deny_message=deny_message,
        )
        return {"ok": True, "result": data}

    def interrupt(self, session_id: str) -> dict[str, Any]:
        data = self._client.session_action(session_id, "interrupt")
        return {"ok": True, "result": data}

    def create_session(
        self,
        project_id: str,
        engine: str = "claude",
        model: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        if engine not in SESSION_ENGINES:
            raise ValueError("engine must be one of: claude, codex, acp")
        data = self._client.create_session(
            project_id,
            engine,
            model=model,
            agent_id=agent_id,
        )
        return {
            "ok": True,
            "result": data,
            "hint": "Creation is asynchronous; call list_sessions to find the new session.",
        }

    def set_session_settings(
        self,
        session_id: str,
        model: str | None = None,
        effort: str | None = None,
        permission_mode: str | None = None,
        plan_mode: bool | None = None,
        acp_permission_behavior: str | None = None,
    ) -> dict[str, Any]:
        requested: dict[str, Any] = {
            "model": model,
            "effort": effort,
            "permission_mode": permission_mode,
            "plan_mode": plan_mode,
            "acp_permission_behavior": acp_permission_behavior,
        }
        payload = {
            key: value
            for key, value in requested.items()
            if value is not None
        }
        if not payload:
            raise ValueError("at least one setting must be provided")
        data = self._client.update_settings(session_id, **payload)
        return {"ok": True, "result": data}

    # -- helpers -----------------------------------------------------------

    def _session_scope_block_reason(self, permission: dict[str, Any]) -> str | None:
        """Return a refusal reason when allowForSession is too broad here."""
        if os.environ.get(SESSION_SCOPE_OPT_OUT_ENV, "").strip() == "1":
            return None
        if permission.get("kind") == "user-input":
            return (
                "allowForSession only applies to approval requests. "
                "Use behavior 'allow' for user-input questions."
            )
        tool_name = str(permission.get("toolName") or "")
        if tool_name not in READ_ONLY_PERMISSION_TOOLS:
            return (
                f"allowForSession is restricted to read-only tools for safety; "
                f"'{tool_name or 'unknown tool'}' is not on the whitelist. "
                f"Use behavior 'allow' for this single operation, or set "
                f"{SESSION_SCOPE_OPT_OUT_ENV}=1 to lift the restriction."
            )
        return None


def _tool(func: Callable[..., dict[str, Any]]) -> Callable[..., str]:
    """Wrap a service method: serialize, cap, and surface errors as content."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return truncate_json(func(*args, **kwargs))
        except (HarnssRemoteError, ValueError) as exc:
            return truncate_json({"ok": False, "error": str(exc)})

    return wrapper


TOOL_DESCRIPTIONS = {
    "list_sessions": (
        "List Harnss desktop agent sessions (Claude Code / Codex), newest first: "
        "engine, model, running state, pending approvals. 列出 Harnss 中的引擎会话"
        "（含引擎、模型、是否运行中、是否有待审批）。"
    ),
    "get_session": (
        "Get one Harnss session's recent messages, cost, context usage and any "
        "pending permission (with question ids for user-input kinds). "
        "查看某个会话的最近消息、成本、上下文用量与待审批详情。"
    ),
    "send_message": (
        "Send an instruction to a Harnss agent session. Requires explicit user "
        "confirmation before calling. 向会话发送指令；调用前必须获得用户明确确认。"
    ),
    "respond_approval": (
        "Answer a Harnss session's pending permission: behavior allow / deny / "
        "allowForSession (the last one only for read-only tools). Provide "
        "answers_by_question_id for user-input kinds. Requires user confirmation. "
        "答复待审批：allow / deny / allowForSession（仅限只读工具），需用户确认。"
    ),
    "interrupt": (
        "Interrupt (stop) a running Harnss agent session. Requires user "
        "confirmation. 打断正在运行的会话，需用户确认。"
    ),
    "create_session": (
        "Create a new Harnss session in a project: engine claude / codex / acp, "
        "optional model and agent_id. project_id comes from list_sessions. "
        "Requires user confirmation. 在指定项目新建会话，需用户确认。"
    ),
    "set_session_settings": (
        "Update a Harnss session's model, effort, permission_mode, plan_mode or "
        "acp_permission_behavior. Requires user confirmation. "
        "修改会话的模型/权限/计划模式等设置，需用户确认。"
    ),
}


def build_server(
    service: HarnssMcpService,
    *,
    host: str = "127.0.0.1",
    port: int = 8790,
) -> FastMCP:
    server: FastMCP = FastMCP(
        "harnss",
        instructions=(
            "Tools for operating the Harnss desktop app's engine agents "
            "(Claude Code / Codex) over its loopback Remote API. "
            "Read tools return facts; phrase the spoken answer yourself. "
            "Every writable tool requires explicit user confirmation first. "
            "Harnss 桌面引擎会话控制工具；只读工具直接回答，写操作必须先口头确认。"
        ),
        host=host,
        port=port,
    )

    server.tool(
        description=TOOL_DESCRIPTIONS["list_sessions"],
        name="list_sessions",
    )(_tool(service.list_sessions))
    server.tool(
        description=TOOL_DESCRIPTIONS["get_session"],
        name="get_session",
    )(_tool(service.get_session))
    server.tool(
        description=TOOL_DESCRIPTIONS["send_message"],
        name="send_message",
    )(_tool(service.send_message))
    server.tool(
        description=TOOL_DESCRIPTIONS["respond_approval"],
        name="respond_approval",
    )(_tool(service.respond_approval))
    server.tool(description=TOOL_DESCRIPTIONS["interrupt"], name="interrupt")(
        _tool(service.interrupt)
    )
    server.tool(
        description=TOOL_DESCRIPTIONS["create_session"],
        name="create_session",
    )(_tool(service.create_session))
    server.tool(
        description=TOOL_DESCRIPTIONS["set_session_settings"],
        name="set_session_settings",
    )(_tool(service.set_session_settings))
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Harnss voice MCP server (loopback only)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="MCP bind host")
    parser.add_argument("--port", type=int, default=8790, help="MCP bind port")
    parser.add_argument("--harnss-host", default=None, help="Override Harnss host")
    parser.add_argument("--harnss-port", type=int, default=None, help="Override Harnss port")
    args = parser.parse_args(argv)

    harnss_host, harnss_port = load_remote_endpoint()
    client = HarnssClient(
        host=args.harnss_host or harnss_host,
        port=args.harnss_port or harnss_port,
    )
    server = build_server(HarnssMcpService(client), host=args.host, port=args.port)
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()

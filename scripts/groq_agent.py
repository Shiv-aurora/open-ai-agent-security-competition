"""Groq Chat Completions agent for local GPT-OSS probing.

Uses openai/gpt-oss-20b via https://api.groq.com/openai/v1.
This is a fire-rate proxy, not the Kaggle GGUF Harmony runtime.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from aicomp_sdk.agents.tool_specs import build_openai_tool_name_maps, to_hf_function_tool
from aicomp_sdk.agents.types import (
    AGENT_STATE_VERSION,
    FinalResponseDecision,
    ToolCall,
    ToolCallDecision,
    ToolRequestEvent,
    ToolResultEvent,
    UserMessageEvent,
    AssistantMessageEvent,
    AgentStateSnapshot,
)
from aicomp_sdk.core.runtime_history import RuntimeHistory

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_GPT_OSS_MODEL = "openai/gpt-oss-20b"


def require_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError(
        "GROQ_API_KEY is not set. Export it in this shell, e.g.\n"
        "  export GROQ_API_KEY='gsk_...'\n"
        "Get a key at https://console.groq.com/keys"
    )


class GroqChatAgent:
    """OpenAI-compatible Chat Completions tool loop pointed at Groq."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = GROQ_GPT_OSS_MODEL,
        reasoning_effort: str = "low",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as err:
            raise RuntimeError("pip install openai") from err
        self._client = OpenAI(api_key=api_key or require_groq_key(), base_url=GROQ_BASE_URL)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._n = 0

    def reset_state(self) -> None:
        self._n = 0

    def snapshot_state(self) -> AgentStateSnapshot:
        return {
            "version": AGENT_STATE_VERSION,
            "backend": "groq_chat",
            "data": {"n": self._n},
        }

    def restore_state(self, snapshot: AgentStateSnapshot) -> None:
        self._n = int((snapshot.get("data") or {}).get("n", 0))

    def next_action(self, *, history: RuntimeHistory, tools: Sequence[Any]):
        canonical_to_openai, openai_to_canonical = build_openai_tool_name_maps(tools)
        messages = _history_to_chat(history, canonical_to_openai)
        payload_tools = [to_hf_function_tool(spec) for spec in tools]
        for spec, tool in zip(tools, payload_tools):
            alias = canonical_to_openai.get(spec.name, spec.name)
            tool["function"]["name"] = alias

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": payload_tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 512,
        }
        if self._reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self._reasoning_effort}

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            call = message.tool_calls[0]
            name = openai_to_canonical.get(call.function.name, call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            self._n += 1
            return ToolCallDecision(
                call=ToolCall(
                    call_id=call.id or f"groq_{self._n:04d}",
                    tool_name=name,
                    arguments=args,
                )
            )

        text = (message.content or "").strip()
        harmony = _try_harmony_tool_call(text, openai_to_canonical)
        if harmony is not None:
            self._n += 1
            return harmony
        return FinalResponseDecision(text=text)


def _history_to_chat(
    history: RuntimeHistory,
    canonical_to_openai: dict[str, str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = "\n\n".join(ev.text for ev in history.instructions if ev.text.strip())
    if instructions:
        messages.append({"role": "system", "content": instructions})

    pending_assistant_tools: list[dict[str, Any]] = []
    for event in history.events:
        if isinstance(event, UserMessageEvent):
            _flush_assistant_tools(messages, pending_assistant_tools)
            messages.append({"role": "user", "content": event.text})
        elif isinstance(event, AssistantMessageEvent):
            _flush_assistant_tools(messages, pending_assistant_tools)
            messages.append({"role": "assistant", "content": event.text})
        elif isinstance(event, ToolRequestEvent):
            pending_assistant_tools.append(
                {
                    "id": event.call.call_id,
                    "type": "function",
                    "function": {
                        "name": canonical_to_openai.get(
                            event.call.tool_name, event.call.tool_name
                        ),
                        "arguments": json.dumps(dict(event.call.arguments), sort_keys=True),
                    },
                }
            )
        elif isinstance(event, ToolResultEvent):
            _flush_assistant_tools(messages, pending_assistant_tools)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event.result.call_id,
                    "content": event.result.output_text,
                }
            )
    _flush_assistant_tools(messages, pending_assistant_tools)
    return messages


def _flush_assistant_tools(
    messages: list[dict[str, Any]], pending: list[dict[str, Any]]
) -> None:
    if not pending:
        return
    messages.append({"role": "assistant", "content": None, "tool_calls": list(pending)})
    pending.clear()


def _try_harmony_tool_call(text: str, openai_to_canonical: dict[str, str]) -> ToolCallDecision | None:
    if not text:
        return None
    try:
        from aicomp_sdk.agents.gpt_oss_agent import _parse_gpt_oss_harmony_response
    except Exception:
        return None
    parsed = _parse_gpt_oss_harmony_response(text)
    if not parsed or not parsed.get("tool_calls"):
        return None
    first = parsed["tool_calls"][0]["function"]
    name = openai_to_canonical.get(first["name"], first["name"])
    args = first.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    return ToolCallDecision(
        call=ToolCall(call_id="groq_harmony_1", tool_name=name, arguments=args)
    )

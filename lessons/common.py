"""Shared helpers for every lesson.

Key idea: lessons run OFFLINE by default using a scripted fake chat model, so you can learn how the
LangChain pieces fit into LangGraph without spending money or needing an API key.
Put a key in `.env` (see .env.example) and the same code uses a real LLM.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from typing import Any, override

from dotenv import load_dotenv
from langchain.messages import AIMessage, AIMessageChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk

load_dotenv()

# Newest Sonnet-class model as of Nov 2026 (https://platform.claude.com/docs/en/models/overview).
# This is the ONLY place the default model id lives. Override with LC_MODEL (any init_chat_model
# string, e.g. "anthropic:claude-opus-5-5" or "anthropic:claude-haiku-5-5").
DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"


def have_real_llm() -> bool:
    """True only when we can actually build the configured model.

    * ANTHROPIC_API_KEY alone -> the default Anthropic model works.
    * Any other provider key requires an explicit LC_MODEL ("openai:<id>", ...), because the
      default model is an Anthropic one and would fail with a confusing auth error.
    """
    if bool(os.getenv("ANTHROPIC_API_KEY")):
        return True
    return bool(os.getenv("LC_MODEL") and (os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")))


def model_name() -> str:
    return os.getenv("LC_MODEL") or DEFAULT_MODEL


def make_real_chat_model(**kwargs: Any) -> Any:
    """`init_chat_model` for the configured model (lesson 01 explains it).

    Sampling params are left at provider defaults (newer models restrict some of them);
    set LC_TEMPERATURE to opt in.
    """
    from langchain.chat_models import init_chat_model

    if bool(os.getenv("LC_TEMPERATURE")):
        kwargs.setdefault("temperature", float(os.environ["LC_TEMPERATURE"]))
    return init_chat_model(model_name(), **kwargs)


class ScriptedChatModel(GenericFakeChatModel):
    """A fake chat model that replays a fixed list of AIMessages, one per model call.

    * `bind_tools` is a no-op so it can stand in for a tool-calling model.
    * It streams word by word, and streams tool calls as `tool_call_chunks` (the stock fake model
      streams nothing for a message that has only tool calls), so lesson 11 works offline.
    * It is an iterator underneath, so it can be consumed ONCE. Build a fresh one per agent.
    """

    @override
    def bind_tools(self, tools: Any, *args: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs).generations[0].message
        assert isinstance(message, AIMessage)
        pieces = [AIMessageChunk(content=tok, id=message.id) for tok in re.split(r"(\s)", message.text) if tok]
        pieces += [
            AIMessageChunk(
                content="",
                id=message.id,
                tool_call_chunks=[
                    {"name": call["name"], "args": json.dumps(call["args"]), "id": call["id"], "index": i}
                ],
            )
            for i, call in enumerate(message.tool_calls)
        ]
        pieces.append(AIMessageChunk(content="", id=message.id, chunk_position="last"))
        for piece in pieces:
            chunk = ChatGenerationChunk(message=piece)
            if run_manager and piece.content:
                run_manager.on_llm_new_token(str(piece.content), chunk=chunk)
            yield chunk


def tool_call_msg(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    """Build an AIMessage that *asks* for a tool call (what a real LLM would emit)."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def get_model(script: list[AIMessage] | None = None) -> Any:
    """Return a real chat model if a key is configured, else a scripted fake.

    `script` is only used offline: the sequence of AI replies the fake will give.
    """
    if have_real_llm():
        return make_real_chat_model()
    script = script or [AIMessage(content="(offline fake model) Hello from the fake LLM!")]
    return ScriptedChatModel(messages=iter(script))


def banner(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def show(messages: Iterable[BaseMessage]) -> None:
    """One line per message: type, text, and any tool calls. Used to print traces."""
    for m in messages:
        calls = getattr(m, "tool_calls", None)
        extra = f"  tool_calls={[(c['name'], c['args']) for c in calls]}" if calls else ""
        print(f"  {type(m).__name__:<13} {m.text!r}{extra}")

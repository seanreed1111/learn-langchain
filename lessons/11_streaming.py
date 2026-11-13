"""LESSON 11 - Streaming an agent
===============================
Agents are slow (several model calls and tools), so real products stream. Because `create_agent`
returns a LangGraph graph, you use LangGraph's `stream_mode` (learn-langgraph lesson 09), and the
LangChain layer adds a few conveniences. The modes you will use:

  stream_mode="updates"   one event per graph STEP: {"model": {...}}, {"tools": {...}}  - progress UI
  stream_mode="messages"  (token, metadata) pairs from any LLM call inside the graph      - typing effect
  stream_mode="custom"    whatever a tool writes with `runtime.stream_writer(...)`        - tool progress
  stream_mode="values"    the FULL state after every step                                  - simple clients
  stream_mode=[...]       several at once; every chunk says which mode it came from

Use `version="v2"` (needs langgraph >= 1.1): every chunk is a dict `{"type": mode, "ns": ..., "data": ...}`
so a list of modes is easy to dispatch on. Without it, a multi-mode chunk is a `(mode, data)` tuple.

For new code the LangChain docs recommend EVENT STREAMING, `agent.stream_events(..., version="v3")`:
typed projections (`.messages`, `.tool_calls`, `.values`, `.interleave(...)`) instead of raw chunk
dicts. In langgraph 1.2.21 it is still flagged experimental (it emits a LangChainBetaWarning), so
treat the API as moving.

Streaming tokens needs a model that streams; the scripted offline model streams word by word.

Run:  python lessons/11_streaming.py
"""

import warnings
from collections.abc import Iterator
from typing import Any, cast

from common import banner, get_model, tool_call_msg
from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool

INPUT = {"messages": "What's the weather in Paris?"}


@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get the weather for a city."""
    # Emitted on the "custom" stream. This only works inside a graph run (not when you call the tool directly).
    runtime.stream_writer(f"looking up {city}...")
    return f"It is sunny in {city}."


def weather_agent():
    script = [tool_call_msg("get_weather", {"city": "Paris"}), AIMessage(content="It is sunny in Paris.")]
    return create_agent(model=get_model(script), tools=[get_weather], name="weather_bot")


def parts(stream_mode: str | list[str]) -> Iterator[dict[str, Any]]:
    """`stream(..., version="v2")` yields StreamPart dicts: {"type", "ns", "data"}. Typed loosely here so
    the lessons below read simply."""
    agent: Any = weather_agent()  # the stream() overloads are picky about a str | list mode
    return cast("Iterator[dict[str, Any]]", agent.stream(INPUT, stream_mode=stream_mode, version="v2"))


def updates() -> list[tuple[str, list[str]]]:
    """(node, [message types written]) per step."""
    out: list[tuple[str, list[str]]] = []
    for chunk in parts("updates"):
        for node, update in chunk["data"].items():
            out.append((node, [type(m).__name__ for m in update["messages"]]))
    return out


def tokens() -> str:
    """The final answer, rebuilt from streamed tokens of the `model` node only."""
    text = ""
    for chunk in parts("messages"):
        token, metadata = chunk["data"]
        if metadata["langgraph_node"] == "model" and token.text:
            text += str(token.text)
    return text


def custom_events() -> list[Any]:
    return [c["data"] for c in parts("custom")]


def mixed() -> list[str]:
    return [str(c["type"]) for c in parts(["updates", "custom"])]


def values_sizes() -> list[int]:
    """With stream_mode='values' each chunk is the whole state so far (messages grow each step)."""
    return [len(c["messages"]) for c in weather_agent().stream(INPUT, stream_mode="values")]


def event_stream() -> list[str]:
    """v3 event streaming: typed projections instead of raw chunks."""
    lines: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # LangChainBetaWarning: v3 is experimental in langgraph 1.2.21
        stream = weather_agent().stream_events(INPUT, version="v3")
        for kind, item in stream.interleave("messages", "tool_calls"):
            if kind == "messages":
                text = "".join(item.text)
                if text:
                    lines.append(f"message: {text}")
            else:
                lines.append(f"tool call: {item.tool_name}({item.input})")
    return lines


if __name__ == "__main__":
    banner("updates: one event per step")
    for node, kinds in updates():
        print(f"  {node:<6} wrote {kinds}")

    banner("messages: tokens (filter on metadata['langgraph_node'])")
    print(" ", tokens())

    banner("custom: what a tool chooses to emit")
    print(" ", custom_events())

    banner("several modes at once (v2 chunks carry their type)")
    print(" ", mixed())

    banner("values: full state per step (message count)")
    print(" ", values_sizes())

    banner("stream_events v3 (experimental): typed projections")
    for line in event_stream():
        print(" ", line)

# EXERCISES
# 1. Use `stream_mode="updates"` WITHOUT `version="v2"` and print a chunk. What shape is different?
# 2. Stream with `stream_mode=["messages", "custom"]` and print each chunk's type, in order. Where does
#    the tool's custom event land relative to the tool call's message?
# 3. Put an ANTHROPIC_API_KEY in `.env` and re-run `tokens()`. Compare the token granularity to the
#    scripted model's words.

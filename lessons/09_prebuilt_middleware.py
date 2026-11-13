"""LESSON 09 - Prebuilt middleware: the reliability toolbox
========================================================
Lesson 12 of learn-langgraph builds retries, budgets and error handling by hand. For agents made with
`create_agent`, most of that is a one-line middleware (all from `langchain.agents.middleware`):

  PROBLEM                                MIDDLEWARE
  -------------------------------------  --------------------------------------------------------
  A loop that never ends                 ModelCallLimitMiddleware(run_limit=..., thread_limit=...)
  One tool called too often              ToolCallLimitMiddleware(tool_name=..., run_limit=...)
  Flaky tool (network)                   ToolRetryMiddleware(max_retries=..., retry_on=...)
  Tool raises (crashes the run)          ToolErrorMiddleware(on_error=...)  (or wrap_tool_call, lesson 08)
  Flaky or overloaded model              ModelRetryMiddleware(...)
  Provider outage                        ModelFallbackMiddleware(backup_model, ...)
  History outgrows the context window    SummarizationMiddleware(model=..., trigger=..., keep=...)
  Sensitive data in prompts              PIIMiddleware("email", strategy="redact")
  Human sign-off for risky tools         HumanInTheLoopMiddleware        (lesson 10)

Each one is a node-style or wrap-style hook (lesson 08), so it composes with your own hooks and obeys
the same ORDER rules (first in the list = outermost). Order matters, and so does `on_failure`:

  * To "retry the primary, THEN fall back": put the fallback FIRST (outer) and the retry LATER (inner)
    AND give the retry `on_failure="error"`. The default `on_failure="continue"` turns the final failure
    into an AI message ("Model call failed after 3 attempts..."), so an outer fallback never sees an
    exception and never fires. (Section 3 shows all three outcomes.)
  * Put limits early so they see every call.

Run:  python lessons/09_prebuilt_middleware.py
"""

import itertools
from typing import Any, override

from common import ScriptedChatModel, banner, get_model, tool_call_msg
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import InMemorySaver

ATTEMPTS = {"tool": 0, "model": 0}


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def flaky_api(query: str) -> str:
    """Fetch data from an unreliable upstream (fails twice, then works)."""
    ATTEMPTS["tool"] += 1
    if ATTEMPTS["tool"] <= 2:
        raise ConnectionError("upstream timed out")
    return f"data for {query!r}"


@tool
def always_fails(query: str) -> str:
    """A tool that always raises ValueError."""
    raise ValueError(f"cannot handle {query!r}")


class DownModel(ScriptedChatModel):
    """A model whose provider is down."""

    @override
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise ConnectionError("provider outage")


class FlakyModel(ScriptedChatModel):
    """Fails on the first call, then plays its script."""

    @override
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        ATTEMPTS["model"] += 1
        if ATTEMPTS["model"] == 1:
            raise ConnectionError("rate limited")
        return super()._generate(*args, **kwargs)


def last_text(result: dict[str, Any]) -> str:
    return str(result["messages"][-1].text)


# ---- 1) Limits ---------------------------------------------------------------------------------------
def model_limit_agent():
    endless = (tool_call_msg("add", {"a": i, "b": 1}, f"c{i}") for i in itertools.count())
    return create_agent(
        model=ScriptedChatModel(messages=endless),
        tools=[add],
        middleware=[ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")],
    )


def tool_limit_agent():
    script = [
        tool_call_msg("add", {"a": 1, "b": 1}, "c1"),
        tool_call_msg("add", {"a": 2, "b": 2}, "c2"),
        AIMessage(content="I could only add once."),
    ]
    return create_agent(
        model=get_model(script),
        tools=[add],
        middleware=[ToolCallLimitMiddleware(tool_name="add", run_limit=1)],  # default: block, keep going
    )


# ---- 2) Tool failures --------------------------------------------------------------------------------
def tool_retry_agent():
    ATTEMPTS["tool"] = 0
    script = [tool_call_msg("flaky_api", {"query": "x"}), AIMessage(content="Got the data.")]
    return create_agent(
        model=get_model(script),
        tools=[flaky_api],
        middleware=[ToolRetryMiddleware(max_retries=2, retry_on=(ConnectionError,), initial_delay=0, jitter=False)],
    )


def tool_error_agent():
    script = [tool_call_msg("always_fails", {"query": "x"}), AIMessage(content="That tool can't do it.")]
    return create_agent(
        model=get_model(script),
        tools=[always_fails],
        middleware=[ToolErrorMiddleware(on_error=lambda exc, req: f"{type(exc).__name__}: {exc}")],
    )


# ---- 3) Model failures -------------------------------------------------------------------------------
def model_retry_agent():
    ATTEMPTS["model"] = 0
    return create_agent(
        model=FlakyModel(messages=iter([AIMessage(content="Recovered after one retry.")])),
        tools=[],
        middleware=[ModelRetryMiddleware(max_retries=2, initial_delay=0, jitter=False)],
    )


class CountingDownModel(DownModel):
    """A down provider that counts how many times it was called."""

    @override
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        ATTEMPTS["model"] += 1
        raise ConnectionError("provider outage")


def retry_then_fallback_agent(*, on_failure: Any = "error", fallback_outer: bool = True):
    """Retry the primary twice, then switch to a backup. Returns an agent; `ATTEMPTS["model"]` counts
    primary calls. The two knobs are exactly what decides whether the fallback ever fires."""
    ATTEMPTS["model"] = 0
    backup = ScriptedChatModel(messages=iter([AIMessage(content="Answered by the backup model.")]))
    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0, jitter=False, on_failure=on_failure)
    fallback = ModelFallbackMiddleware(backup)
    return create_agent(
        model=CountingDownModel(messages=iter([])),
        tools=[],
        middleware=[fallback, retry] if fallback_outer else [retry, fallback],
    )


def fallback_agent():
    backup = ScriptedChatModel(messages=iter([AIMessage(content="Answered by the backup model.")]))
    return create_agent(
        model=DownModel(messages=iter([])), tools=[], middleware=[ModelFallbackMiddleware(backup)]
    )


# ---- 4) Context and privacy --------------------------------------------------------------------------
def pii_agent():
    return create_agent(
        model=get_model([AIMessage(content="Noted.")]),
        tools=[],
        middleware=[PIIMiddleware("email", strategy="redact", apply_to_input=True)],
    )


def summarizing_agent():
    main = [AIMessage(content=f"reply {i}") for i in range(1, 5)]
    summarizer = ScriptedChatModel(messages=iter([AIMessage(content="SUMMARY: the user is Bob and likes graphs.")]))
    return create_agent(
        model=get_model(main),
        tools=[],
        middleware=[SummarizationMiddleware(model=summarizer, trigger=("messages", 6), keep=("messages", 2))],
        checkpointer=InMemorySaver(),
    )


if __name__ == "__main__":
    banner("Limits")
    out = model_limit_agent().invoke({"messages": "go"})
    real_calls = [m for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    print("ModelCallLimit(run_limit=3):", len(real_calls), "real model calls, then ->", last_text(out))
    out = tool_limit_agent().invoke({"messages": "add twice"})
    print("ToolCallLimit blocked 2nd call:", [m.text for m in out["messages"] if m.type == "tool"], "->", last_text(out))

    banner("Tool failures")
    out = tool_retry_agent().invoke({"messages": "fetch"})
    print(f"ToolRetry: tool ran {ATTEMPTS['tool']} times ->", [m.text for m in out["messages"] if m.type == "tool"])
    out = tool_error_agent().invoke({"messages": "go"})
    print("ToolError:", [m.text for m in out["messages"] if m.type == "tool"], "->", last_text(out))

    banner("Model failures")
    out = model_retry_agent().invoke({"messages": "hi"})
    print(f"ModelRetry: model attempted {ATTEMPTS['model']} times ->", last_text(out))
    print("ModelFallback ->", last_text(fallback_agent().invoke({"messages": "hi"})))
    cases: list[tuple[str, dict[str, Any]]] = [
        ("fallback outer + retry on_failure='error' ", {}),
        ("fallback outer + retry on_failure='continue'", {"on_failure": "continue"}),
        ("retry outer + fallback inner               ", {"fallback_outer": False}),
    ]
    for label, kwargs in cases:
        result = retry_then_fallback_agent(**kwargs).invoke({"messages": "hi"})
        print(f"  {label}: primary called {ATTEMPTS['model']}x -> {last_text(result)[:48]}")

    banner("Context and privacy")
    out = pii_agent().invoke({"messages": "My email is ada@example.com, please remember it."})
    print("PII redacted:", out["messages"][0].text)
    agent = summarizing_agent()
    cfg: Any = {"configurable": {"thread_id": "s"}}
    for turn in ("hi, I'm Bob", "I like graphs", "tell me a fact", "another fact"):
        agent.invoke({"messages": turn}, cfg)
    kept = agent.get_state(cfg).values["messages"]
    print("Summarization kept:", [(m.type, m.text[:48]) for m in kept])

# EXERCISES
# 1. Explain the three lines printed under "ModelFallback": why does the primary get called three
#    times, three times, and once, and why does only the middle line print a failure message?
# 2. Make ToolCallLimitMiddleware use exit_behavior="error" and catch the exception type.
# 3. Write a PIIMiddleware with a custom regex `detector` for an order number like ORD-12345.

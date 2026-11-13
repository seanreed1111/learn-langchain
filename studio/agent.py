"""Studio entry point: the graph that `langgraph dev` serves (lesson 17).

Rules for a graph that Studio / the Agent Server loads:
  * export a COMPILED graph at module level (here `graph`); `langgraph.json` points at `module.py:graph`
  * do NOT pass a checkpointer: the server provides persistence (pauses, threads and time travel in Studio
    all rely on it). That is why `HumanInTheLoopMiddleware` works here without `checkpointer=`.
  * the module is imported once and the graph is reused for every run, so it must be reusable. A scripted
    model is single-use, so offline we cycle its script; with ANTHROPIC_API_KEY set it uses a real model.

Run:  langgraph dev          (from the repo root; then open the Studio URL it prints)
"""

import itertools
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lessons"))

from common import ScriptedChatModel, have_real_llm, make_real_chat_model, tool_call_msg
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware
from langchain.messages import AIMessage
from langchain.tools import tool


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order (read-only)."""
    return f"order {order_id}: $79.00, delivered"


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund money to the customer (a real side effect: it needs approval)."""
    return f"Refunded ${amount:.2f} for order {order_id}."


def build_model() -> Any:
    if have_real_llm():
        return make_real_chat_model()
    script = [
        tool_call_msg("lookup_order", {"order_id": "123"}, "c1"),
        tool_call_msg("issue_refund", {"order_id": "123", "amount": 79.0}, "c2"),
        AIMessage(content="All done: the refund has been processed."),
    ]
    return ScriptedChatModel(messages=itertools.cycle(script))  # reusable: the script repeats


graph = create_agent(  # ty: ignore[no-matching-overload]
    model=build_model(),
    tools=[lookup_order, issue_refund],
    system_prompt="You are a refunds agent. Look the order up, then refund it.",
    middleware=[
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        HumanInTheLoopMiddleware(
            interrupt_on={"issue_refund": {"allowed_decisions": ["approve", "edit", "reject"]}, "lookup_order": False}
        ),
    ],
    name="helpdesk",
)

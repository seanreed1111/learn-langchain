"""CAPSTONE - Help desk: LangChain agents inside a LangGraph workflow
=================================================================
Every lesson in one program. The outer shape is a StateGraph you control; the inner worker is a
`create_agent` with production middleware.

    START -> triage --(refund)--> refund_desk (an AGENT, as a subgraph node) -> END
                    \\--(other)---> faq (a plain function node)              -> END

  triage       lesson 04  structured output (`Triage` schema) decides the route
  refund_desk  lesson 06  create_agent, named, as a node                         (lesson 12)
               lesson 07  identity from CONTEXT (`Ctx.user_id`), never from a tool argument
               lesson 08/09  ModelCallLimit (budget) + ToolErrorMiddleware (errors become messages)
               lesson 10  HumanInTheLoopMiddleware: `issue_refund` cannot run without approval
  outer graph  lesson 07  one checkpointer (the pause is a saved checkpoint) + thread_id

Security properties the tests prove:
  * nothing is refunded before a human approves
  * a rejection leaves the refund store empty and the model is told why
  * a user cannot read another user's order, because ownership is checked against context

Run:  python capstone/helpdesk_agent.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lessons"))

from common import banner, get_model, show, tool_call_msg
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolErrorMiddleware,
)
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

ORDERS = {
    "123": {"owner": "u1", "total": 79.0, "status": "delivered"},
    "456": {"owner": "u2", "total": 15.0, "status": "shipped"},
}
REFUNDS: list[dict[str, Any]] = []  # a stand-in for a payments system


@dataclass
class Ctx:
    """Set by the application from the authenticated session. The model never sees or chooses it."""

    user_id: str


@tool
def lookup_order(order_id: str, runtime: ToolRuntime[Ctx]) -> str:
    """Look up one of the current customer's orders."""
    order = ORDERS.get(order_id)
    if order is None or order["owner"] != runtime.context.user_id:
        raise LookupError(f"order {order_id} not found")  # same answer for 'missing' and 'not yours'
    return f"order {order_id}: ${order['total']:.2f}, {order['status']}"


@tool
def issue_refund(order_id: str, amount: float, runtime: ToolRuntime[Ctx]) -> str:
    """Refund money to the customer for an order (sensitive)."""
    order = ORDERS.get(order_id)
    if order is None or order["owner"] != runtime.context.user_id:
        raise LookupError(f"order {order_id} not found")
    REFUNDS.append({"order_id": order_id, "amount": amount})
    return f"Refunded ${amount:.2f} for order {order_id}."


class Triage(BaseModel):
    """Choose where the message goes."""

    route: Literal["refund", "faq"]


class DeskState(MessagesState):
    route: str


def build_desk(
    triage_route: str = "refund",
    agent_script: list[AIMessage] | None = None,
    model_call_budget: int = 6,
):
    """Build the whole workflow. Scripts are single-use, so tests call this once per scenario."""
    script = agent_script or [
        tool_call_msg("lookup_order", {"order_id": "123"}, "c1"),
        tool_call_msg("issue_refund", {"order_id": "123", "amount": 79.0}, "c2"),
        AIMessage(content="Done - your refund has been processed."),
    ]
    refund_agent = create_agent(  # ty: ignore[no-matching-overload]
        model=get_model(script),
        tools=[lookup_order, issue_refund],
        system_prompt="You are a refunds agent. Look the order up, then refund it.",
        middleware=[
            ModelCallLimitMiddleware(run_limit=model_call_budget, exit_behavior="end"),
            ToolErrorMiddleware(on_error=lambda exc, req: f"Tool failed: {exc}"),
            HumanInTheLoopMiddleware(
                interrupt_on={"issue_refund": {"allowed_decisions": ["approve", "reject"]}, "lookup_order": False}
            ),
        ],
        context_schema=Ctx,
        name="refund_desk",
        # No checkpointer here: the OUTER graph owns persistence, and the subgraph inherits it.
    )
    classifier = get_model([tool_call_msg("Triage", {"route": triage_route})]).with_structured_output(Triage)

    def triage(state: DeskState) -> dict[str, Any]:
        return {"route": classifier.invoke(state["messages"]).route}

    def pick(state: DeskState) -> Literal["refund_desk", "faq"]:
        return "refund_desk" if state["route"] == "refund" else "faq"

    def faq(state: DeskState) -> dict[str, Any]:
        return {"messages": [AIMessage("See our help center: returns are free within 30 days.")]}

    b = StateGraph(DeskState, context_schema=Ctx)  # ty: ignore[invalid-argument-type]
    b.add_node("triage", triage)
    b.add_node("refund_desk", refund_agent)
    b.add_node("faq", faq)
    b.add_edge(START, "triage")
    b.add_conditional_edges("triage", pick)
    b.add_edge("refund_desk", END)
    b.add_edge("faq", END)
    return b.compile(checkpointer=InMemorySaver())


def cfg(thread_id: str) -> Any:
    return {"configurable": {"thread_id": thread_id}}


def run_refund(decision: dict[str, Any], user_id: str = "u1") -> dict[str, Any]:
    """Drive the whole flow: ask, pause for approval, then resume with `decision`.
    The scripted model's closing line matches the decision (a real model would react to the ToolMessage)."""
    REFUNDS.clear()
    closing = (
        "Done - your refund has been processed."
        if decision["type"] == "approve"
        else "Understood - no refund was issued."
    )
    desk = build_desk(
        agent_script=[
            tool_call_msg("lookup_order", {"order_id": "123"}, "c1"),
            tool_call_msg("issue_refund", {"order_id": "123", "amount": 79.0}, "c2"),
            AIMessage(content=closing),
        ]
    )
    config = cfg(f"thread-{user_id}-{decision['type']}")
    paused = desk.invoke({"messages": "Please refund order 123."}, config, context=Ctx(user_id))
    before_approval = list(REFUNDS)
    final = desk.invoke(Command(resume={"decisions": [decision]}), config, context=Ctx(user_id))
    return {"paused": paused, "refunds_before": before_approval, "final": final, "refunds_after": list(REFUNDS)}


if __name__ == "__main__":
    banner("Approve path")
    r = run_refund({"type": "approve"})
    request = r["paused"]["__interrupt__"][0].value["action_requests"][0]
    print("PAUSED for human:", request["name"], request["args"])
    print("refunds before approval:", r["refunds_before"])
    show(r["final"]["messages"])
    print("refunds after approval :", r["refunds_after"])

    banner("Reject path")
    r = run_refund({"type": "reject", "message": "Outside the refund window."})
    print("refunds after rejection:", r["refunds_after"])
    print("last message:", r["final"]["messages"][-1].text)

    banner("Another user's order is invisible")
    REFUNDS.clear()
    desk = build_desk(
        agent_script=[tool_call_msg("lookup_order", {"order_id": "456"}, "c1"), AIMessage(content="I can't find that order.")]
    )
    out = desk.invoke({"messages": "Where is order 456?"}, cfg("thread-spy"), context=Ctx("u1"))
    print([m.text for m in out["messages"] if m.type == "tool"])

    banner("Non-refund messages skip the agent entirely")
    out = build_desk("faq").invoke({"messages": "What is your return policy?"}, cfg("thread-faq"), context=Ctx("u1"))
    print(out["messages"][-1].text)

# EXTENSIONS
# 1. Swap InMemorySaver for SqliteSaver (pip install langgraph-checkpoint-sqlite) and resume after a restart.
# 2. Only require approval above $50 (HumanInTheLoopMiddleware's `when=` option on the tool config).
# 3. Stream the whole run with stream_mode=["updates", "messages"], subgraphs=True, version="v2" (lesson 11).
# 4. Add an eval set of support conversations with assertions on tool calls and REFUNDS.

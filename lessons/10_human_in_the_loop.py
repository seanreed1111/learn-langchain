"""LESSON 10 - Human-in-the-loop with HumanInTheLoopMiddleware
============================================================
learn-langgraph lesson 08 pauses a graph with `interrupt()` and resumes with `Command(resume=...)`.
`HumanInTheLoopMiddleware` is that same machinery, pre-built for the most common case: "a human must
approve this tool call before it runs".

    create_agent(
        ...,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={
            "issue_refund": {"allowed_decisions": ["approve", "edit", "reject"]},
            "lookup_order": False,                      # never ask
        })],
        checkpointer=InMemorySaver(),                   # REQUIRED: the pause is a saved checkpoint
    )

Flow (look at `result["__interrupt__"]`):
    1. invoke(...)  -> the model proposes a tool call; the graph PAUSES before the tool runs.
       The payload lists `action_requests` (name + args) and `review_configs` (allowed decisions).
    2. invoke(Command(resume={"decisions": [...]}), same thread config) -> continues.
       One decision per action request, in order:
         {"type": "approve"}
         {"type": "edit",   "edited_action": {"name": ..., "args": {...}}}
         {"type": "reject", "message": "why"}          (the model is told, and can adapt)

Why it is safe by construction: the middleware is an `after_model` NODE (look at the graph). It pauses
after the model proposes a call and before the tools node, so no side effect can happen first. The
interrupt rule from LangGraph still applies: the node re-runs from the top on resume, which is fine
here because the hook itself has no side effects.

Always pass the SAME `thread_id` when resuming, and use a durable checkpointer (SQLite/Postgres) in
production so a pause survives a restart.

Run:  python lessons/10_human_in_the_loop.py
"""

from typing import Any

from common import banner, get_model, tool_call_msg
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

REFUNDS: list[tuple[str, float]] = []  # a stand-in for a payments system


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order (read-only)."""
    return f"order {order_id}: $79.00, delivered"


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund money to the customer (a real side effect)."""
    REFUNDS.append((order_id, amount))
    return f"Refunded ${amount:.2f} for order {order_id}."


def refund_agent():
    script = [
        tool_call_msg("lookup_order", {"order_id": "123"}, "c1"),
        tool_call_msg("issue_refund", {"order_id": "123", "amount": 79.0}, "c2"),
        AIMessage(content="All done."),
    ]
    return create_agent(
        model=get_model(script),
        tools=[lookup_order, issue_refund],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "issue_refund": {"allowed_decisions": ["approve", "edit", "reject"]},
                    "lookup_order": False,  # read-only: no approval needed
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )


def threshold_agent(amount: float):
    """Only refunds ABOVE $50 need approval: `when` is a predicate on the proposed tool call."""
    script = [tool_call_msg("issue_refund", {"order_id": "123", "amount": amount}, "c1"), AIMessage(content="Handled.")]
    return create_agent(
        model=get_model(script),
        tools=[issue_refund],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "issue_refund": {
                        "allowed_decisions": ["approve", "reject"],
                        "when": lambda request: request.tool_call["args"]["amount"] > 50,
                    }
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )


def cfg(thread_id: str) -> Any:
    return {"configurable": {"thread_id": thread_id}}


def start(agent: Any, thread_id: str) -> Any:
    """Run until the refund is proposed. Returns the invoke result (it contains `__interrupt__`)."""
    return agent.invoke({"messages": "Please refund order 123."}, cfg(thread_id))


def decide(agent: Any, thread_id: str, decision: dict[str, Any]) -> Any:
    return agent.invoke(Command(resume={"decisions": [decision]}), cfg(thread_id))


if __name__ == "__main__":
    banner("The middleware is a node")
    print(list(refund_agent().get_graph().nodes))

    banner("1) Paused: nothing has been refunded yet")
    REFUNDS.clear()
    agent = refund_agent()
    paused = start(agent, "t-approve")
    payload = paused["__interrupt__"][0].value
    print("action_requests :", payload["action_requests"][0]["name"], payload["action_requests"][0]["args"])
    print("allowed         :", payload["review_configs"][0]["allowed_decisions"])
    print("graph is waiting at:", agent.get_state(cfg("t-approve")).next)
    print("REFUNDS         :", REFUNDS)

    banner("2) Approve")
    done = decide(agent, "t-approve", {"type": "approve"})
    print("final:", done["messages"][-1].text, "| REFUNDS:", REFUNDS)

    banner("3) Edit: the human lowers the amount before it runs")
    REFUNDS.clear()
    agent = refund_agent()
    start(agent, "t-edit")
    edited = {"name": "issue_refund", "args": {"order_id": "123", "amount": 40.0}}
    decide(agent, "t-edit", {"type": "edit", "edited_action": edited})
    print("REFUNDS:", REFUNDS)

    banner("4) Reject: the tool never runs and the model is told why")
    REFUNDS.clear()
    agent = refund_agent()
    start(agent, "t-reject")
    done = decide(agent, "t-reject", {"type": "reject", "message": "Refunds over $50 need a manager."})
    tool_msgs = [m.text for m in done["messages"] if m.type == "tool"]
    print("tool messages:", tool_msgs[-1])
    print("REFUNDS:", REFUNDS)

    banner("5) Conditional approval: `when=` gates only the risky cases")
    for amount in (20.0, 90.0):
        REFUNDS.clear()
        result = threshold_agent(amount).invoke({"messages": "refund please"}, cfg(f"t-{amount}"))
        print(f"${amount:>5.2f} ->", "PAUSED for approval" if "__interrupt__" in result else "ran without asking", "| REFUNDS:", REFUNDS)

# EXERCISES
# 1. Remove `checkpointer=` and run `start`. Read the error. Why can a pause not exist without one?
# 2. Resume with a thread_id that was never paused. What happens?
# 3. Change the threshold rule in (5) to ALSO require approval for any order id that starts with "VIP".
#    Then write the same rule as a custom after_model hook (lesson 08). Which is shorter, and which
#    can do something `when` cannot?

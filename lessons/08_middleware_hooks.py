"""LESSON 08 - Middleware: hooks that become graph nodes
======================================================
Middleware is how you customize `create_agent` without rewriting the graph. There are two styles,
and the difference is the whole lesson:

  NODE-STYLE hooks become real NODES in the compiled graph and update STATE (return a dict or None):
      before_agent   once, before the loop starts
      before_model   before EVERY model call
      after_model    after EVERY model response
      after_agent    once, when the loop ends
  WRAP-STYLE hooks wrap a call and decide whether/how to make it (they are NOT nodes):
      wrap_model_call(request, handler)   change the request (model, tools, system prompt), retry, fall back
      wrap_tool_call(request, handler)    intercept ONE tool call: catch errors, rewrite, block

  `@dynamic_prompt` is sugar for a wrap_model_call that only builds the system prompt.

Mental map to learn-langgraph: a node-style hook == a node you added with `add_node` + edges.
That is why a hook can RETURN `{"jump_to": "end"}` (like `Command(goto=END)`), why its state keys need a
`state_schema`, and why you can see it in `agent.get_graph()`.

ORDER with `middleware=[A, B]`:
    before_*   A then B        (first to last)
    wrap_*     A( B( model ))  (first is outermost)
    after_*    B then A        (last to first)

Write hooks as decorators (`@before_model`) for one hook, or as an `AgentMiddleware` class when you
need several hooks together, configuration, or async variants.

Run:  python lessons/08_middleware_hooks.py
"""

from collections.abc import Callable
from typing import Any, NotRequired, override

from common import banner, get_model, tool_call_msg
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    after_model,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime

TRACE: list[str] = []


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def flaky(x: int) -> int:
    """A tool that always fails."""
    raise RuntimeError("upstream is down")


# ---- A) Class-based middleware: all six hooks, and the order they run in ------------------------------
class Tracer(AgentMiddleware):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    @property
    @override
    def name(self) -> str:  # the node name is "<name>.<hook>"; make each instance unique
        return f"Tracer{self.label}"

    @override
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        TRACE.append(f"{self.label}.before_agent")

    @override
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        TRACE.append(f"{self.label}.before_model")

    @override
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        TRACE.append(f"{self.label}.wrap_model_call:enter")
        response = handler(request)
        TRACE.append(f"{self.label}.wrap_model_call:exit")
        return response

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        TRACE.append(f"{self.label}.after_model")

    @override
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        TRACE.append(f"{self.label}.after_agent")


def traced_agent():
    script = [tool_call_msg("add", {"a": 1, "b": 2}), AIMessage(content="3")]
    return create_agent(
        model=get_model(script), tools=[add], middleware=[Tracer("A"), Tracer("B")]
    )


# ---- B) A budget in STATE, enforced with jump_to ----------------------------------------------------
class BudgetState(AgentState):
    model_calls: NotRequired[int]  # NotRequired: callers need not pass it; hooks initialize it


@after_model(state_schema=BudgetState)
def count_calls(state: BudgetState, runtime: Runtime) -> dict[str, Any] | None:
    return {"model_calls": state.get("model_calls", 0) + 1}


@before_model(state_schema=BudgetState, can_jump_to=["end"])
def enforce_budget(state: BudgetState, runtime: Runtime) -> dict[str, Any] | None:
    if state.get("model_calls", 0) >= 2:
        # Same idea as Command(goto=END): leave the loop now, with a defined final message.
        return {"messages": [AIMessage("Stopped: model-call budget used up.")], "jump_to": "end"}
    return None


@before_model(state_schema=BudgetState)  # NOTE: no can_jump_to declared
def undeclared_jump(state: BudgetState, runtime: Runtime) -> dict[str, Any] | None:
    return {"messages": [AIMessage("Stopped by an undeclared jump.")], "jump_to": "end"}


def undeclared_jump_agent():
    """GOTCHA: a hook that returns `jump_to` without declaring `can_jump_to` is silently ignored."""
    return create_agent(
        model=get_model([AIMessage(content="The model answered normally.")]),
        tools=[add],
        middleware=[undeclared_jump],
        state_schema=BudgetState,
    )


def budget_agent():
    endless = [tool_call_msg("add", {"a": i, "b": 1}, f"c{i}") for i in range(20)]
    return create_agent(
        model=get_model(endless), tools=[add], middleware=[count_calls, enforce_budget], state_schema=BudgetState
    )


# ---- C) wrap_model_call / dynamic_prompt: change what the MODEL sees, not the state -----------------
SEEN: list[dict[str, Any]] = []


@dynamic_prompt
def prompt_by_progress(request: ModelRequest) -> str:
    n = len(request.messages)
    return "Be brief." if n < 3 else f"You have used {n} messages; wrap up now."


def prompt_agent():
    @wrap_model_call
    def record(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        SEEN.append({"system": request.system_message.text if request.system_message else None, "n": len(request.messages)})
        return handler(request)

    script = [tool_call_msg("add", {"a": 1, "b": 2}), AIMessage(content="3")]
    return create_agent(model=get_model(script), tools=[add], middleware=[prompt_by_progress, record])


# ---- D) wrap_tool_call: turn a tool failure into something the model can read -----------------------
@wrap_tool_call
def tool_errors_to_messages(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Any]
) -> ToolMessage | Any:
    try:
        return handler(request)
    except Exception as exc:
        return ToolMessage(
            content=f"Tool failed: {exc}. Try something else.",
            tool_call_id=request.tool_call["id"],
            status="error",
        )


def resilient_agent():
    script = [tool_call_msg("flaky", {"x": 1}), AIMessage(content="The tool is down; I can't help right now.")]
    return create_agent(model=get_model(script), tools=[flaky], middleware=[tool_errors_to_messages])


if __name__ == "__main__":
    banner("A) Hooks are nodes (look at the graph)")
    print(list(traced_agent().get_graph().nodes))

    banner("A) Execution order with middleware=[A, B]")
    TRACE.clear()
    traced_agent().invoke({"messages": "1+2?"})
    for line in TRACE:
        print(" ", line)

    banner("B) A budget in state + jump_to")
    out = budget_agent().invoke({"messages": "keep adding"})
    print("model_calls:", out["model_calls"], "| last message:", out["messages"][-1].text)

    out = undeclared_jump_agent().invoke({"messages": "hi"})
    print("jump_to without can_jump_to ->", out["messages"][-1].text, "(the jump was ignored)")

    banner("C) dynamic_prompt / wrap_model_call: the model's view of the request")
    SEEN.clear()
    prompt_agent().invoke({"messages": "1+2?"})
    for s in SEEN:
        print(" ", s)

    banner("D) wrap_tool_call: tool exception -> error ToolMessage -> model recovers")
    out = resilient_agent().invoke({"messages": "use the flaky tool"})
    print([(type(m).__name__, m.text) for m in out["messages"][1:]])

# EXERCISES
# 1. In (A), swap the order to [B, A]. Which trace lines flip, and which stay put? Why?
# 2. In (B), delete `can_jump_to=["end"]` from `enforce_budget` and rerun. What changes, and why is a
#    silent no-op worse than an error? (See `undeclared_jump_agent`.)
# 3. Write a `wrap_model_call` that retries a failing model call up to 3 times with a short sleep.
#    (Then read lesson 09's ModelRetryMiddleware and compare.)

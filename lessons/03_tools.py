"""LESSON 03 - Tools: how a Python function becomes something the model can call
==============================================================================
A tool is a Runnable with three things the MODEL sees: a name, a description, and an argument schema.
`@tool` builds all three from your function:

    name         <- the function name            (override: @tool("web_search"))
    description  <- the docstring                (override: @tool(description="..."))
    args schema  <- the type hints (+ Args: section of the docstring)

The model never runs code. It returns an AIMessage with `tool_calls`. A **ToolNode** (a LangGraph node
from `langgraph.prebuilt`) executes them and returns ToolMessages. `create_agent` wires that for you
(lesson 06); here we use the pieces directly.

What to remember for LangGraph:
  * Type hints and docstrings are PROMPT ENGINEERING: they decide when and how the model calls the tool.
  * ToolNode runs the calls in a message CONCURRENTLY and returns one ToolMessage per call.
  * By default ToolNode lets a tool's own exceptions PROPAGATE (the run crashes). Bad arguments and
    unknown tool names become error ToolMessages. Pass `handle_tool_errors=True` (or use middleware,
    lesson 09) to turn exceptions into messages the model can react to. (`handle_tool_errors=False`
    is stricter still: even bad arguments raise.)
  * A ToolNode must run inside a graph (it needs graph config), so we wrap it in a one-node graph.

Run:  python lessons/03_tools.py
"""

from typing import Any, Literal, cast

from common import banner, show
from langchain.messages import AIMessage, AnyMessage
from langchain.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


# ---- 1) The simplest tool ----------------------------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: The first integer.
        b: The second integer.
    """
    return a * b


# ---- 2) A tool with an explicit Pydantic schema (richer constraints and descriptions) ----------------
class WeatherInput(BaseModel):
    """Input for weather queries."""

    city: str = Field(description="City name, for example 'Paris'")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="Temperature unit")


@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city} ({'22C' if units == 'celsius' else '72F'})."


# ---- 3) A tool that fails -----------------------------------------------------------------------------
@tool
def flaky_lookup(order_id: str) -> str:
    """Look up an order. (Raises for unknown orders.)"""
    if order_id != "123":
        raise KeyError(f"order {order_id} not found")
    return "order 123: shipped"


def tool_graph(*tools: Any, **tool_node_kwargs: Any):
    """A one-node graph around ToolNode, so we can run tool calls the way an agent would.
    Extra kwargs go to ToolNode, so omitting `handle_tool_errors` keeps ToolNode's real default."""
    b = StateGraph(MessagesState)  # ty: ignore[invalid-argument-type]
    b.add_node("tools", ToolNode(list(tools), **tool_node_kwargs))
    b.add_edge(START, "tools")
    return b.compile()


def run_calls(graph: Any, *calls: dict[str, Any]) -> list[AnyMessage]:
    """Feed an AIMessage that requests `calls` and return the ToolMessages that come back."""
    ai = AIMessage(content="", tool_calls=list(calls))
    messages = graph.invoke({"messages": [ai]})["messages"]
    assert isinstance(messages, list)
    return cast("list[AnyMessage]", messages[1:])  # drop our own AIMessage


if __name__ == "__main__":
    banner("What the MODEL sees")
    print("name       :", multiply.name)
    print("description:", multiply.description)
    print("args       :", multiply.args)
    print("weather    :", get_weather.args)

    banner("A tool is a Runnable")
    print("multiply.invoke ->", multiply.invoke({"a": 6, "b": 7}))
    # Given a full tool-call dict, a tool returns a ToolMessage (this is what ToolNode does for you).
    msg = multiply.invoke({"name": "multiply", "args": {"a": 6, "b": 7}, "id": "call_1", "type": "tool_call"})
    print("with a tool_call ->", type(msg).__name__, msg.content, msg.tool_call_id)

    banner("ToolNode: several calls in one AIMessage -> one ToolMessage each")
    g = tool_graph(multiply, get_weather)
    show(
        run_calls(
            g,
            {"name": "multiply", "args": {"a": 3, "b": 4}, "id": "c1", "type": "tool_call"},
            {"name": "get_weather", "args": {"city": "Paris"}, "id": "c2", "type": "tool_call"},
        )
    )

    banner("Errors: bad arguments and unknown tools become messages")
    show(run_calls(g, {"name": "multiply", "args": {"a": "six", "b": 7}, "id": "c3", "type": "tool_call"}))
    show(run_calls(g, {"name": "divide", "args": {}, "id": "c4", "type": "tool_call"}))

    banner("A tool that RAISES: crash by default, message with handle_tool_errors=True")
    bad_call = {"name": "flaky_lookup", "args": {"order_id": "999"}, "id": "c5", "type": "tool_call"}
    try:
        run_calls(tool_graph(flaky_lookup), bad_call)
    except KeyError as exc:
        print("default ToolNode -> exception escaped:", repr(exc))
    result = run_calls(tool_graph(flaky_lookup, handle_tool_errors=True), bad_call)
    show(result)
    print("status:", getattr(result[0], "status", None))

# EXERCISES
# 1. Remove the docstring from `multiply` and read the error. Then remove the type hints. Which is fatal?
# 2. Add a `Literal["add", "sub"]` argument to a tool and call it with a value outside the Literal.
# 3. Give ToolNode `handle_tool_errors="Something went wrong, try a different input."` (a string) and
#    see what the ToolMessage content becomes.

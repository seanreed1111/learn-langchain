"""LESSON 05 - The bridge: LangChain parts inside a LangGraph loop
================================================================
Lessons 01-04 gave you the parts. This lesson snaps them together into the agent loop, using ONLY
LangGraph's graph API plus LangChain objects:

    model.bind_tools(tools)   -> the model node's brain          (lesson 03)
    messages + add_messages   -> the state                       (lesson 02)
    ToolNode(tools)           -> the tools node
    tools_condition           -> the router: "tools" if the last AIMessage has tool_calls, else END

    START -> model --(tool_calls?)--> tools --> model ... --> END

That is the entire ReAct agent. (learn-langgraph lesson 05 writes the tools node by hand; here we use
the prebuilt one.) Lesson 06 shows that `create_agent` builds a graph with this same shape and adds
the hooks you will want in production.

Two things that bite people here:
  * `bind_tools` returns a NEW runnable; keep that one and call it in the node.
  * The loop ends only when the model stops asking for tools. Set a recursion limit as a seatbelt.

Run:  python lessons/05_model_and_tools_in_a_graph.py
"""

from typing import Any

from common import banner, get_model, show, tool_call_msg
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


TOOLS = [multiply, add]
SYSTEM = SystemMessage("You are a careful calculator. Use tools for arithmetic.")


def scripted_conversation() -> list[AIMessage]:
    """Step 1: ask for multiply(6, 7). Step 2: ask for add(42, 8). Step 3: answer."""
    return [
        tool_call_msg("multiply", {"a": 6, "b": 7}, "call_1"),
        tool_call_msg("add", {"a": 42, "b": 8}, "call_2"),
        AIMessage(content="6 x 7 + 8 = 50."),
    ]


def build_agent(script: list[AIMessage] | None = None):
    model = get_model(script or scripted_conversation()).bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict[str, Any]:
        # The system message is added per call, not stored in state (so it never piles up in history).
        return {"messages": [model.invoke([SYSTEM, *state["messages"]])]}

    b = StateGraph(MessagesState)  # ty: ignore[invalid-argument-type]
    b.add_node("model", call_model)
    b.add_node("tools", ToolNode(TOOLS))
    b.add_edge(START, "model")
    b.add_conditional_edges("model", tools_condition)  # -> "tools" or END
    b.add_edge("tools", "model")  # close the loop
    return b.compile()


agent = build_agent()

if __name__ == "__main__":
    banner("The loop, as a graph")
    print(build_agent().get_graph().draw_mermaid().split("graph TD;")[1].split("classDef")[0].strip())

    banner("A two-tool question")
    result = build_agent().invoke({"messages": [HumanMessage("What is 6 times 7, plus 8?")]})
    show(result["messages"])

    banner("Seatbelt: a model that never stops asking for tools")
    forever = [tool_call_msg("add", {"a": 1, "b": 1}, f"call_{i}") for i in range(50)]
    try:
        build_agent(forever).invoke({"messages": [HumanMessage("go")]}, {"recursion_limit": 6})
    except Exception as exc:  # GraphRecursionError
        print(type(exc).__name__, "-", str(exc).splitlines()[0])

# EXERCISES
# 1. Add an `llm_calls: int` counter to the state (a custom subclass of MessagesState) that the model
#    node increments, and route to END after 3 calls even if the model wants more tools.
# 2. Make `add` raise for negative numbers. Run it with `ToolNode(TOOLS)` and with
#    `ToolNode(TOOLS, handle_tool_errors=True)` (lesson 03). Which one lets the model recover?
# 3. Print `result["messages"][-1].usage_metadata` after swapping in a real model.

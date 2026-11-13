"""LESSON 06 - create_agent: the lesson-05 loop, as one call
==========================================================
`langchain.agents.create_agent` builds the SAME graph you hand-wired in lesson 05 and returns a
`CompiledStateGraph`. It is not a different engine: every LangGraph feature (checkpointers, streaming,
interrupts, subgraphs, Studio) works on it because it IS a LangGraph graph.

    create_agent(
        model,                  # "provider:model-id" string OR a chat-model object
        tools,                  # functions, @tool objects, or tool dicts
        system_prompt=...,      # str or SystemMessage
        response_format=...,    # schema -> result["structured_response"]
        state_schema=...,       # extend AgentState with your own keys      (lesson 07)
        context_schema=...,     # per-run read-only data (user id, ...)      (lesson 07)
        middleware=[...],       # hooks around the loop                      (lessons 08-09)
        checkpointer=..., store=..., name=...,
    )

What you get that lesson 05 lacked: middleware hooks, structured output, a typed `AgentState`, tool
runtime injection, per-run context, a proper name for use as a subgraph.

What it replaces: the model node, ToolNode wiring and router. The node names are `model` and `tools`.
Older tutorials import `create_react_agent` from `langgraph.prebuilt`; that is deprecated in favor of this.

Run:  python lessons/06_create_agent.py
"""

import importlib
from typing import Any

from common import banner, get_model, show, tool_call_msg
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from pydantic import BaseModel

lesson05 = importlib.import_module("05_model_and_tools_in_a_graph")  # reuse its tools and script


def build_agent(script: list[AIMessage] | None = None, **kwargs: Any):
    """The same agent as lesson 05, in one call. `model` accepts a model object (what we pass here, so
    the scripted fake works) or a string like "anthropic:claude-sonnet-5-5"."""
    return create_agent(
        model=get_model(script or lesson05.scripted_conversation()),
        tools=lesson05.TOOLS,
        system_prompt="You are a careful calculator. Use tools for arithmetic.",
        **kwargs,
    )


class Answer(BaseModel):
    """The final answer, as data."""

    value: int
    explanation: str


def structured_agent():
    script = [
        tool_call_msg("multiply", {"a": 6, "b": 7}, "call_1"),
        # With response_format, the model "calls" a tool named after the schema to deliver its answer.
        tool_call_msg("Answer", {"value": 42, "explanation": "6 times 7"}, "call_2"),
    ]
    return build_agent(script, response_format=Answer)


def same_shape_as_lesson_05() -> bool:
    """create_agent and the hand-built graph produce the same message trace for the same script."""
    mine = build_agent().invoke({"messages": [HumanMessage("6 times 7, plus 8?")]})["messages"]
    theirs = lesson05.build_agent().invoke({"messages": [HumanMessage("6 times 7, plus 8?")]})["messages"]
    return [(type(m).__name__, m.text) for m in mine] == [(type(m).__name__, m.text) for m in theirs]


@tool
def explode(x: int) -> int:
    """A tool that raises."""
    raise ValueError("boom")


if __name__ == "__main__":
    banner("It is a compiled LangGraph graph")
    agent = build_agent()
    print("type:", type(agent).__name__)
    print("nodes:", list(agent.get_graph().nodes))
    print("input keys:", list(agent.get_input_jsonschema()["properties"]))
    print("output keys:", list(agent.get_output_jsonschema()["properties"]))

    banner("Same behavior as the lesson-05 graph")
    print("identical trace:", same_shape_as_lesson_05())

    banner("Input can be a string, a dict, or message objects")
    out = build_agent().invoke({"messages": "What is 6 times 7, plus 8?"})  # a bare string is a user message
    show(out["messages"])

    banner("response_format -> structured_response")
    result = structured_agent().invoke({"messages": "What is 6 times 7?"})
    print("structured_response:", result["structured_response"])

    banner("Defaults to know: a raising tool crashes the run")
    crashing = create_agent(
        model=get_model([tool_call_msg("explode", {"x": 1}), AIMessage(content="recovered")]),
        tools=[explode],
    )
    try:
        crashing.invoke({"messages": "go"})
    except ValueError as exc:
        print("ValueError escaped:", exc, "(fix: lesson 09, ToolErrorMiddleware or wrap_tool_call)")

    banner("The seatbelt is the same recursion_limit as any graph")
    loop = [tool_call_msg("add", {"a": 1, "b": 1}, f"c{i}") for i in range(50)]
    try:
        build_agent(loop).invoke({"messages": "go"}, {"recursion_limit": 6})
    except Exception as exc:
        print(type(exc).__name__)

# EXERCISES
# 1. Pass `name="calculator"` and print `agent.name`. Why does a subgraph benefit from a name? (lesson 12)
# 2. Pass a real model string, `model="anthropic:claude-haiku-5-5"`, with an API key in `.env`, and drop
#    the scripted model. Does the trace keep the same shape?
# 3. Add a third tool and a question needing all three. Print `agent.get_graph().draw_mermaid()`: what
#    changed in the diagram, and what did not?

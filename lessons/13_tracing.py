"""LESSON 13 - Tracing: seeing inside an agent run with LangSmith
===============================================================
LangSmith is the observability layer for LangChain and LangGraph. A TRACE is the tree of everything one
run did, and you already know its shape, because it is the graph you built:

    chain LangGraph              <- the whole run (root)
      chain model                <- a graph node
        llm   ChatModel          <- the actual model call: prompt in, AIMessage out, tokens, latency
      chain tools                <- a graph node
        tool  multiply           <- one tool call: arguments in, result out
      chain model ...            <- the loop repeats

Every node is a RUN with a `run_type` (chain, llm, tool, retriever, ...), inputs, outputs, timing, tags and
metadata. LangGraph and `create_agent` emit all of this automatically through LangChain's callback system.

Turning it on, for real (no code changes):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=learn-langchain        # which project the traces land in (optional)

Controlling it in code:
    ls.tracing_context(enabled=True/False, project_name=..., tags=[...], metadata={...}, client=...)
    config={"run_name": ..., "tags": [...], "metadata": {...}}   on any invoke (children inherit them)
    @traceable                                                   wrap YOUR functions so they appear too

HOW THESE LESSONS RUN OFFLINE: `LocalLangSmith` (lessons/ls_offline.py) gives the LangSmith client a
recording HTTP session. The agent is traced exactly as it would be for real; the payloads are captured in
memory instead of being uploaded. So what you print here IS what the LangSmith UI would show.

Run:  python lessons/13_tracing.py
"""

import importlib
import json
from typing import Any

import langsmith as ls
from common import banner
from langchain.agents import create_agent
from langchain.tools import tool
from langsmith import traceable
from ls_offline import LocalLangSmith

lesson05 = importlib.import_module("05_model_and_tools_in_a_graph")  # reuse its scripted two-tool run


# ---- A traced helper that a TOOL calls: it shows up as a child of the tool run ------------------------
@traceable(name="price_lookup", run_type="retriever", tags=["pricing"])
def price_lookup(sku: str) -> float:
    """Imagine a database or HTTP call. `@traceable` makes it a visible step in the trace."""
    return {"A1": 19.5}.get(sku, 0.0)


@tool
def get_price(sku: str) -> str:
    """Get the price of a product by SKU."""
    return f"{sku} costs ${price_lookup(sku):.2f}"  # ty: ignore[invalid-argument-type]


def priced_agent():
    from common import get_model, tool_call_msg
    from langchain.messages import AIMessage

    script = [tool_call_msg("get_price", {"sku": "A1"}), AIMessage(content="A1 costs $19.50.")]
    return create_agent(model=get_model(script), tools=[get_price], name="shop")


# ---- Your own function as the ROOT: it wraps the agent call and carries tags/metadata -----------------
@traceable(name="handle_request", tags=["api"], metadata={"route": "/chat"})
def handle_request(question: str) -> str:
    result = lesson05.build_agent().invoke(
        {"messages": question},
        {"tags": ["lesson13"], "metadata": {"user_id": "u1"}},  # config tags/metadata reach every child run
    )
    return str(result["messages"][-1].text)


def trace_of(fn: Any, *args: Any, project: str = "learn-langchain", **context: Any) -> LocalLangSmith:
    """Run `fn(*args)` under a tracing context bound to a fresh local (recording) client."""
    local = LocalLangSmith()
    with ls.tracing_context(enabled=True, client=local.client, project_name=project, **context):
        fn(*args)
    return local


def run_by_name(local: LocalLangSmith, name: str) -> dict[str, Any]:
    return next(r for r in local.runs() if r["name"] == name)


if __name__ == "__main__":
    banner("A) The trace of one agent run is the graph you built")
    local = LocalLangSmith()
    config: Any = {"run_name": "calc", "tags": ["demo"], "metadata": {"user": "u1"}}
    with ls.tracing_context(enabled=True, client=local.client, project_name="learn-langchain"):
        lesson05.build_agent().invoke({"messages": "What is 6 times 7, plus 8?"}, config)
    print("\n".join(local.tree()))

    banner("B) What one run records (the 'model' node's first run)")
    node = next(r for r in local.runs() if r["name"] == "model")
    print("run_type  :", node["run_type"])
    print("project   :", node["session_name"])
    print("tags      :", node["tags"])
    meta = node["extra"]["metadata"]
    print("metadata  :", {k: meta[k] for k in ("user", "langgraph_node", "langgraph_step") if k in meta})

    banner("C) @traceable puts YOUR code in the trace (a root, and a helper inside a tool)")
    local = trace_of(handle_request, "What is 6 times 7, plus 8?")
    print("\n".join(local.tree()))
    root = run_by_name(local, "handle_request")
    print("root tags    :", root["tags"], "| metadata:", root["extra"]["metadata"].get("route"))
    print("root inputs  :", root["inputs"])
    print("root outputs :", root["outputs"])
    local = trace_of(priced_agent().invoke, {"messages": "price of A1?"})
    print("--- a @traceable helper called by a tool:")
    print("\n".join(local.tree()))

    banner("D) tracing_context decides whether anything is recorded at all")
    off = LocalLangSmith()
    with ls.tracing_context(enabled=False, client=off.client):
        handle_request("hi")  # ty: ignore[invalid-argument-type]
    print("enabled=False -> runs recorded:", len(off.runs()))
    print("enabled=True  -> runs recorded:", len(trace_of(handle_request, "hi").runs()))

    banner("E) The payload is plain JSON (this is what would be uploaded)")
    first = json.dumps(local.runs()[0], default=str)
    print(first[:160], "...")

# EXERCISES
# 1. In (A), pass `run_name="my_agent"` only on the `tools` node by wrapping ToolNode with .with_config().
#    Hint: in lesson 05's graph, `add_node("tools", ToolNode(TOOLS).with_config({"run_name": "run_tools"}))`.
# 2. Add `tags=["experiment-b"]` to `trace_of(...)` via tracing_context and find which runs inherit it.
# 3. Put LANGSMITH_TRACING=true and a real LANGSMITH_API_KEY in `.env`, run lesson 06 against a real model,
#    and compare the trace in the LangSmith UI with the tree printed in (A). Which fields are new?

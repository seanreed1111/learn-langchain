"""LESSON 14 - Trace hygiene: privacy, cost, and conversations
============================================================
Tracing records prompts, tool arguments and answers. Before it goes near production you need four controls,
and each one has a sharp edge. All of them are configured on the LangSmith `Client` or `tracing_context`
(so they apply to LangChain and LangGraph runs without touching agent code).

  1. REDACTION   Client(anonymizer=create_anonymizer([...regex rules...]))   scrub inputs/outputs/metadata
                 Client(hide_inputs=..., hide_outputs=...)                  blank them entirely
                 @traceable(process_inputs=..., process_outputs=...)        per function, wins over Client
                 !! It does NOT touch run NAMES or TAGS. Never put a secret there.
  2. SAMPLING    Client(tracing_sampling_rate=0.1)  or  LANGSMITH_TRACING_SAMPLING_RATE=0.1
                 Sampling is per TRACE: you keep whole runs, never half of one.
  3. CONDITIONAL tracing_context(enabled=<your business rule>)   by tenant, data sensitivity, feature flag
  4. THREADS     LangSmith groups a conversation's traces by `thread_id` (or `session_id`) metadata that
                 must be on EVERY run. With LangGraph you get it free: the `thread_id` in
                 `config["configurable"]` is copied into the metadata of every run.

Why this matters for LangGraph: a checkpointer already stores the full state per thread in YOUR database;
LangSmith traces are a SECOND copy, in a vendor's. Decide deliberately what goes in it.

We prove each claim by scanning the captured payloads (the offline `LocalLangSmith` from lesson 13).

Run:  python lessons/14_trace_hygiene.py
"""

import importlib
import json
import re
from typing import Any

import langsmith as ls
from common import banner, get_model
from langchain.agents import create_agent
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable
from langsmith.anonymizer import StringNodeRule, create_anonymizer
from ls_offline import LocalLangSmith

lesson05 = importlib.import_module("05_model_and_tools_in_a_graph")

EMAIL = r"[\w.+-]+@[\w-]+\.\w{2,}"
ORDER = r"ORD-\d{5}"
SECRET_QUESTION = "Mail ada@example.com about ORD-12345: what is 6 times 7, plus 8?"


def anonymizing_client() -> LocalLangSmith:
    """Rules run over every string in inputs, outputs and metadata before upload."""
    rules = create_anonymizer(
        [
            StringNodeRule(pattern=re.compile(EMAIL), replace="<email>"),
            StringNodeRule(pattern=re.compile(ORDER), replace="<order>"),
        ]
    )
    return LocalLangSmith(anonymizer=rules)


def leaked(local: LocalLangSmith, needle: str) -> bool:
    """Does `needle` appear anywhere in anything that would have been uploaded?"""
    return needle in json.dumps(local.session.calls, default=str)


def run_agent(local: LocalLangSmith, question: str, config: Any = None) -> None:
    with ls.tracing_context(enabled=True, client=local.client):
        lesson05.build_agent().invoke({"messages": question}, config)


# ---- 1b) Per-function control with process_inputs / process_outputs ----------------------------------
def only_length(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"text_length": len(inputs.get("text", ""))}


def only_ok(output: Any) -> dict[str, Any]:
    return {"summary": "redacted", "ok": bool(output)}


@traceable(name="summarize", process_inputs=only_length, process_outputs=only_ok)
def summarize(text: str) -> str:
    return f"Summary of: {text}"


# ---- 3) Conditional tracing: a business rule decides --------------------------------------------------
TRACED_TENANTS = {"internal", "beta-customer"}


def handle(local: LocalLangSmith, tenant: str, question: str) -> None:
    """Customers who have not agreed to trace-sharing are never traced."""
    with ls.tracing_context(enabled=tenant in TRACED_TENANTS, client=local.client):
        lesson05.build_agent().invoke({"messages": question})


# ---- 4) Threads ----------------------------------------------------------------------------------------
def conversation_agent():
    script = [AIMessage(content="Nice to meet you, Ada."), AIMessage(content="Your name is Ada.")]
    return create_agent(model=get_model(script), tools=[], checkpointer=InMemorySaver())


if __name__ == "__main__":
    banner("1) Redaction: the anonymizer scrubs inputs, outputs, tool arguments and metadata")
    local = anonymizing_client()
    run_agent(local, SECRET_QUESTION, {"metadata": {"requester": "ada@example.com"}})
    print("email leaked :", leaked(local, "ada@example.com"))
    print("order leaked :", leaked(local, "ORD-12345"))
    root = local.runs()[0]
    print("root inputs  :", json.dumps(root["inputs"], default=str)[:90], "...")
    print("metadata     :", root["extra"]["metadata"]["requester"])

    banner("1a) GOTCHA: run names and tags are NOT scrubbed")
    local = anonymizing_client()
    run_agent(local, "hi", {"run_name": "run-for-ada@example.com", "tags": ["ada@example.com"]})
    print("root name   :", local.runs()[0]["name"])
    print("root tags   :", local.runs()[0]["tags"])
    print("email leaked:", leaked(local, "ada@example.com"), "<- put identifiers in context, never in names or tags")

    banner("1b) Per-function control: @traceable(process_inputs=..., process_outputs=...)")
    local = LocalLangSmith()
    with ls.tracing_context(enabled=True, client=local.client):
        summarize("a long confidential document")  # ty: ignore[invalid-argument-type]
    run = local.runs()[0]
    print("inputs :", run["inputs"], "| outputs:", run["outputs"])
    print("text leaked:", leaked(local, "confidential"))

    banner("1c) Blanket hiding: hide_inputs / hide_outputs")
    local = LocalLangSmith(hide_inputs=lambda inputs: {}, hide_outputs=lambda outputs: {})
    run_agent(local, SECRET_QUESTION)
    print("root inputs:", local.runs()[0]["inputs"], "| outputs:", local.runs()[0]["outputs"])
    print("question leaked:", leaked(local, "ORD-12345"))

    banner("2) Sampling is per trace: whole runs are kept or dropped")
    for rate in (0.0, 1.0):
        local = LocalLangSmith(tracing_sampling_rate=rate)
        run_agent(local, "hi")
        print(f"rate {rate}: runs recorded = {len(local.runs())}")
    local = LocalLangSmith(tracing_sampling_rate=0.5)
    with ls.tracing_context(enabled=True, client=local.client):
        for _ in range(20):
            lesson05.build_agent().invoke({"messages": "hi"})
    per_trace: dict[str, int] = {}
    for r in local.runs():
        per_trace[r["trace_id"]] = per_trace.get(r["trace_id"], 0) + 1
    print(f"rate 0.5: kept {len(per_trace)} of 20 traces; runs per kept trace = {sorted(set(per_trace.values()))}")

    banner("3) Conditional tracing by tenant")
    for tenant in ("internal", "acme-corp"):
        local = LocalLangSmith()
        handle(local, tenant, "hi")
        print(f"{tenant:<10} -> runs recorded: {len(local.runs())}")

    banner("4) Threads: thread_id reaches every run's metadata automatically")
    local = LocalLangSmith()
    agent = conversation_agent()
    with ls.tracing_context(enabled=True, client=local.client):
        for turn in ("Hi, I'm Ada.", "What's my name?"):
            agent.invoke({"messages": turn}, {"configurable": {"thread_id": "T-1"}})
    ids = {r["extra"]["metadata"].get("thread_id") for r in local.runs()}
    roots = [r for r in local.runs() if r.get("parent_run_id") is None]
    print(f"{len(roots)} root traces, thread_id values across ALL {len(local.runs())} runs: {ids}")

# EXERCISES
# 1. Add a rule that redacts phone numbers like 555-123-4567 and prove with `leaked(...)`.
# 2. Make a `@traceable` tool helper (lesson 13) take a secret argument and hide it with process_inputs.
# 3. In (4), start a second thread "T-2" and group the roots by thread_id the way the LangSmith UI does.

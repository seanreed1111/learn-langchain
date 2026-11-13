"""LESSON 12 - Multi-agent systems, and when to drop down to a StateGraph
=======================================================================
Two ways to combine agents. Both come straight from the idea that an agent is a graph (lesson 06).

PATTERN 1: SUBAGENT AS A TOOL (the supervisor keeps control)
    A specialist is a full `create_agent(...)`. The supervisor sees it as an ordinary @tool whose body
    calls `specialist.invoke(...)` and returns the last message's text.
      + the specialist gets a CLEAN context window; the supervisor only sees the final answer
      + trivial to build, test and swap
      - everything flows through the supervisor, and the specialist's inner steps are hidden from it

PATTERN 2: AGENTS AS NODES IN YOUR OWN GRAPH (you keep control; LangGraph decides the flow)
    Drop down to `StateGraph`. A compiled agent is a Runnable, so `add_node("billing", billing_agent)`
    just works (it becomes a SUBGRAPH). When parent and agent share the `messages` key, no glue code is
    needed. Routing is explicit: a classifier node (lesson 04) plus a conditional edge.
      + deterministic, testable control flow; each hop is visible in the graph and in traces
      + mix agents with plain functions, human approval, parallel branches, retries
      - you write the wiring

HOW TO CHOOSE (the lesson-06 question, "when would you NOT use create_agent alone?"):
  * One domain, a handful of tools                    -> a single create_agent
  * Several domains, one conversation owner           -> subagents as tools
  * Fixed routing / approvals / parallel branches     -> StateGraph with agents as nodes
  * Cycles or state that is not "a list of messages"  -> StateGraph (agents optional)
Give every agent a `name=`: it labels the subgraph, and streaming metadata (`lc_agent_name`).

Run:  python lessons/12_multi_agent_and_dropping_down.py
"""

from typing import Any, Literal

from common import banner, get_model, show, tool_call_msg
from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel


# ---- Pattern 1: a specialist agent wrapped as a tool ---------------------------------------------------
@tool
def lookup_fact(topic: str) -> str:
    """Look up a fact in the knowledge base."""
    return {"graphs": "A graph is nodes plus edges."}.get(topic, "no fact found")


def build_supervisor():
    researcher = create_agent(
        model=get_model(
            [tool_call_msg("lookup_fact", {"topic": "graphs"}), AIMessage(content="Fact: a graph is nodes plus edges.")]
        ),
        tools=[lookup_fact],
        system_prompt="You are a researcher. Use lookup_fact, then answer in one sentence.",
        name="researcher",
    )

    @tool("research", description="Research a topic and return a one-sentence finding.")
    def call_researcher(query: str) -> str:
        result = researcher.invoke({"messages": [{"role": "user", "content": query}]})
        return str(result["messages"][-1].text)  # only the final text crosses the boundary

    return create_agent(
        model=get_model(
            [tool_call_msg("research", {"query": "graphs"}), AIMessage(content="Summary: a graph is nodes plus edges.")]
        ),
        tools=[call_researcher],
        system_prompt="You are a supervisor. Delegate research with the research tool.",
        name="supervisor",
    )


# ---- Pattern 2: agents as nodes in a custom graph ------------------------------------------------------
class Route(BaseModel):
    """Choose which specialist should answer."""

    team: Literal["billing", "technical"]


class SupportState(MessagesState):
    team: str


def specialist(name: str, reply: str):
    """A (tool-less) specialist agent; tools would work exactly the same way."""
    return create_agent(model=get_model([AIMessage(content=reply)]), tools=[], system_prompt=f"You are {name}.", name=name)


def build_router_graph(team: str):
    classifier = get_model([tool_call_msg("Route", {"team": team})]).with_structured_output(Route)

    def triage(state: SupportState) -> dict[str, Any]:
        return {"team": classifier.invoke(state["messages"]).team}

    def pick(state: SupportState) -> Literal["billing", "technical"]:
        return "billing" if state["team"] == "billing" else "technical"

    b = StateGraph(SupportState)  # ty: ignore[invalid-argument-type]
    b.add_node("triage", triage)
    b.add_node("billing", specialist("billing", "Billing here: I refunded the duplicate charge."))  # an agent, as a node
    b.add_node("technical", specialist("technical", "Tech here: please clear your cache and retry."))
    b.add_edge(START, "triage")
    b.add_conditional_edges("triage", pick)
    b.add_edge("billing", END)
    b.add_edge("technical", END)
    return b.compile()


if __name__ == "__main__":
    banner("Pattern 1: subagent as a tool")
    out = build_supervisor().invoke({"messages": "Find out what a graph is."})
    show(out["messages"])
    print("(the supervisor's history has", len(out["messages"]), "messages; the researcher's own tool loop is hidden)")

    banner("Pattern 2: agents as nodes in a StateGraph")
    graph = build_router_graph("billing")
    out = graph.invoke({"messages": "I was charged twice."})
    print("team:", out["team"])
    show(out["messages"])
    print("no duplicated user message:", sum(m.type == "human" for m in out["messages"]) == 1)

    out = build_router_graph("technical").invoke({"messages": "The app will not load."})
    print("team:", out["team"], "->", out["messages"][-1].text)

    banner("The parent graph: collapsed, then with the subgraphs expanded (xray)")
    print("collapsed:", list(build_router_graph("billing").get_graph().nodes))
    print("xray     :", list(build_router_graph("billing").get_graph(xray=True).nodes))

# EXERCISES
# 1. Give `researcher` a state key the supervisor does not have. What does `call_researcher` need to do
#    to pass it in or out? (Hint: ToolRuntime and Command, lesson 07.)
# 2. Add a third specialist, "sales", to Route and the graph. How many places must you edit?
# 3. Add a human approval node between `triage` and the specialists (learn-langgraph lesson 08).
#    Which pattern made that easy, and why would it be awkward with subagents-as-tools?

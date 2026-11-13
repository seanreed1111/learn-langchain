"""LESSON 01 - Runnables and chat models: the one interface under everything
==========================================================================
Why a LangGraph learner starts here: a LangGraph graph is "just" a LangChain Runnable.

    RUNNABLE   = anything with  invoke / stream / batch  (+ async: ainvoke / astream / abatch)
    CHAT MODEL = a Runnable that maps  messages -> AIMessage
    PROMPT     = a Runnable that maps  dict -> messages
    PARSER     = a Runnable that maps  AIMessage -> python value
    `a | b`    = a Runnable that runs a, then feeds its output to b   (LCEL, the "pipe")

Three facts that make LangGraph easier:
  1. `StateGraph(...).compile()` returns a Runnable, so graphs compose with chains and with other graphs.
  2. A node can be ANY Runnable, not only a function: `add_node("write", prompt | model | parser)`.
  3. Every Runnable takes the same `config` (tags, run_name, max_concurrency, callbacks, and for graphs
     `configurable.thread_id` and `recursion_limit`). Learn it once.

Chains (`|`) are straight lines. The moment you need a loop, a branch or a pause, use a graph (lesson 05).

Scripted models are single-use (they replay a list), so this lesson builds a fresh one per use.

Run:  python lessons/01_runnables_and_models.py
"""

from typing import Any

from common import banner, get_model
from langchain.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ---- 1) Prompt: dict in, messages out ----------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are terse."), ("user", "Explain {topic} in one sentence.")]
)


# ---- 2) A chain: prompt | model | parser -------------------------------------------------------------
def make_chain(reply: str) -> Runnable[dict[str, Any], str]:
    """`get_model` is a scripted fake offline, a real `init_chat_model(...)` model with a key."""
    return prompt | get_model([AIMessage(content=reply)]) | StrOutputParser()  # a RunnableSequence


# ---- 3) Plain functions become Runnables with RunnableLambda -----------------------------------------
shout: Runnable[str, str] = RunnableLambda(str.upper)


# ---- 4) A graph is a Runnable, and a chain can be a graph node ---------------------------------------
class DraftState(TypedDict):
    topic: str
    draft: str


def build_draft_graph(reply: str = "state plus nodes plus edges"):
    """The node is a CHAIN (not a function). It must return a state update, so the last step wraps the
    text into `{"draft": text}`. The prompt reads `topic` from the state dict and ignores other keys."""
    writer = make_chain(reply) | RunnableLambda(lambda text: {"draft": text})
    b = StateGraph(DraftState)  # ty: ignore[invalid-argument-type]
    b.add_node("write", writer)
    b.add_edge(START, "write")
    b.add_edge("write", END)
    return b.compile()


def graph_as_chain_step(reply: str = "state plus nodes plus edges") -> Runnable[Any, Any]:
    """Because a compiled graph is a Runnable, it pipes like any other step."""
    return (
        RunnableLambda(lambda topic: {"topic": topic})
        | build_draft_graph(reply)
        | RunnableLambda(lambda state: state["draft"])
    )


if __name__ == "__main__":
    banner("invoke / stream / batch")
    print("invoke ->", make_chain("A graph is state, nodes and edges.").invoke({"topic": "a graph"}))

    chunks = [c.text for c in get_model([AIMessage(content="streamed reply")]).stream("hi") if c.text]
    print("stream chunks ->", chunks)

    # `batch` runs inputs in parallel threads. A scripted model hands out replies in call order, so cap
    # concurrency at 1 here. With a real model you would normally let it run in parallel.
    two = get_model([AIMessage(content="first"), AIMessage(content="second")])
    print("batch ->", [m.text for m in two.batch(["q1", "q2"], config={"max_concurrency": 1})])

    banner("LCEL: pipe one Runnable into another")
    print("piped ->", (make_chain("hello") | shout).invoke({"topic": "x"}))

    banner("A graph is a Runnable")
    print("graph.invoke ->", build_draft_graph().invoke({"topic": "graphs"}))
    print("graph as a chain step ->", graph_as_chain_step().invoke("graphs"))
    print(build_draft_graph().get_graph().draw_mermaid())

# EXERCISES
# 1. Add `config={"run_name": "my_chain", "tags": ["lesson01"]}` to a `chain.invoke` call (no visible
#    change offline; with LANGSMITH_TRACING=true they show up on the trace).
# 2. Pipe `build_draft_graph()` into a second graph node that title-cases the draft.
# 3. Put an ANTHROPIC_API_KEY in `.env` and compare `.invoke("hi").usage_metadata` with the scripted
#    model's (which is None).

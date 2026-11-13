"""LESSON 07 - State, context and memory in an agent
==================================================
Four different places data can live. Mixing them up is the most common agent design mistake.

  KIND                 LIVES IN                      LIFETIME            WRITABLE BY TOOLS?
  -------------------  ----------------------------  ------------------  -------------------
  State (AgentState)   the graph state               one thread          yes, via `Command`
  Context              `context=` on invoke          one run             no (read-only)
  Short-term memory    the checkpointer (per thread) one thread, saved   (it IS the state)
  Long-term memory     the Store (namespaced keys)   across threads      yes, via runtime.store

How each is wired in `create_agent`:
  * `state_schema=MyState`   with `class MyState(AgentState): user_name: str`  - extra state keys
  * `context_schema=Ctx`     a dataclass (or Pydantic model); pass `context=Ctx(...)` to invoke
  * `checkpointer=...`       plus `config={"configurable": {"thread_id": "..."}}` on every call
  * `store=...`              any LangGraph BaseStore

Tools reach all of it through ONE injected parameter, `runtime: ToolRuntime[Ctx, State]`:
    runtime.state          current state (including messages)
    runtime.context        the per-run context object
    runtime.store          the long-term store
    runtime.tool_call_id   id to put on the ToolMessage you return
    runtime.stream_writer  emit custom stream events (lesson 11)
`runtime` is hidden from the model's schema, so the model cannot see or forge it. Do not name your own
tool argument `runtime` or `config`.

Identity belongs in CONTEXT (the app sets it), never in a tool argument the model could fill in.

Run:  python lessons/07_state_context_and_memory.py
"""

from dataclasses import dataclass
from typing import Any

from common import banner, get_model, tool_call_msg
from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

USERS = {"u1": "Ada Lovelace", "u2": "Grace Hopper"}


@dataclass
class Ctx:
    """Per-run, read-only. The application sets it; the model never sees it."""

    user_id: str


class ProfileState(AgentState):
    """AgentState plus our own key. State is saved by the checkpointer and visible to every node."""

    user_name: str


# ---- A) Tools that read context and state, and WRITE state with Command -------------------------------
@tool
def lookup_user(runtime: ToolRuntime[Ctx, ProfileState]) -> Command[Any]:
    """Look up the current user's name and remember it."""
    name = USERS.get(runtime.context.user_id, "stranger")  # identity comes from CONTEXT
    return Command(
        update={
            "user_name": name,  # writes our custom state key
            # Returning Command REPLACES the normal tool result, so you must add the ToolMessage yourself,
            # linked by tool_call_id. Forget it and the next model call is invalid.
            "messages": [ToolMessage(f"Found user {name}.", tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def greet(runtime: ToolRuntime[Ctx, ProfileState]) -> str:
    """Greet the user by name. Call lookup_user first."""
    name = runtime.state.get("user_name")  # reads STATE written by the other tool
    return f"Hello, {name}!" if name else "I don't know your name yet."


def profile_agent():
    script = [
        tool_call_msg("lookup_user", {}, "c1"),
        tool_call_msg("greet", {}, "c2"),
        AIMessage(content="Greeted the user."),
    ]
    return create_agent(
        model=get_model(script),
        tools=[lookup_user, greet],
        state_schema=ProfileState,
        context_schema=Ctx,
    )


# ---- B) Short-term memory: checkpointer + thread_id -------------------------------------------------
def memory_agent():
    """A scripted model does not literally remember (it replays a script). What this demonstrates is the
    STATE: the checkpointer stores the whole message list per thread, and a real model reads it."""
    script = [AIMessage(content="Nice to meet you, Bob."), AIMessage(content="Your name is Bob.")]
    return create_agent(model=get_model(script), tools=[], checkpointer=InMemorySaver())


# ---- C) Long-term memory: the Store (shared across threads) -----------------------------------------
@tool
def remember_color(color: str, runtime: ToolRuntime[Ctx]) -> str:
    """Remember the user's favorite color."""
    assert runtime.store is not None
    runtime.store.put(("prefs", runtime.context.user_id), "color", {"value": color})
    return f"Saved {color}."


@tool
def recall_color(runtime: ToolRuntime[Ctx]) -> str:
    """Recall the user's favorite color."""
    assert runtime.store is not None
    item = runtime.store.get(("prefs", runtime.context.user_id), "color")
    return str(item.value["value"]) if item else "unknown"


def store_agent(store: InMemoryStore):
    script = [
        tool_call_msg("remember_color", {"color": "green"}, "c1"),
        AIMessage(content="Got it."),
        tool_call_msg("recall_color", {}, "c2"),  # a DIFFERENT thread, later
        AIMessage(content="Your favorite color is green."),
    ]
    return create_agent(
        model=get_model(script),
        tools=[remember_color, recall_color],
        context_schema=Ctx,
        checkpointer=InMemorySaver(),
        store=store,
    )


def thread(tid: str) -> Any:
    return {"configurable": {"thread_id": tid}}


if __name__ == "__main__":
    banner("A) Context in, state out (Command)")
    out = profile_agent().invoke({"messages": "greet me"}, context=Ctx(user_id="u1"))
    print("user_name in state:", out["user_name"])
    print("greet tool said   :", [m.text for m in out["messages"] if isinstance(m, ToolMessage)][-1])

    banner("B) Short-term memory: one thread keeps history, another does not")
    agent = memory_agent()
    agent.invoke({"messages": "Hi, I'm Bob."}, thread("t1"))
    agent.invoke({"messages": "What's my name?"}, thread("t1"))
    saved = agent.get_state(thread("t1")).values["messages"]
    print("thread t1 messages:", [(type(m).__name__, m.text) for m in saved])
    print("thread t2 messages:", agent.get_state(thread("t2")).values.get("messages", []))

    banner("C) Long-term memory: the Store crosses threads")
    store = InMemoryStore()
    agent = store_agent(store)
    agent.invoke({"messages": "My favorite color is green."}, thread("a"), context=Ctx("u1"))
    agent.invoke({"messages": "What's my favorite color?"}, thread("b"), context=Ctx("u1"))
    item = store.get(("prefs", "u1"), "color")
    assert item is not None
    print("store item:", item.value)
    print("another user:", store.get(("prefs", "u2"), "color"))

# EXERCISES
# 1. In (A), make `lookup_user` return a plain string instead of a Command. Where does `user_name` go?
# 2. In (A), drop the ToolMessage from the Command's `messages` and read the ValueError. Why does
#    LangChain refuse to continue instead of letting the next model call fail at the provider?
# 3. In (B), swap InMemorySaver for SqliteSaver (pip install langgraph-checkpoint-sqlite), restart the
#    script between turns, and show memory survives a restart.

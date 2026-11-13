"""LESSON 02 - Messages: the data that flows through every agent
=============================================================
LangGraph state for chat agents is a list of LangChain messages. If you can read a message list, you
can read any agent trace. The four types you will meet constantly:

    SystemMessage   instructions for the model (not shown to the user)
    HumanMessage    what the user said
    AIMessage       what the model said; carries `.tool_calls` when it wants a tool run
    ToolMessage     the result of one tool call; linked back by `tool_call_id`

Why this matters for LangGraph:
  * `MessagesState` and `AgentState` both declare `messages: Annotated[list, add_messages]`.
    `add_messages` is the REDUCER that makes a node's returned messages APPEND (lesson 02 of
    learn-langgraph covers reducers). It also REPLACES a message whose `id` already exists, and it
    deletes one when you return `RemoveMessage(id=...)`.
  * Dicts like {"role": "user", "content": "hi"} are accepted anywhere and coerced to message objects.
  * `.text` is the plain string; `.content_blocks` is the provider-neutral structured form.

Run:  python lessons/02_messages.py
"""

from typing import Any, cast

from common import banner, show
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ---- 1) The four message types and how a tool call is linked to its result ---------------------------
conversation: list[AnyMessage] = [
    SystemMessage("You are a calculator assistant."),
    HumanMessage("What is 6 times 7?"),
    # The model does not run tools. It EMITS a request; your code (a ToolNode) runs it.
    AIMessage(content="", tool_calls=[{"name": "multiply", "args": {"a": 6, "b": 7}, "id": "call_1"}]),
    # `tool_call_id` must match the request's id, or the provider rejects the next model call.
    ToolMessage(content="42", tool_call_id="call_1", name="multiply"),
    AIMessage(content="6 times 7 is 42."),
]


# ---- 2) add_messages: the reducer behind `messages` -------------------------------------------------
def reducer_demo() -> dict[str, list[BaseMessage]]:
    history: list[BaseMessage] = [HumanMessage("hi", id="m1"), AIMessage("hello", id="m2")]

    def merge(new: list[Any]) -> list[BaseMessage]:
        merged: Any = add_messages(cast("Any", history), new)
        assert isinstance(merged, list)
        return cast("list[BaseMessage]", merged)

    return {
        "appended": merge([HumanMessage("new question", id="m3")]),  # new id -> appended
        "replaced": merge([AIMessage("hello, edited", id="m2")]),  # same id -> replaced in place
        "removed": merge([RemoveMessage(id="m1")]),  # RemoveMessage -> deleted
        "coerced": merge([{"role": "user", "content": "from a dict"}]),  # dict -> HumanMessage
    }


# ---- 3) trim_messages: keep the context window bounded ----------------------------------------------
def trim_demo() -> list[BaseMessage]:
    long_history: list[BaseMessage] = [SystemMessage("be brief")]
    for i in range(1, 5):
        long_history += [HumanMessage(f"question {i}"), AIMessage(f"answer {i}")]
    # Keep the system message and the last 3 messages. `token_counter=len` counts MESSAGES, not tokens;
    # with a real model pass the model itself (or a token-counting function) instead.
    return trim_messages(
        long_history,
        strategy="last",
        max_tokens=3,
        token_counter=len,
        include_system=True,
        start_on="human",
    )


def message_facts() -> dict[str, Any]:
    ai = conversation[2]
    assert isinstance(ai, AIMessage)
    return {
        "types": [type(m).__name__ for m in conversation],
        "tool_call": ai.tool_calls[0],
        "linked": isinstance(conversation[3], ToolMessage) and conversation[3].tool_call_id == ai.tool_calls[0]["id"],
        "text": conversation[-1].text,
        "block_types": [b["type"] for b in conversation[-1].content_blocks],
    }


if __name__ == "__main__":
    banner("A conversation with one tool call")
    show(conversation)
    facts = message_facts()
    print("tool_call:", facts["tool_call"])
    print("linked by id:", facts["linked"])
    print("content_blocks types:", facts["block_types"])

    banner("add_messages: append, replace, remove, coerce")
    for name, msgs in reducer_demo().items():
        print(f"{name}:", [f"{type(m).__name__}({m.text!r})" for m in msgs])

    banner("trim_messages keeps the system message + the tail")
    show(trim_demo())

    banner("Dicts are accepted as messages")
    print(add_messages([], [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi!"}]))

# EXERCISES
# 1. Build a ToolMessage with a WRONG tool_call_id and (with an API key) send the list to a real model.
#    Read the provider error. This is the most common hand-built-agent bug.
# 2. Use `add_messages` to replace the AIMessage "6 times 7 is 42." with a corrected one by reusing its id.
# 3. Change `max_tokens` in `trim_demo` and watch which messages survive. Why does start_on="human" matter?

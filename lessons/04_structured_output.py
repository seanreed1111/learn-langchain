"""LESSON 04 - Structured output: typed model answers you can route on
===================================================================
LangGraph routes on STATE, and state wants typed values, not prose. Structured output asks the model
for data that matches a schema (Pydantic model, dataclass or TypedDict), and returns a validated
object instead of text.

    model.with_structured_output(Schema)          -> a Runnable returning a Schema instance
    model.with_structured_output(S, include_raw=True) -> {"raw": AIMessage, "parsed": ..., "parsing_error": ...}
    create_agent(..., response_format=Schema)     -> result["structured_response"]  (lesson 06)

Under the hood most models do this with a forced TOOL CALL whose arguments are the schema. That is why
the same `tool_calls` machinery from lesson 03 shows up here, and why our scripted model can fake it:
the script just contains an AIMessage that "calls" a tool named after the schema.

The pattern you will use constantly in LangGraph: a CLASSIFIER NODE that returns a `Literal`, followed
by a conditional edge that reads it (learn-langgraph lesson 03). Field descriptions are prompts too.

Run:  python lessons/04_structured_output.py
"""

from typing import Any, Literal

from common import banner, get_model, tool_call_msg
from langchain.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---- 1) A schema. The class docstring and Field descriptions are sent to the model ---------------------
class Triage(BaseModel):
    """Classify a customer message."""

    category: Literal["billing", "technical", "other"] = Field(description="Which team should handle it")
    urgency: int = Field(ge=1, le=5, description="1 = whenever, 5 = on fire")
    summary: str = Field(description="One short sentence")


def triage_message(category: str, urgency: int, summary: str, call_id: str = "t1") -> AIMessage:
    """What a real model emits for `Triage`: a tool call named after the schema."""
    return tool_call_msg("Triage", {"category": category, "urgency": urgency, "summary": summary}, call_id)


# ---- 2) with_structured_output -----------------------------------------------------------------------
def extract(text: str) -> Triage:
    model = get_model([triage_message("billing", 4, "Charged twice for one order")])
    result = model.with_structured_output(Triage).invoke(text)  # a validated Triage instance
    assert isinstance(result, Triage)
    return result


def extract_with_raw(text: str) -> dict[str, Any]:
    model = get_model([triage_message("technical", 2, "App crashes on login")])
    return model.with_structured_output(Triage, include_raw=True).invoke(text)


# ---- 3) The LangGraph pattern: classifier node -> conditional edge -> specialist node -----------------
class TicketState(TypedDict):
    text: str
    triage: Triage | None
    handled_by: str


def make_triage_graph(script: list[AIMessage]):
    classifier = get_model(script).with_structured_output(Triage)

    def classify(state: TicketState) -> dict[str, Any]:
        return {"triage": classifier.invoke([HumanMessage(state["text"])])}

    def route(state: TicketState) -> Literal["billing_team", "tech_team", "general_team"]:
        triage = state["triage"]
        assert triage is not None
        if triage.category == "billing":
            return "billing_team"
        return "tech_team" if triage.category == "technical" else "general_team"

    def team(name: str):
        def run(state: TicketState) -> dict[str, Any]:
            triage = state["triage"]
            assert triage is not None
            return {"handled_by": f"{name} (urgency {triage.urgency})"}

        return run

    b = StateGraph(TicketState)  # ty: ignore[invalid-argument-type]
    b.add_node("classify", classify)
    for node in ("billing_team", "tech_team", "general_team"):
        b.add_node(node, team(node))
        b.add_edge(node, END)
    b.add_edge(START, "classify")
    b.add_conditional_edges("classify", route)  # reads the typed field, not prose
    return b.compile()


if __name__ == "__main__":
    banner("The schema the model sees")
    print(Triage.model_json_schema()["properties"]["category"])

    banner("with_structured_output -> a validated object")
    t = extract("I was charged twice!")
    print(type(t).__name__, t)

    banner("include_raw=True keeps the AIMessage (token counts, tool call id) next to the parsed value")
    r = extract_with_raw("The app crashes when I log in")
    print("keys:", sorted(r))
    print("parsed:", r["parsed"])
    print("parsing_error:", r["parsing_error"])

    banner("Classifier node + conditional edge")
    g = make_triage_graph([triage_message("billing", 4, "Double charge")])
    print(g.invoke({"text": "I was charged twice!", "triage": None, "handled_by": ""})["handled_by"])
    g = make_triage_graph([triage_message("technical", 2, "Crash on login")])
    print(g.invoke({"text": "App crashes", "triage": None, "handled_by": ""})["handled_by"])

# EXERCISES
# 1. Make the script return urgency=9 (outside ge=1, le=5) and catch the validation error. What would a
#    retry loop in a graph look like? (learn-langgraph lesson 03 has the loop pattern.)
# 2. Add "sales" to the `Literal`, then run a script that returns "sales". `route` quietly sends it to
#    `general_team`. Is that what you want? Change `route` so an unknown category fails loudly, and say
#    when a silent fallback is the better design.
# 3. Switch Triage to a TypedDict. What does `with_structured_output` return now: dict or object?

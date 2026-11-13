"""Agent evals as ordinary pytest tests, tracked by LangSmith when you opt in (lesson 16).

`@pytest.mark.langsmith` makes each parametrized case a row in a LangSmith experiment. tests/conftest.py sets
LANGSMITH_TEST_TRACKING=false, so offline these are plain tests: the `t.log_*` calls do nothing. To see them in
LangSmith, run with a key and tracking on:

    $env:LANGSMITH_API_KEY = "..."; $env:LANGSMITH_TEST_TRACKING = "true"; pytest tests/test_agent_evals.py
"""

import importlib
from typing import Any

import pytest
from langsmith import testing as t

lesson15 = importlib.import_module("15_evaluation")

CASES = [(q, a, tools) for q, a, tools in lesson15.CASES]


@pytest.mark.langsmith
@pytest.mark.parametrize(("question", "answer", "tools"), CASES)
def test_good_agent_answers_and_uses_the_right_tools(question: str, answer: str, tools: list[str]):
    outputs: dict[str, Any] = lesson15.make_target("good")({"question": question})
    reference = {"answer": answer, "tools": tools}

    t.log_inputs({"question": question})
    t.log_outputs({"answer": outputs["answer"], "tools": outputs["tools"]})
    t.log_reference_outputs(reference)

    final = lesson15.final_answer_correct(outputs, reference)
    trajectory = lesson15.tools_exact(outputs, reference)
    t.log_feedback(key=final["key"], score=final["score"])
    t.log_feedback(key=trajectory["key"], score=trajectory["score"])

    assert final["score"] == 1.0
    assert trajectory["score"] == 1.0


@pytest.mark.langsmith
def test_a_regressed_agent_is_caught_by_the_same_evaluators():
    question, answer, tools = CASES[1]
    outputs = lesson15.make_target("regressed")({"question": question})
    reference = {"answer": answer, "tools": tools}
    assert lesson15.final_answer_correct(outputs, reference)["score"] == 0.0
    assert lesson15.tools_exact(outputs, reference)["score"] == 0.0

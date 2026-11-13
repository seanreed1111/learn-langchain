"""Lessons 13-17 (LangSmith): every claim they make, asserted against captured payloads. Offline.

The tracing tests use `LocalLangSmith` (lessons/ls_offline.py): a LangSmith client whose HTTP session records
requests instead of sending them, so we can assert on exactly what WOULD be uploaded.
"""

import asyncio
import importlib
import json
import warnings
from typing import Any

import langsmith as ls
import pytest
from langsmith.utils import LangSmithWarning
from ls_offline import LocalLangSmith


def load(name: str) -> Any:
    return importlib.import_module(name)


QUESTION = "What is 6 times 7, plus 8?"


# ---------------------------------------------------------------- offline helper
def test_local_langsmith_records_and_never_sends():
    m = load("13_tracing")
    local = m.trace_of(m.handle_request, QUESTION)
    assert local.runs(), "expected captured runs"
    assert all(url.startswith("http://langsmith.invalid") for _, url, _ in local.session.calls)


def test_local_langsmith_answers_project_lookups():
    local = LocalLangSmith()
    project = local.client.read_project(project_name="demo")
    assert str(project.id) == local.session.project_id and project.name == "demo"


# ---------------------------------------------------------------- 13 tracing
def test_13_trace_is_the_graph_you_built():
    m = load("13_tracing")
    local = m.trace_of(m.handle_request, QUESTION)
    assert {"handle_request", "LangGraph", "model", "tools", "multiply", "add", "ScriptedChatModel"} <= set(local.names())
    by_name = {r["name"]: r for r in local.runs()}
    assert by_name["model"]["run_type"] == "chain" and by_name["multiply"]["run_type"] == "tool"
    assert by_name["ScriptedChatModel"]["run_type"] == "llm"
    assert by_name["LangGraph"]["parent_run_id"] == by_name["handle_request"]["id"]


def test_13_tags_metadata_project_and_run_name():
    m = load("13_tracing")
    local = m.trace_of(m.handle_request, QUESTION, project="proj-x")
    root = m.run_by_name(local, "handle_request")
    assert root["tags"] == ["api"] and root["extra"]["metadata"]["route"] == "/chat"
    assert root["session_name"] == "proj-x"
    model = m.run_by_name(local, "model")
    assert "lesson13" in model["tags"] and model["extra"]["metadata"]["user_id"] == "u1"  # config reaches children
    assert model["extra"]["metadata"]["langgraph_node"] == "model"
    assert root["inputs"] == {"question": QUESTION} and root["outputs"] == {"output": "6 x 7 + 8 = 50."}


def test_13_run_name_renames_only_the_root():
    m = load("13_tracing")
    local = LocalLangSmith()
    with ls.tracing_context(enabled=True, client=local.client):
        m.lesson05.build_agent().invoke({"messages": "x"}, {"run_name": "calc"})
    assert local.names()[0] == "calc" and "model" in local.names()


def test_13_traceable_helper_inside_a_tool_is_a_child_of_the_tool_run():
    m = load("13_tracing")
    local = m.trace_of(m.priced_agent().invoke, {"messages": "price of A1?"})
    helper = m.run_by_name(local, "price_lookup")
    assert helper["run_type"] == "retriever" and helper["parent_run_id"] == m.run_by_name(local, "get_price")["id"]


def test_13_enabled_false_wins_even_if_the_environment_says_true(monkeypatch: pytest.MonkeyPatch):
    m = load("13_tracing")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    off = LocalLangSmith()
    with ls.tracing_context(enabled=False, client=off.client):
        m.handle_request("hi")
    assert off.runs() == []


# ---------------------------------------------------------------- 14 trace hygiene
def test_14_anonymizer_scrubs_inputs_outputs_tool_args_and_metadata():
    m = load("14_trace_hygiene")
    local = m.anonymizing_client()
    m.run_agent(local, m.SECRET_QUESTION, {"metadata": {"requester": "ada@example.com"}})
    assert not m.leaked(local, "ada@example.com") and not m.leaked(local, "ORD-12345")
    root = local.runs()[0]
    assert "<email>" in root["inputs"]["messages"] and "<order>" in root["inputs"]["messages"]
    assert root["extra"]["metadata"]["requester"] == "<email>"


def test_14_gotcha_names_and_tags_are_not_scrubbed():
    m = load("14_trace_hygiene")
    local = m.anonymizing_client()
    m.run_agent(local, "hi", {"run_name": "run-for-ada@example.com", "tags": ["ada@example.com"]})
    assert local.runs()[0]["name"] == "run-for-ada@example.com"
    assert local.runs()[0]["tags"] == ["ada@example.com"]
    assert m.leaked(local, "ada@example.com")


def test_14_process_inputs_and_hide_inputs():
    m = load("14_trace_hygiene")
    local = LocalLangSmith()
    with ls.tracing_context(enabled=True, client=local.client):
        m.summarize("a long confidential document")
    run = local.runs()[0]
    assert run["inputs"] == {"text_length": 28} and run["outputs"] == {"summary": "redacted", "ok": True}
    assert not m.leaked(local, "confidential")

    hidden = LocalLangSmith(hide_inputs=lambda _inputs: {}, hide_outputs=lambda _outputs: {})
    m.run_agent(hidden, m.SECRET_QUESTION)
    assert hidden.runs()[0]["inputs"] == {} and hidden.runs()[0]["outputs"] == {}
    assert not m.leaked(hidden, "ORD-12345")


def test_14_sampling_zero_one_and_whole_traces_only():
    m = load("14_trace_hygiene")
    none = LocalLangSmith(tracing_sampling_rate=0.0)
    m.run_agent(none, "hi")
    assert none.runs() == []
    everything = LocalLangSmith(tracing_sampling_rate=1.0)
    m.run_agent(everything, "hi")
    full = len(everything.runs())
    assert full > 0
    half = LocalLangSmith(tracing_sampling_rate=0.5)
    with ls.tracing_context(enabled=True, client=half.client):
        for _ in range(12):
            m.lesson05.build_agent().invoke({"messages": "hi"})
    per_trace: dict[str, int] = {}
    for r in half.runs():
        per_trace[r["trace_id"]] = per_trace.get(r["trace_id"], 0) + 1
    assert set(per_trace.values()) <= {full}, "sampling must keep or drop WHOLE traces"


def test_14_conditional_tracing_by_tenant():
    m = load("14_trace_hygiene")
    traced, untraced = LocalLangSmith(), LocalLangSmith()
    m.handle(traced, "internal", "hi")
    m.handle(untraced, "acme-corp", "hi")
    assert traced.runs() and untraced.runs() == []


def test_14_thread_id_reaches_every_run():
    m = load("14_trace_hygiene")
    local = LocalLangSmith()
    agent = m.conversation_agent()
    with ls.tracing_context(enabled=True, client=local.client):
        for turn in ("Hi, I'm Ada.", "What's my name?"):
            agent.invoke({"messages": turn}, {"configurable": {"thread_id": "T-1"}})
    assert {r["extra"]["metadata"].get("thread_id") for r in local.runs()} == {"T-1"}
    assert len([r for r in local.runs() if r.get("parent_run_id") is None]) == 2


# ---------------------------------------------------------------- 15 evaluation
def test_15_good_variant_scores_perfectly_and_regression_is_caught():
    m = load("15_evaluation")
    good, bad = m.run_experiment("good"), m.run_experiment("regressed")
    assert {"final_answer", "tools_exact", "tools_superset"} <= set(good)
    assert all(score == 1.0 for score in good.values())
    assert bad["final_answer"] == pytest.approx(1 / 3) and bad["tools_exact"] == pytest.approx(1 / 3)
    assert {"final_answer", "tools_exact"} <= set(m.regressions(good, bad))


def test_15_trajectory_evaluators_catch_a_right_answer_by_the_wrong_route():
    m = load("15_evaluation")
    lucky = {"answer": "6 x 7 = 42.", "tools": [], "messages": []}
    reference = {"answer": "42", "tools": ["multiply"]}
    assert m.final_answer_correct(lucky, reference)["score"] == 1.0
    assert m.tools_exact(lucky, reference)["score"] == 0.0
    extra = {"answer": "42", "tools": ["multiply", "add"], "messages": []}
    assert m.tools_exact(extra, reference)["score"] == 0.0 and m.tools_superset(extra, reference)["score"] == 1.0


def test_15_agentevals_unordered_matches_when_installed():
    pytest.importorskip("agentevals")
    m = load("15_evaluation")
    assert m.run_experiment("good")["agentevals_unordered"] == 1.0
    assert m.run_experiment("regressed")["agentevals_unordered"] == pytest.approx(1 / 3)


def test_15_judge_is_just_structured_output():
    m = load("15_evaluation")
    from common import get_model, tool_call_msg

    judge = m.make_judge(get_model([tool_call_msg("Verdict", {"helpful": False, "reason": "wrong"})]))
    assert judge({"answer": "48"}, {"answer": "50"}) == {"key": "judge_helpful", "score": 0.0, "comment": "wrong"}


def test_15_dataset_examples_are_well_formed():
    m = load("15_evaluation")
    examples = m.make_examples()
    assert len(examples) == len(m.CASES) and len({e.id for e in examples}) == len(examples)
    assert examples[1].outputs == {"answer": "50", "tools": ["multiply", "add"]}


# ---------------------------------------------------------------- 16 CI gates and feedback
def test_16_gates_and_exit_codes():
    m = load("16_evals_in_ci_and_feedback")
    assert m.check_gates({"final_answer": 1.0, "tools_exact": 1.0}) == []
    failures = m.check_gates({"final_answer": 0.5})
    assert any("final_answer" in f for f in failures) and any("no score produced" in f for f in failures)
    assert m.gate_variant("good") == 0
    assert m.gate_variant("regressed") == 1


def test_16_feedback_attaches_to_the_chosen_run_with_the_project_id():
    m = load("16_evals_in_ci_and_feedback")
    local = LocalLangSmith()
    with warnings.catch_warnings():
        warnings.simplefilter("error", LangSmithWarning)  # the deprecated no-session_id form would raise here
        text, run_id = m.answer_with_feedback(local, m.lesson15.CASES[1][0], thumbs_up=True)
    assert text == "6 x 7 + 8 = 50."
    root = next(r for r in local.runs() if r.get("parent_run_id") is None)
    feedback = local.requests_to("/feedback")[0]
    assert root["id"] == str(run_id) == feedback["run_id"]
    assert feedback["session_id"] == local.session.project_id
    assert (feedback["key"], feedback["score"]) == ("user_score", 1.0)


# ---------------------------------------------------------------- 17 studio and the agent server
def test_17_langgraph_json_points_at_a_reusable_compiled_graph():
    m = load("17_studio_and_agent_server")
    config = m.load_config()
    _module, graph = m.import_graph(config["graphs"]["helpdesk"])
    assert type(graph).__name__ == "CompiledStateGraph"
    assert graph.checkpointer is None, "the server supplies persistence; do not compile one in"
    assert "HumanInTheLoopMiddleware.after_model" in graph.get_graph().nodes


@pytest.mark.skipif(not load("17_studio_and_agent_server").server_available(), reason='pip install -e ".[studio]"')
def test_17_real_dev_server_pauses_and_resumes():
    m = load("17_studio_and_agent_server")
    with m.dev_server() as url:
        result = asyncio.run(m.drive(url))
    assert result["graphs"] == ["helpdesk"]
    assert result["paused_at"] == ["HumanInTheLoopMiddleware.after_model"]
    assert result["request"] == ("issue_refund", {"order_id": "123", "amount": 79.0})
    assert result["final"].startswith("All done") and result["checkpoints"] >= 6


def test_17_dev_server_stops_even_if_the_body_raises():
    m = load("17_studio_and_agent_server")
    if not m.server_available():
        pytest.skip('pip install -e ".[studio]"')
    with pytest.raises(OSError, match="boom"), m.dev_server():
        raise OSError("boom")  # must propagate (a `suppress(OSError)` around the yield would hide it)


def test_json_payloads_are_plain_json():
    m = load("13_tracing")
    local = m.trace_of(m.handle_request, "hi")
    json.dumps(local.runs(), default=str)  # would raise if a payload held a non-serializable object

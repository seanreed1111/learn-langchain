"""LESSON 15 - Evaluation: datasets, evaluators and experiments
============================================================
Tracing tells you WHAT happened. Evaluation tells you whether it was GOOD, and whether a change made it
better or worse. LangSmith's offline evaluation has four nouns:

    DATASET      examples: `inputs` to send, plus `outputs` = the reference you expect
    TARGET       a function `inputs -> outputs` that runs YOUR agent
    EVALUATORS   functions `(outputs, reference_outputs) -> {"key": ..., "score": ...}`
    EXPERIMENT   one pass of `evaluate(target, data, evaluators)`; compare experiments across changes

Three things worth scoring for an agent (cheap to expensive):
    1. FINAL ANSWER       is the last message right?                 (exact match, contains, regex)
    2. TRAJECTORY         did it use the right tools, in a sane way?  (compare the tool calls)
    3. LLM-AS-JUDGE       is it helpful, grounded, polite?            (a model grades; needs its own checks)
Trajectory checks catch what answer checks miss: a lucky right answer reached by skipping the tool.

For real use you upload a dataset once and let LangSmith store experiments:
    client = Client()
    client.create_dataset("calc-evals");  client.create_examples(dataset_name="calc-evals", examples=[...])
    evaluate(target, data="calc-evals", evaluators=[...], experiment_prefix="gpt-vs-claude",
             metadata={"variant": "v2"})                    # then compare experiments in the UI
OFFLINE we pass `Example` objects and `upload_results=False`, so nothing is created on a server (and our
recording client captures the traces locally). The functions and the scores are the same either way.

Honesty note: offline, the model is SCRIPTED, so these evals test the EVAL HARNESS (does it catch a
regression?) not the model. Re-run with a real model and a real dataset to measure the model.

Run:  python lessons/15_evaluation.py
"""

import datetime
import importlib
import importlib.util
import uuid
import warnings
from typing import Any

from common import banner, tool_call_msg
from langchain.messages import AIMessage, HumanMessage
from langsmith import evaluate
from langsmith.schemas import Example
from ls_offline import LocalLangSmith
from pydantic import BaseModel

lesson05 = importlib.import_module("05_model_and_tools_in_a_graph")  # the calculator agent under test

# (question, expected answer, expected tool calls in order)
CASES = [
    ("What is 6 times 7?", "42", ["multiply"]),
    ("What is 6 times 7, plus 8?", "50", ["multiply", "add"]),
    ("What is 2 plus 3?", "5", ["add"]),
]


def make_examples() -> list[Example]:
    dataset_id = uuid.uuid4()
    now = datetime.datetime.now(datetime.UTC)
    return [
        Example(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            created_at=now,
            inputs={"question": q},
            outputs={"answer": a, "tools": tools},
        )
        for q, a, tools in CASES
    ]


# ---- The agent under test, in two variants ------------------------------------------------------------
def script_for(question: str, variant: str) -> list[AIMessage]:
    """What the (scripted) model does. `regressed` simulates a bad prompt change: it stops using tools
    for the harder questions and guesses."""
    good: dict[str, list[AIMessage]] = {
        CASES[0][0]: [tool_call_msg("multiply", {"a": 6, "b": 7}), AIMessage(content="6 x 7 = 42.")],
        CASES[1][0]: [
            tool_call_msg("multiply", {"a": 6, "b": 7}, "c1"),
            tool_call_msg("add", {"a": 42, "b": 8}, "c2"),
            AIMessage(content="6 x 7 + 8 = 50."),
        ],
        CASES[2][0]: [tool_call_msg("add", {"a": 2, "b": 3}), AIMessage(content="2 + 3 = 5.")],
    }
    if variant == "good":
        return good[question]
    guesses = {CASES[1][0]: "6 x 7 + 8 = 48.", CASES[2][0]: "2 + 3 = 6."}
    return good[question] if question == CASES[0][0] else [AIMessage(content=guesses[question])]


def make_target(variant: str):
    """The TARGET: runs the agent for one example and returns what the evaluators need."""

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        agent = lesson05.build_agent(script_for(question, variant))
        messages = agent.invoke({"messages": question})["messages"]
        tools = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        return {"answer": messages[-1].text, "tools": tools, "messages": messages}

    return target


# ---- Evaluators: plain functions returning {"key", "score"} -------------------------------------------
def final_answer_correct(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    return {"key": "final_answer", "score": float(reference_outputs["answer"] in outputs["answer"])}


def tools_exact(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """Strict trajectory: exactly the expected tools, in the expected order."""
    return {"key": "tools_exact", "score": float(outputs["tools"] == reference_outputs["tools"])}


def tools_superset(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """Looser: at least the expected tools were used (extra calls are allowed)."""
    return {"key": "tools_superset", "score": float(set(reference_outputs["tools"]) <= set(outputs["tools"]))}


def agentevals_unordered(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """The same idea with the `agentevals` package (pip install -e ".[evals]"): it compares message
    trajectories and supports strict / unordered / subset / superset modes."""
    from agentevals.trajectory.match import create_trajectory_match_evaluator

    reference = [HumanMessage("x")] + [
        AIMessage(content="", tool_calls=[{"id": f"ref{i}", "name": name, "args": {}}])
        for i, name in enumerate(reference_outputs["tools"])
    ]
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="unordered", tool_args_match_mode="ignore")
    result: Any = evaluator(outputs=outputs["messages"], reference_outputs=reference)
    return {"key": "agentevals_unordered", "score": float(bool(result["score"]))}


class Verdict(BaseModel):
    """A judge's verdict."""

    helpful: bool
    reason: str


def make_judge(judge_model: Any):
    """LLM-as-judge: a model grades the answer against a rubric using structured output (lesson 04).
    Judges are models too: validate them against human labels before trusting their scores."""
    structured = judge_model.with_structured_output(Verdict)

    def judge(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        rubric = f"Is this answer correct and concise?\nExpected: {reference_outputs['answer']}\nAnswer: {outputs['answer']}"
        verdict = structured.invoke(rubric)
        return {"key": "judge_helpful", "score": float(verdict.helpful), "comment": verdict.reason}

    return judge


def evaluators() -> list[Any]:
    base = [final_answer_correct, tools_exact, tools_superset]
    return [*base, agentevals_unordered] if importlib.util.find_spec("agentevals") else base


# ---- Running an experiment and reading it --------------------------------------------------------------
def run_experiment(variant: str) -> dict[str, float]:
    """Run the whole dataset through `variant`; return the mean score per evaluator key."""
    local = LocalLangSmith()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # `upload_results` is flagged beta by LangSmith
        results = evaluate(
            make_target(variant),
            data=make_examples(),
            evaluators=evaluators(),
            upload_results=False,  # nothing is created on a server
            client=local.client,  # and traces go to our recording client, not the network
            max_concurrency=0,
        )
    scores: dict[str, list[float]] = {}
    for row in results:
        for result in row["evaluation_results"]["results"]:
            scores.setdefault(result.key, []).append(float(result.score or 0.0))
    return {key: sum(vals) / len(vals) for key, vals in scores.items()}


def regressions(baseline: dict[str, float], candidate: dict[str, float], tolerance: float = 0.0) -> list[str]:
    """Metrics where the candidate is worse than the baseline by more than `tolerance`."""
    return [k for k in baseline if candidate.get(k, 0.0) < baseline[k] - tolerance]


if __name__ == "__main__":
    banner("The dataset")
    for ex in make_examples():
        assert ex.inputs is not None
        print(f"  {ex.inputs['question']!r:<32} -> {ex.outputs}")

    banner("One experiment per variant (mean score per evaluator)")
    base = run_experiment("good")
    cand = run_experiment("regressed")
    print(f"  {'evaluator':<22}{'good':>8}{'regressed':>12}")
    for key in base:
        print(f"  {key:<22}{base[key]:>8.2f}{cand[key]:>12.2f}")

    banner("Regressions the evals caught")
    print("  ", regressions(base, cand))

    banner("Why trajectory checks matter: a right answer by the wrong route")
    lucky = {"answer": "6 x 7 = 42.", "tools": [], "messages": []}
    ref = {"answer": "42", "tools": ["multiply"]}
    print("  final_answer :", final_answer_correct(lucky, ref)["score"], "(looks fine)")
    print("  tools_exact  :", tools_exact(lucky, ref)["score"], "(it never used the tool)")

    banner("LLM-as-judge wiring (offline the judge model is scripted: this shows plumbing, not judgement)")
    from common import get_model

    verdict_script = [tool_call_msg("Verdict", {"helpful": True, "reason": "correct and short"})]
    judge = make_judge(get_model(verdict_script))
    print(" ", judge({"answer": "6 x 7 = 42."}, {"answer": "42"}))

# EXERCISES
# 1. Add a fourth case where the agent must NOT call a tool ("Say hello"). What do `tools_exact` and
#    `tools_superset` each say about an agent that calls `add` anyway?
# 2. Add a latency or cost evaluator that fails any run with more than 4 tool calls.
# 3. Write a `summary_evaluator` that returns the pass rate across the dataset, and pass it via
#    `evaluate(..., summary_evaluators=[...])`.
# 4. With an API key: replace `make_target` with a real agent, upload the dataset, run `evaluate(data="name")`
#    twice (two models) and compare the experiments in the LangSmith UI.

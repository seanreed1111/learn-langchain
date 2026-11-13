"""LESSON 16 - Evals in CI, pytest, and feedback
==============================================
Lesson 15 produced scores. This lesson makes them DECISIONS and closes the loop with production data.

  1. A REGRESSION GATE     thresholds per metric; the script exits 1 if a candidate falls below them,
                           so a pull request that makes the agent worse cannot merge.
  2. PYTEST + LANGSMITH    `@pytest.mark.langsmith` turns a test into a tracked experiment row:
                              t.log_inputs(...) / t.log_outputs(...) / t.log_reference_outputs(...)
                              t.log_feedback(key=..., score=...)
                           See tests/test_agent_evals.py. It is an ordinary pytest test; with
                           LANGSMITH_TEST_TRACKING=false (set in tests/conftest.py) it logs nothing and runs
                           offline. With a key and tracking on, each case shows up in LangSmith.
                           !! WITHOUT a key and without that variable, the marker fails with an auth error.
  3. FEEDBACK              Attach a score to a SPECIFIC production run: thumbs up/down, a human grade, an
                           automated check. You need the run's id, so choose it yourself:
                              agent.invoke(..., {"run_id": my_uuid})   # becomes the trace's root id
                              project = client.read_project(project_name="my-project")
                              client.create_feedback(run_id=my_uuid, key="user_score", score=1.0,
                                                     session_id=project.id)
                           `session_id` (the PROJECT's UUID) is now required: feedback by run id alone is
                           deprecated and emits a LangSmithWarning (see the migration guide in the README).
                           Feedback is what you filter on to find bad traces and turn them into dataset
                           examples (the flywheel: production failure -> dataset -> eval -> fix).
  4. ONLINE EVALUATORS     (LangSmith UI) run evaluators automatically on a sample of live traces. Not
                           reproducible offline; the docs cover LLM-as-judge, code and multi-turn variants.

Run:  python lessons/16_evals_in_ci_and_feedback.py             # demo: gate a good and a regressed variant
      python lessons/16_evals_in_ci_and_feedback.py regressed   # exit code 1 (what a CI job would see)
"""

import importlib
import sys
import uuid

import langsmith as ls
from common import banner
from ls_offline import LocalLangSmith

lesson15 = importlib.import_module("15_evaluation")

# Minimum acceptable mean score per metric. Start from your baseline and ratchet UP, never down.
GATES: dict[str, float] = {"final_answer": 1.0, "tools_exact": 1.0}
PROJECT = "learn-langchain"


def check_gates(scores: dict[str, float], gates: dict[str, float] | None = None) -> list[str]:
    """Human-readable failures; an empty list means the candidate passes."""
    failures = []
    for metric, minimum in (gates or GATES).items():
        got = scores.get(metric)
        if got is None:
            failures.append(f"{metric}: no score produced (evaluator missing?)")
        elif got < minimum:
            failures.append(f"{metric}: {got:.2f} < required {minimum:.2f}")
    return failures


def gate_variant(variant: str) -> int:
    """Run the eval set for `variant` and return a process exit code: 0 pass, 1 fail."""
    scores = lesson15.run_experiment(variant)
    failures = check_gates(scores)
    print(f"variant={variant}: " + ", ".join(f"{k}={v:.2f}" for k, v in scores.items()))
    for failure in failures:
        print("  FAIL", failure)
    print("  PASS" if not failures else f"  {len(failures)} gate(s) failed")
    return 1 if failures else 0


# ---- Feedback on a specific run -----------------------------------------------------------------------
def answer_with_feedback(local: LocalLangSmith, question: str, thumbs_up: bool) -> tuple[str, uuid.UUID]:
    """Answer a question, then record the user's reaction against exactly that run."""
    run_id = uuid.uuid4()
    agent = lesson15.lesson05.build_agent(lesson15.script_for(lesson15.CASES[1][0], "good"))
    with ls.tracing_context(enabled=True, client=local.client, project_name=PROJECT):
        result = agent.invoke({"messages": question}, {"run_id": run_id})  # we pick the root run's id
    project = local.client.read_project(project_name=PROJECT)  # feedback needs the project's UUID
    local.client.create_feedback(
        run_id=run_id,
        session_id=project.id,
        key="user_score",
        score=1.0 if thumbs_up else 0.0,
        comment="clicked thumbs up" if thumbs_up else "clicked thumbs down",
    )
    return str(result["messages"][-1].text), run_id


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(gate_variant(sys.argv[1]))

    banner("1) The regression gate (what a CI job runs)")
    codes = {variant: gate_variant(variant) for variant in ("good", "regressed")}
    print("exit codes a CI job would see:", codes)

    banner("3) Feedback attaches to one specific run")
    local = LocalLangSmith()
    text, run_id = answer_with_feedback(local, lesson15.CASES[1][0], thumbs_up=False)
    root = next(r for r in local.runs() if r.get("parent_run_id") is None)
    fb = local.requests_to("/feedback")[0]
    print("answer           :", text)
    print("root trace id    :", root["id"])
    print("feedback.run_id  :", fb["run_id"], "(same run:", str(run_id) == fb["run_id"] == root["id"], ")")
    print("feedback payload :", {k: fb[k] for k in ("key", "score", "comment")})
    print("feedback.session_id is the project's id:", fb["session_id"] == local.session.project_id)

# EXERCISES
# 1. Add a latency gate (for example "p95 under 3 seconds") as a second kind of metric. Why is a
#    threshold on an AVERAGE a weak gate for latency?
# 2. Make `check_gates` tolerate a drop of 0.05 versus a stored baseline file instead of a fixed number.
#    What are the risks of a tolerance that is too loose?
# 3. With an API key, write feedback from a real chat UI's thumbs button. Where would the run id come from
#    in a web app (hint: return it with the response)?

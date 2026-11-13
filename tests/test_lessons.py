"""Tests = how you prove an agent works. Offline, deterministic, fast.

One group per lesson. Each asserts the exact claim the lesson makes.
Run:  pytest -q
"""

import importlib
from typing import Any

import pytest
from langchain.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command


def load(name: str) -> Any:
    return importlib.import_module(name)


# ---------------------------------------------------------------- 01 runnables
def test_01_chain_graph_and_graph_as_step():
    m = load("01_runnables_and_models")
    assert m.make_chain("hello").invoke({"topic": "x"}) == "hello"
    assert m.build_draft_graph("d").invoke({"topic": "t"}) == {"topic": "t", "draft": "d"}
    assert m.graph_as_chain_step("via chain").invoke("t") == "via chain"
    assert [c.text for c in m.get_model([AIMessage(content="a b")]).stream("hi") if c.text] == ["a", " ", "b"]


# ---------------------------------------------------------------- 02 messages
def test_02_add_messages_append_replace_remove():
    m = load("02_messages")
    out = m.reducer_demo()
    assert [x.text for x in out["appended"]] == ["hi", "hello", "new question"]
    assert [x.text for x in out["replaced"]] == ["hi", "hello, edited"]
    assert [x.text for x in out["removed"]] == ["hello"]
    assert isinstance(out["coerced"][-1], HumanMessage)


def test_02_tool_message_links_to_call_and_trim_keeps_system():
    m = load("02_messages")
    assert m.message_facts()["linked"] is True
    trimmed = m.trim_demo()
    assert trimmed[0].type == "system" and trimmed[-1].text == "answer 4"


# ---------------------------------------------------------------- 03 tools
def test_03_tool_schema_comes_from_signature_and_docstring():
    m = load("03_tools")
    assert m.multiply.name == "multiply"
    assert "Multiply two integers" in m.multiply.description
    assert set(m.multiply.args) == {"a", "b"}
    assert m.multiply.invoke({"a": 6, "b": 7}) == 42


def test_03_toolnode_error_behavior():
    m = load("03_tools")
    bad_args = {"name": "multiply", "args": {"a": "six", "b": 7}, "id": "x", "type": "tool_call"}
    unknown = {"name": "nope", "args": {}, "id": "y", "type": "tool_call"}
    assert "Error invoking tool" in m.run_calls(m.tool_graph(m.multiply), bad_args)[0].text
    assert "not a valid tool" in m.run_calls(m.tool_graph(m.multiply), unknown)[0].text
    raising = {"name": "flaky_lookup", "args": {"order_id": "9"}, "id": "z", "type": "tool_call"}
    with pytest.raises(KeyError):
        m.run_calls(m.tool_graph(m.flaky_lookup), raising)
    handled = m.run_calls(m.tool_graph(m.flaky_lookup, handle_tool_errors=True), raising)[0]
    assert handled.status == "error"


# ---------------------------------------------------------------- 04 structured output
def test_04_structured_output_and_routing():
    m = load("04_structured_output")
    assert m.extract("x").category == "billing"
    raw = m.extract_with_raw("x")
    assert set(raw) == {"raw", "parsed", "parsing_error"} and raw["parsing_error"] is None
    g = m.make_triage_graph([m.triage_message("technical", 2, "crash")])
    assert g.invoke({"text": "t", "triage": None, "handled_by": ""})["handled_by"].startswith("tech_team")


def test_04_invalid_schema_value_is_rejected():
    m = load("04_structured_output")
    bad = m.get_model([m.triage_message("billing", 9, "too urgent")]).with_structured_output(m.Triage)
    with pytest.raises(Exception, match=r"urgency|less than or equal"):
        bad.invoke("x")


# ---------------------------------------------------------------- 05 / 06 loop and create_agent
def test_05_loop_calls_both_tools():
    m = load("05_model_and_tools_in_a_graph")
    msgs = m.build_agent().invoke({"messages": [HumanMessage("go")]})["messages"]
    assert [x.text for x in msgs if x.type == "tool"] == ["42", "50"]
    assert msgs[-1].text.endswith("50.")


def test_05_recursion_limit_is_a_seatbelt():
    m = load("05_model_and_tools_in_a_graph")
    from common import tool_call_msg
    from langgraph.errors import GraphRecursionError

    forever = [tool_call_msg("add", {"a": 1, "b": 1}, f"c{i}") for i in range(50)]
    with pytest.raises(GraphRecursionError):
        m.build_agent(forever).invoke({"messages": "go"}, {"recursion_limit": 6})


def test_06_create_agent_is_the_same_graph_shape():
    m = load("06_create_agent")
    assert m.same_shape_as_lesson_05()
    assert set(m.build_agent().get_graph().nodes) == {"__start__", "model", "tools", "__end__"}


def test_06_structured_response():
    m = load("06_create_agent")
    out = m.structured_agent().invoke({"messages": "q"})
    assert out["structured_response"].value == 42


def test_06_raising_tool_crashes_by_default():
    m = load("06_create_agent")
    from common import get_model, tool_call_msg
    from langchain.agents import create_agent

    agent = create_agent(model=get_model([tool_call_msg("explode", {"x": 1})]), tools=[m.explode])
    with pytest.raises(ValueError, match="boom"):
        agent.invoke({"messages": "go"})


# ---------------------------------------------------------------- 07 state, context, memory
def test_07_context_in_state_out():
    m = load("07_state_context_and_memory")
    out = m.profile_agent().invoke({"messages": "greet me"}, context=m.Ctx(user_id="u1"))
    assert out["user_name"] == "Ada Lovelace"
    assert [x.text for x in out["messages"] if isinstance(x, ToolMessage)][-1] == "Hello, Ada Lovelace!"


def test_07_threads_are_isolated():
    m = load("07_state_context_and_memory")
    agent = m.memory_agent()
    agent.invoke({"messages": "Hi, I'm Bob."}, m.thread("t1"))
    agent.invoke({"messages": "What's my name?"}, m.thread("t1"))
    assert len(agent.get_state(m.thread("t1")).values["messages"]) == 4
    assert agent.get_state(m.thread("t2")).values == {}


def test_07_store_crosses_threads_per_user():
    m = load("07_state_context_and_memory")
    store = m.InMemoryStore()
    agent = m.store_agent(store)
    agent.invoke({"messages": "x"}, m.thread("a"), context=m.Ctx("u1"))
    out = agent.invoke({"messages": "y"}, m.thread("b"), context=m.Ctx("u1"))
    assert [x.text for x in out["messages"] if x.type == "tool"][-1] == "green"
    assert store.get(("prefs", "u2"), "color") is None


def test_07_command_without_tool_message_is_rejected():
    m = load("07_state_context_and_memory")
    from common import get_model, tool_call_msg
    from langchain.agents import create_agent
    from langchain.tools import ToolRuntime, tool

    @tool
    def sloppy(runtime: ToolRuntime[m.Ctx, m.ProfileState]) -> Command[Any]:
        """Updates state but forgets the ToolMessage."""
        return Command(update={"user_name": "Ada"})

    agent = create_agent(
        model=get_model([tool_call_msg("sloppy", {})]), tools=[sloppy], state_schema=m.ProfileState, context_schema=m.Ctx
    )
    with pytest.raises(ValueError, match="matching ToolMessage"):
        agent.invoke({"messages": "go"}, context=m.Ctx("u1"))


# ---------------------------------------------------------------- 08 middleware hooks
def test_08_hooks_are_nodes_and_order_is_first_to_last_then_reverse():
    m = load("08_middleware_hooks")
    assert "TracerA.before_model" in m.traced_agent().get_graph().nodes
    m.TRACE.clear()
    m.traced_agent().invoke({"messages": "1+2?"})
    t = m.TRACE
    assert t[:2] == ["A.before_agent", "B.before_agent"]
    assert t[-2:] == ["B.after_agent", "A.after_agent"]
    first_wrap = [x for x in t if "wrap_model_call" in x][:4]
    assert first_wrap == ["A.wrap_model_call:enter", "B.wrap_model_call:enter", "B.wrap_model_call:exit", "A.wrap_model_call:exit"]
    assert t.index("B.after_model") < t.index("A.after_model")


def test_08_budget_in_state_with_jump():
    m = load("08_middleware_hooks")
    out = m.budget_agent().invoke({"messages": "keep adding"})
    assert out["model_calls"] == 2
    assert out["messages"][-1].text.startswith("Stopped")


def test_08_jump_to_without_can_jump_to_is_silently_ignored():
    m = load("08_middleware_hooks")
    out = m.undeclared_jump_agent().invoke({"messages": "hi"})
    assert out["messages"][-1].text == "The model answered normally."


def test_08_duplicate_middleware_instances_are_rejected():
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware

    class Twin(AgentMiddleware):
        pass

    from common import get_model

    with pytest.raises(AssertionError, match="duplicate middleware"):
        create_agent(model=get_model([AIMessage(content="x")]), tools=[], middleware=[Twin(), Twin()])


def test_08_dynamic_prompt_and_tool_error_wrapper():
    m = load("08_middleware_hooks")
    m.SEEN.clear()
    m.prompt_agent().invoke({"messages": "1+2?"})
    assert [s["system"] for s in m.SEEN] == ["Be brief.", "You have used 3 messages; wrap up now."]
    out = m.resilient_agent().invoke({"messages": "go"})
    assert out["messages"][2].status == "error"
    assert out["messages"][-1].text.startswith("The tool is down")


# ---------------------------------------------------------------- 09 prebuilt middleware
def test_09_limits():
    m = load("09_prebuilt_middleware")
    out = m.model_limit_agent().invoke({"messages": "go"})
    assert sum(1 for x in out["messages"] if isinstance(x, AIMessage) and x.tool_calls) == 3
    assert "limits exceeded" in out["messages"][-1].text
    out = m.tool_limit_agent().invoke({"messages": "go"})
    assert any("Tool call limit exceeded" in x.text for x in out["messages"] if x.type == "tool")


def test_09_retries_fallback_and_errors():
    m = load("09_prebuilt_middleware")
    out = m.tool_retry_agent().invoke({"messages": "go"})
    assert m.ATTEMPTS["tool"] == 3 and "data for" in [x.text for x in out["messages"] if x.type == "tool"][-1]
    out = m.tool_error_agent().invoke({"messages": "go"})
    assert [x.text for x in out["messages"] if x.type == "tool"] == ["ValueError: cannot handle 'x'"]
    out = m.model_retry_agent().invoke({"messages": "hi"})
    assert m.ATTEMPTS["model"] == 2 and out["messages"][-1].text.startswith("Recovered")
    assert m.last_text(m.fallback_agent().invoke({"messages": "hi"})).startswith("Answered by the backup")


def test_09_retry_then_fallback_depends_on_order_and_on_failure():
    m = load("09_prebuilt_middleware")
    out = m.retry_then_fallback_agent().invoke({"messages": "hi"})
    assert m.ATTEMPTS["model"] == 3 and m.last_text(out).startswith("Answered by the backup")
    out = m.retry_then_fallback_agent(on_failure="continue").invoke({"messages": "hi"})
    assert m.ATTEMPTS["model"] == 3 and m.last_text(out).startswith("Model call failed after 3 attempts")
    out = m.retry_then_fallback_agent(fallback_outer=False).invoke({"messages": "hi"})
    assert m.ATTEMPTS["model"] == 1 and m.last_text(out).startswith("Answered by the backup")


def test_09_pii_and_summarization():
    m = load("09_prebuilt_middleware")
    out = m.pii_agent().invoke({"messages": "mail me at ada@example.com"})
    assert "ada@example.com" not in out["messages"][0].text and "[REDACTED_EMAIL]" in out["messages"][0].text
    agent = m.summarizing_agent()
    cfg = {"configurable": {"thread_id": "s"}}
    for turn in ("hi, I'm Bob", "I like graphs", "tell me a fact", "another fact"):
        agent.invoke({"messages": turn}, cfg)
    kept = agent.get_state(cfg).values["messages"]
    assert len(kept) < 8 and "summary" in kept[0].text.lower()


# ---------------------------------------------------------------- 10 human in the loop
@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ({"type": "approve"}, [("123", 79.0)]),
        ({"type": "edit", "edited_action": {"name": "issue_refund", "args": {"order_id": "123", "amount": 40.0}}}, [("123", 40.0)]),
        ({"type": "reject", "message": "no"}, []),
    ],
)
def test_10_nothing_happens_before_the_decision(decision: dict[str, Any], expected: list[tuple[str, float]]):
    m = load("10_human_in_the_loop")
    m.REFUNDS.clear()
    agent = m.refund_agent()
    paused = m.start(agent, "t")
    assert "__interrupt__" in paused and m.REFUNDS == []
    assert agent.get_state(m.cfg("t")).next == ("HumanInTheLoopMiddleware.after_model",)
    m.decide(agent, "t", decision)
    assert expected == m.REFUNDS


def test_10_when_gates_only_large_refunds():
    m = load("10_human_in_the_loop")
    m.REFUNDS.clear()
    small = m.threshold_agent(20.0).invoke({"messages": "x"}, m.cfg("small"))
    assert "__interrupt__" not in small and m.REFUNDS == [("123", 20.0)]
    m.REFUNDS.clear()
    large = m.threshold_agent(90.0).invoke({"messages": "x"}, m.cfg("large"))
    assert "__interrupt__" in large and m.REFUNDS == []


# ---------------------------------------------------------------- 11 streaming
def test_11_stream_modes():
    m = load("11_streaming")
    assert m.updates() == [("model", ["AIMessage"]), ("tools", ["ToolMessage"]), ("model", ["AIMessage"])]
    assert m.tokens() == "It is sunny in Paris."
    assert m.custom_events() == ["looking up Paris..."]
    assert set(m.mixed()) == {"updates", "custom"}
    assert m.values_sizes() == [1, 2, 3, 4]
    assert m.event_stream() == ["tool call: get_weather({'city': 'Paris'})", "message: It is sunny in Paris."]


# ---------------------------------------------------------------- 12 multi-agent
def test_12_subagent_as_tool_hides_inner_loop():
    m = load("12_multi_agent_and_dropping_down")
    out = m.build_supervisor().invoke({"messages": "what is a graph"})
    assert len(out["messages"]) == 4
    assert out["messages"][2].text == "Fact: a graph is nodes plus edges."


def test_12_agents_as_nodes_route_without_duplicating_messages():
    m = load("12_multi_agent_and_dropping_down")
    out = m.build_router_graph("billing").invoke({"messages": "charged twice"})
    assert out["team"] == "billing" and out["messages"][-1].text.startswith("Billing here")
    assert sum(x.type == "human" for x in out["messages"]) == 1
    assert "billing:model" in m.build_router_graph("technical").get_graph(xray=True).nodes


# ---------------------------------------------------------------- capstone
def test_capstone_approve_reject_and_isolation():
    cap = load("helpdesk_agent")
    r = cap.run_refund({"type": "approve"})
    assert r["refunds_before"] == [] and r["refunds_after"] == [{"order_id": "123", "amount": 79.0}]
    r = cap.run_refund({"type": "reject", "message": "no"})
    assert r["refunds_before"] == [] and r["refunds_after"] == []
    # user u2 cannot refund or look up u1's order: the tool error surfaces as a message, no refund is made
    r = cap.run_refund({"type": "approve"}, user_id="u2")
    assert r["refunds_after"] == []
    assert any("not found" in x.text for x in r["final"]["messages"] if x.type == "tool")


def test_capstone_faq_route_skips_agent():
    cap = load("helpdesk_agent")
    out = cap.build_desk("faq").invoke({"messages": "return policy?"}, cap.cfg("t"), context=cap.Ctx("u1"))
    assert out["route"] == "faq" and "help center" in out["messages"][-1].text


def test_capstone_model_call_budget_ends_the_run():
    cap = load("helpdesk_agent")
    from common import tool_call_msg

    endless = [tool_call_msg("lookup_order", {"order_id": "123"}, f"c{i}") for i in range(20)]
    desk = cap.build_desk(agent_script=endless, model_call_budget=2)
    out = desk.invoke({"messages": "refund"}, cap.cfg("budget"), context=cap.Ctx("u1"))
    assert "limits exceeded" in out["messages"][-1].text


def test_command_import_is_the_langgraph_one():
    assert Command.__module__.startswith("langgraph")


def test_remove_message_is_the_delete_marker():
    result: Any = add_messages([HumanMessage("a", id="1")], [RemoveMessage(id="1")])
    assert [m.text for m in result] == []

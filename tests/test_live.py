"""Opt-in tests that call a REAL model. Run with:  pytest -q -m live   (needs ANTHROPIC_API_KEY).

They prove the same graph shapes work with a real provider, not just the scripted fake.
"""

import os

import pytest
from common import make_real_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

pytestmark = pytest.mark.live


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@pytest.mark.skipif(not bool(os.getenv("ANTHROPIC_API_KEY")), reason="needs ANTHROPIC_API_KEY")
def test_real_model_calls_the_tool_and_answers():
    agent = create_agent(model=make_real_chat_model(), tools=[multiply])
    out = agent.invoke({"messages": "What is 6 times 7? Use the tool."})
    assert any(getattr(m, "tool_calls", None) for m in out["messages"])
    assert "42" in out["messages"][-1].text

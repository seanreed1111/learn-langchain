"""The offline/online switch in lessons/common.py. A wrong default here would make every lesson call the network."""

import pytest
from common import DEFAULT_MODEL, ScriptedChatModel, get_model, have_real_llm, model_name
from langchain.messages import AIMessage


def test_offline_by_default():
    assert have_real_llm() is False
    assert isinstance(get_model(), ScriptedChatModel)


def test_default_model_is_anthropic(monkeypatch: pytest.MonkeyPatch):
    assert model_name() == DEFAULT_MODEL == "anthropic:claude-sonnet-5-5"
    monkeypatch.setenv("LC_MODEL", "anthropic:claude-haiku-5-5")
    assert model_name() == "anthropic:claude-haiku-5-5"


def test_anthropic_key_alone_enables_real_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert have_real_llm() is True


def test_other_provider_key_needs_explicit_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert have_real_llm() is False  # the default model is Anthropic: an OpenAI key alone must not switch
    monkeypatch.setenv("LC_MODEL", "openai:some-model")
    assert have_real_llm() is True


def test_scripted_model_is_single_use():
    model = get_model([AIMessage(content="once")])
    assert model.invoke("x").text == "once"
    with pytest.raises(StopIteration):
        model.invoke("y")

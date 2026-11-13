"""Shared test setup: import paths, and every test starts OFFLINE.

Keys in your real environment or .env must never make a unit test hit the network, so each test gets
blank provider keys and no LC_MODEL override. Opt-in live tests (-m live) clear this themselves.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "lessons", ROOT / "capstone"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PROVIDER_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")

# Set at import time too: the langsmith pytest plugin reads this before any fixture runs.
os.environ["LANGSMITH_TEST_TRACKING"] = "false"


@pytest.fixture(autouse=True)
def _offline_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return
    for key in PROVIDER_KEYS:
        monkeypatch.setenv(key, "")
    # LangSmith: never trace to the network from a unit test, and run `@pytest.mark.langsmith` locally.
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TEST_TRACKING", "false")
    monkeypatch.delenv("LC_MODEL", raising=False)
    monkeypatch.delenv("LC_TEMPERATURE", raising=False)

"""Smoke tests: run every lesson exactly as a learner would (`python lessons/NN_x.py`).

Unit tests import functions; these run the real `__main__` blocks in a fresh interpreter, offline, so a
lesson that imports fine but crashes when run (or an API a dependency bump removed) is caught here.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LESSON_FILES = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "lessons").glob("[0-9][0-9]_*.py"))
SCRIPTS = [*LESSON_FILES, "capstone/helpdesk_agent.py"]
EXPECTED_LESSONS = {f"{n:02d}" for n in range(1, 17)}


def _run(script: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in {"LC_MODEL", "LC_TEMPERATURE", "LANGSMITH_API_KEY"}}
    env.update(
        ANTHROPIC_API_KEY="",
        OPENAI_API_KEY="",
        GOOGLE_API_KEY="",
        LANGSMITH_TRACING="false",
        LANGSMITH_API_KEY="",
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
    )
    return subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        encoding="utf-8",
    )


def test_every_expected_lesson_file_exists():
    assert {Path(f).name[:2] for f in LESSON_FILES} >= EXPECTED_LESSONS


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs_offline(script: str):
    done = _run(script)
    assert done.returncode == 0, f"{script} failed:\n{done.stderr[-1500:]}"
    assert done.stdout.strip(), f"{script} printed nothing"

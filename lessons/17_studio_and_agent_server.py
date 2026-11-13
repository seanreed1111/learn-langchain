"""LESSON 17 - LangSmith Studio and the Agent Server
===================================================
Tracing (13-14) and evals (15-16) look at runs after the fact. LangSmith STUDIO lets you drive and debug an
agent interactively, and it is built on something you can use without it: the **Agent Server**, a local or
hosted HTTP API that serves your graphs with threads, runs, checkpoints and interrupts built in.

    langgraph.json  ->  { "graphs": { "helpdesk": "./studio/agent.py:graph" }, "dependencies": ["."] }
    langgraph dev   ->  http://127.0.0.1:2024   (the API)
                        https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024   (the UI)

What Studio shows (from the docs): every step of a run (prompts, tool calls and results, final output),
token and latency metrics, exceptions with the surrounding state, hot reload when you edit code, and the
ability to re-run a thread from any step. Everything it does goes through the same API we call below.

Rules for a graph the server loads (see studio/agent.py):
  * export a COMPILED graph at module level; `langgraph.json` points at `file.py:variable`
  * do NOT pass a checkpointer: the server supplies persistence. That is why interrupts, thread history
    and "re-run from here" work, and why `HumanInTheLoopMiddleware` needs no `checkpointer=` in this file
  * the graph is imported once and reused, so it must be reusable (offline, our scripted model cycles)
  * keep traces local while experimenting: LANGSMITH_TRACING=false (Studio itself needs a LangSmith account)

Practical notes: install with `pip install -e ".[studio]"` (it is `langgraph-cli[inmem]`). Safari blocks
localhost connections to Studio: use `langgraph dev --tunnel`. On Windows, if `langgraph --help` crashes with
a UnicodeEncodeError, set PYTHONUTF8=1.

This script: (A) checks the config and the graph, (B) starts a real `langgraph dev` server on a free port,
(C) drives it with the Python SDK (run, pause for approval, resume, read history), then stops it.

Run:  python lessons/17_studio_and_agent_server.py
"""

import asyncio
import contextlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Generator
from pathlib import Path
from types import ModuleType
from typing import Any

from common import banner

ROOT = Path(__file__).resolve().parent.parent


# ---- A) Static checks: no server needed ---------------------------------------------------------------
def load_config() -> dict[str, Any]:
    config = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return config


def import_graph(spec: str) -> tuple[ModuleType, Any]:
    """Resolve a `./path/file.py:variable` graph spec the way the server does."""
    file_part, _, variable = spec.partition(":")
    path = (ROOT / file_part).resolve()
    module_spec = importlib.util.spec_from_file_location(f"_studio_{path.stem}", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module, getattr(module, variable)


def server_available() -> bool:
    return importlib.util.find_spec("langgraph_cli") is not None


# ---- B) A real dev server, started and always stopped -------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Kill the server AND its children (the dev server spawns workers)."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.killpg(os.getpgid(process.pid), 15)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/ok", timeout=2) as response:
            return bool(response.status == 200)
    except OSError:
        return False


@contextlib.contextmanager
def dev_server(timeout: float = 90.0) -> Generator[str, None, None]:
    """Run `langgraph dev` on a free port; yield its base URL; stop it on exit."""
    port = _free_port()
    env = {**os.environ, "LANGSMITH_TRACING": "false", "ANTHROPIC_API_KEY": "", "PYTHONUTF8": "1"}
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "langgraph_cli", "dev", "--no-browser", "--no-reload", "--port", str(port)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=sys.platform != "win32",
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError("langgraph dev exited early:\n" + log.read().decode(errors="replace")[-1500:])
                if _ready(url):
                    yield url  # NOT inside a `suppress`: the caller's exceptions must propagate
                    return
                time.sleep(1)
            raise TimeoutError(f"server did not become ready within {timeout}s")
        finally:
            _stop(process)


# ---- C) Drive it like Studio does: through the API ----------------------------------------------------
async def drive(url: str) -> dict[str, Any]:
    from langgraph_sdk import get_client

    client = get_client(url=url)
    assistants: Any = await client.assistants.search()
    thread: Any = await client.threads.create()
    thread_id = thread["thread_id"]

    paused: Any = await client.runs.wait(
        thread_id, "helpdesk", input={"messages": [{"role": "user", "content": "Please refund order 123."}]}
    )
    state: Any = await client.threads.get_state(thread_id)
    request = paused["__interrupt__"][0]["value"]["action_requests"][0]

    done: Any = await client.runs.wait(thread_id, "helpdesk", command={"resume": {"decisions": [{"type": "approve"}]}})
    history: Any = await client.threads.get_history(thread_id)
    return {
        "graphs": [a["graph_id"] for a in assistants],
        "paused_at": state["next"],
        "request": (request["name"], request["args"]),
        "final": done["messages"][-1]["content"],
        "message_types": [m["type"] for m in done["messages"]],
        "checkpoints": len(history),
    }


if __name__ == "__main__":
    banner("A) langgraph.json and the graph it points to")
    config = load_config()
    print("graphs:", config["graphs"])
    module, graph = import_graph(config["graphs"]["helpdesk"])
    print("type  :", type(graph).__name__, "| nodes:", list(graph.get_graph().nodes))
    print("checkpointer on the exported graph:", graph.checkpointer, "(the server provides one)")

    if not server_available():
        print('\nlanggraph-cli is not installed. Run: pip install -e ".[studio]"  then re-run this lesson.')
        sys.exit(0)

    banner("B) Start `langgraph dev` on a free port (takes ~10 seconds)")
    with dev_server() as base_url:
        print("server is up at", base_url)
        print("Studio URL:", f"https://smith.langchain.com/studio/?baseUrl={base_url}  (while it runs)")

        banner("C) Drive it through the SDK: run, pause, approve, history")
        result = asyncio.run(drive(base_url))
        print("graphs served   :", result["graphs"])
        print("paused at       :", result["paused_at"])
        print("approval request:", result["request"])
        print("final answer    :", result["final"])
        print("message types   :", result["message_types"])
        print("checkpoints     :", result["checkpoints"], "(each is a point Studio can re-run from)")
    print("\nserver stopped.")

# EXERCISES
# 1. Run `langgraph dev` yourself (PowerShell: `$env:PYTHONUTF8=1; langgraph dev`), open the Studio URL,
#    send a message, and approve the refund in the UI. Find the same interrupt payload that `drive` printed.
# 2. In Studio, edit a tool's docstring in studio/agent.py and watch hot reload pick it up. Why can that not
#    happen with `--no-reload`, which this lesson uses?
# 3. Extend `drive` to reject instead of approve, and to fork the thread from the first checkpoint with
#    `client.runs.wait(thread_id, "helpdesk", input=..., checkpoint_id=...)`.

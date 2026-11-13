"""Offline LangSmith for the tracing lessons (13-16): see exactly what WOULD be uploaded, with no network.

A LangSmith `Client` talks to the server through a `requests.Session`. `LocalLangSmith` hands the client a
recording session that answers 200 to everything and keeps every request, so:

  * the lessons can show the real run tree, tags, metadata, redaction and feedback payloads;
  * tests can assert on them (for example, that an email address never leaves the process);
  * nothing is sent anywhere, whatever is in your environment.

With a real `LANGSMITH_API_KEY` you would simply use `Client()` (or the default tracing from the
environment) and look at the same data in the LangSmith UI. Never ship this helper; it exists so you can
learn and TEST tracing behavior deterministically.
"""

from __future__ import annotations

import datetime
import json
import uuid
from collections.abc import Callable
from typing import Any, override

import requests
from langsmith import Client


class _RecordingSession(requests.Session):
    """A `requests.Session` that records every call and returns an empty 200."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, Any]] = []
        self.project_id = str(uuid.uuid4())  # what the "server" says the project's id is

    @override
    def request(self, method: str | bytes, url: str | bytes, *args: Any, **kwargs: Any) -> requests.Response:
        body = kwargs.get("data") if kwargs.get("data") is not None else kwargs.get("json")
        self.calls.append((str(method), str(url), body))
        response = requests.Response()
        response.status_code = 200
        payload: Any = {}
        if str(method) == "GET" and str(url).endswith("/sessions"):  # `client.read_project(project_name=...)`
            params = kwargs.get("params") or {}
            payload = [
                {
                    "id": self.project_id,
                    "name": params.get("name", "default"),
                    "start_time": datetime.datetime.now(datetime.UTC).isoformat(),
                    "tenant_id": str(uuid.uuid4()),
                    "reference_dataset_id": None,
                }
            ]
        response._content = json.dumps(payload).encode()
        response.url = str(url)
        return response


def _decode(body: Any) -> dict[str, Any] | None:
    if isinstance(body, bytes | str):
        try:
            data = json.loads(body)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None
    return body if isinstance(body, dict) else None


class LocalLangSmith:
    """A LangSmith client whose traffic is captured locally.

        local = LocalLangSmith(anonymizer=...)            # any `Client(...)` option works
        with tracing_context(enabled=True, client=local.client):
            agent.invoke(...)
        local.runs()                                      # merged run dicts (inputs, outputs, tags...)
    """

    def __init__(self, **client_options: Any) -> None:
        self.session = _RecordingSession()
        self.client = Client(
            api_url="http://langsmith.invalid",  # never contacted: the session answers locally
            api_key="offline-test-key",
            session=self.session,
            auto_batch_tracing=False,  # one request per run event, so they are easy to read
            info={},  # skip the server-info lookup
            **client_options,
        )

    def flush(self) -> None:
        self.client.flush()

    def requests_to(self, path: str) -> list[dict[str, Any]]:
        """Decoded JSON bodies of every request whose URL contains `path` (e.g. "/feedback")."""
        self.flush()
        found = (_decode(body) for _, url, body in self.session.calls if path in url)
        return [d for d in found if d is not None]

    def runs(self) -> list[dict[str, Any]]:
        """Run POSTs merged with their PATCHes (a run is created at start and updated at end), in start order."""
        self.flush()
        merged: dict[str, dict[str, Any]] = {}
        for method, url, body in self.session.calls:
            data = _decode(body)
            if data is None or "/runs" not in url or "id" not in data:
                continue
            if method == "POST":
                merged[data["id"]] = dict(data)
            elif method == "PATCH" and data["id"] in merged:
                merged[data["id"]].update({k: v for k, v in data.items() if v is not None})
        return list(merged.values())

    def names(self) -> list[str]:
        return [r["name"] for r in self.runs()]

    def tree(self, show: Callable[[dict[str, Any]], str] | None = None) -> list[str]:
        """Indented `run_type name` lines (parents before children) for printing."""
        runs = self.runs()
        children: dict[str | None, list[dict[str, Any]]] = {}
        for run in runs:
            children.setdefault(run.get("parent_run_id"), []).append(run)
        lines: list[str] = []

        def walk(parent: str | None, depth: int) -> None:
            for run in children.get(parent, []):
                label = show(run) if show else f"{run['run_type']:<5} {run['name']}"
                lines.append("  " * depth + label)
                walk(run["id"], depth + 1)

        walk(None, 0)
        return lines

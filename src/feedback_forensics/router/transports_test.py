"""Tests of the OpenRouter transport, against a local HTTP server (live lane)
and a fake of OpenRouter's batch API (batch lane). No real API calls."""

import json
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from feedback_forensics.models import main as models_main
from feedback_forensics.models import openrouter
from feedback_forensics.models.openrouter import OpenRouterError
from feedback_forensics.router import router as router_module
from feedback_forensics.router.datatypes import Policy
from feedback_forensics.router.router import Router
from feedback_forensics.router.testutils import (
    FIXED_END,
    MIDDLE_END,
    PROFILES,
    make_calls,
)
from feedback_forensics.router.transports import AccountError, OpenRouterTransport
from feedback_forensics.router.wire import build_body


def completion(text="An answer", cached=0, written=0) -> dict:
    return {
        "id": "gen-1",
        "provider": "Anthropic",
        "model": "anthropic/claude-haiku-5.5",
        "choices": [
            {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 12,
            "prompt_tokens_details": {
                "cached_tokens": cached,
                "cache_write_tokens": written,
            },
            "completion_tokens_details": {"reasoning_tokens": 3},
            "cost": 0.00042,
        },
    }


class LiveAPI:
    """Local server for `POST /chat/completions`: records the requests, and
    answers with the queued responses ((status, body)), then with completions."""

    def __init__(self):
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.responses: list[tuple[int, dict]] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                api.requests.append(json.loads(self.rfile.read(length)))
                api.headers.append({k.lower(): v for k, v in self.headers.items()})
                status, body = (
                    api.responses.pop(0) if api.responses else (200, completion())
                )
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api/v1"


@pytest.fixture
def live_api(monkeypatch):
    api = LiveAPI()
    thread = threading.Thread(target=api.server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(models_main, "OPENROUTER_API_BASE", api.url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "OPENAI_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setattr(router_module, "MAX_RETRY_WAIT", 0)
    yield api
    api.server.shutdown()
    api.server.server_close()


def make_router(tmp_path, strategy="live_ordered") -> Router:
    return Router(tmp_path, strategy=strategy, profiles=PROFILES)


def test_live_requests_are_sent_as_written(tmp_path, live_api):
    """Cache breakpoints reach the API as text blocks with `cache_control`."""
    calls = make_calls(groups=1, per_group=2)
    live_api.responses = [
        (200, completion("first", written=MIDDLE_END)),
        (200, completion("second", cached=MIDDLE_END)),
    ]
    router = make_router(tmp_path)  # (default transport: OpenRouter)
    keys = router.add(calls)
    report = router.run(Policy(max_concurrency=2))
    assert report.complete and report.sent_live == 2
    assert isinstance(router.transport, OpenRouterTransport)

    plan_order = sorted(calls, key=lambda call: call.text)
    for request, call in zip(live_api.requests, plan_order):
        assert request == {
            "model": "anthropic/claude-haiku-5.5",
            **build_body(call, (MIDDLE_END,)),
        }
        system, user = request["messages"]
        # Only the message with the breakpoint is split into blocks
        assert system == {"role": "system", "content": call.messages[0].text}
        first, second = user["content"]
        assert first == {
            "type": "text",
            "text": call.messages[1].text[: MIDDLE_END - len(system["content"])],
            "cache_control": {"type": "ephemeral"},
        }
        assert second["type"] == "text" and "cache_control" not in second
        assert first["text"] + second["text"] == call.messages[1].text
        assert request["usage"] == {"include": True}
        assert (request["max_tokens"], request["temperature"]) == (50, 0.0)
        assert "reasoning" not in request
    for headers in live_api.headers:
        assert headers["authorization"] == "Bearer test-key"
        assert headers["x-title"] == "Feedback Forensics"

    # Cached and written tokens are read from the usage
    results = router.collect(keys)
    first, second = (results[call.key] for call in plan_order)
    assert (first.text, second.text) == ("first", "second")
    assert first.usage.cache_write_tokens == MIDDLE_END
    assert second.usage.cached_tokens == MIDDLE_END
    assert second.usage.reasoning_tokens == 3
    assert second.usage.cost_usd == 0.00042
    assert report.cached_tokens == MIDDLE_END and report.input_tokens == 2000
    assert report.cost_usd == pytest.approx(0.00084)
    record = router.store.record(first.key)
    assert record["provider"] == "Anthropic" and record["route"] == "live"


def test_reasoning_effort_and_one_hour_lifetime(tmp_path, live_api):
    from dataclasses import replace

    from feedback_forensics.router.datatypes import Call, Params

    (call,) = make_calls(1, 1)
    call = Call(call.model, call.messages, Params(100, reasoning_effort="low"))
    body = build_body(call, (FIXED_END,), "1h")
    assert body["reasoning"] == {"effort": "low"}
    assert body["messages"][1]["content"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    # A breakpoint at the end of a message marks its (only) block
    body = build_body(call, (len(call.messages[0].text),))
    assert body["messages"][0]["content"] == [
        {
            "type": "text",
            "text": call.messages[0].text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert body["messages"][1]["content"] == call.messages[1].text
    assert replace(call).key == call.key


def test_live_errors(tmp_path, live_api):
    error = {"error": {"message": "slow down", "code": 429}}
    calls = make_calls(groups=1, per_group=1)
    # Rate limit, a provider error reported with status 200, then an answer
    live_api.responses = [(429, error), (200, {"error": {"code": 502}})]
    router = make_router(tmp_path)
    (key,) = router.add(calls)
    assert router.run(Policy(retries=2)).complete
    assert len(live_api.requests) == 3

    # An invalid request is not repeated
    router = make_router(tmp_path / "invalid")
    (key,) = router.add(calls)
    live_api.responses = [(400, {"error": {"message": "invalid", "code": 400}})]
    report = router.run(Policy(retries=2))
    assert "invalid" in report.gave_up[0][1] and len(live_api.requests) == 4

    # Out of credits: the run stops
    live_api.responses = [(402, {"error": {"message": "no credits", "code": 402}})]
    with pytest.raises(AccountError, match="insufficient credits"):
        router.run(Policy(retries=2))
    assert len(live_api.requests) == 5

    # No connection
    live_api.server.shutdown()
    live_api.server.server_close()
    report = router.run(Policy(retries=1))
    assert "Error" in report.gave_up[0][1] and len(live_api.requests) == 5


def test_missing_api_key_and_other_providers(tmp_path, live_api, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    router = make_router(tmp_path)
    router.add(make_calls(1, 1))
    with pytest.raises(AccountError, match="OPENROUTER_API_KEY"):
        router.run()
    assert live_api.requests == []


class BatchAPI:
    """Fake of `openrouter.openrouter_request` for the batch endpoints."""

    def __init__(self):
        self.batches: dict[str, dict] = {}
        self.lose_response = False
        self.posts = 0

    def __call__(self, method, path, **kwargs):
        if method == "POST":
            assert path == "/batches"
            self.posts += 1
            batch_id = f"batch_{len(self.batches) + 1}"
            self.batches[batch_id] = kwargs["json"]
            if self.lose_response:
                raise OpenRouterError(502, "bad gateway")
            return {"id": batch_id, "status": "validating"}
        if path == "/batches":
            return {
                "data": [
                    {
                        "id": batch_id,
                        "model": batch["model"],
                        "request_counts": {"total": len(batch["requests"])},
                    }
                    for batch_id, batch in self.batches.items()
                ],
                "has_more": False,
            }
        batch_id = path.split("/")[-1]
        results = [
            {
                "custom_id": request["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": completion(f"batch answer {request['custom_id'][:6]}"),
                },
                "error": None,
            }
            for request in self.batches[batch_id]["requests"]
        ]
        return {"id": batch_id, "status": "completed", "results": results}


@pytest.fixture
def batch_api(monkeypatch):
    api = BatchAPI()
    monkeypatch.setattr(openrouter, "openrouter_request", api)
    monkeypatch.setattr(router_module, "POLL_INTERVAL", 0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return api


LATER = Policy(deadline=timedelta(hours=30))


def test_batches_use_the_existing_batch_functions(tmp_path, batch_api):
    calls = make_calls(groups=2, per_group=2)
    router = make_router(tmp_path, "batch")
    keys = router.add(calls)
    report = router.run(LATER)
    assert report.complete and report.sent_batch == 4
    (batch,) = batch_api.batches.values()
    assert batch["endpoint"] == "/v1/chat/completions"
    assert batch["model"] == "anthropic/claude-haiku-5.5"
    by_key = {call.key: call for call in calls}
    assert sorted(r["custom_id"] for r in batch["requests"]) == sorted(keys)
    for request in batch["requests"]:
        assert request["body"] == build_body(by_key[request["custom_id"]])
    results = router.collect(keys)
    assert all(r.text == f"batch answer {key[:6]}" for key, r in results.items())
    assert {r.route for r in results.values()} == {"batch"}
    assert not (tmp_path / ".router" / "batches.json").exists()


def test_batch_breakpoints_are_sent_as_written(tmp_path, batch_api, live_api):
    """`warm_then_batch`: the batch requests carry the 1-hour breakpoint."""
    router = make_router(tmp_path, "warm_then_batch")
    router.add(make_calls(groups=2, per_group=2))
    report = router.run(LATER)
    assert report.complete and (report.sent_live, report.sent_batch) == (1, 3)
    (batch,) = batch_api.batches.values()
    marked = {"type": "ephemeral", "ttl": "1h"}
    for request in live_api.requests + [r["body"] for r in batch["requests"]]:
        assert request["messages"][1]["content"][0]["cache_control"] == marked


def test_lost_batch_submission_is_recovered(tmp_path, batch_api):
    """As for annotation batches: the batch is tracked before it is submitted,
    and found again via OpenRouter's batch list, not submitted twice."""
    calls = make_calls(groups=2, per_group=2)
    router = make_router(tmp_path, "batch")
    keys = router.add(calls)
    batch_api.lose_response = True
    report = router.run(LATER)
    assert report.pending == tuple(sorted(keys)) and batch_api.posts == 1
    state = json.loads((tmp_path / ".router" / "batches.json").read_text())
    assert state["batches"][0]["batch_id"] is None

    batch_api.lose_response = False
    router = make_router(tmp_path, "batch")
    router.add(calls)
    report = router.run(LATER)
    assert report.complete and batch_api.posts == 1
    assert all(r.ok for r in router.collect(keys).values())

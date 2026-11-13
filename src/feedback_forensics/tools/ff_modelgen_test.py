"""Tests for batch generation in ff_modelgen (OpenRouter API mocked)."""

import asyncio
import json
import pytest
from langchain_core.messages import AIMessage

import feedback_forensics.models
import feedback_forensics.tools.ff_modelgen as modelgen
import feedback_forensics.tools.ff_model_comparison as model_comparison
from feedback_forensics.data.ap_format import hash_string

MODEL = "openrouter/openai/gpt-6-sol"
PROMPTS = ["prompt one", "prompt two", "prompt three"]


class FakeOpenRouter:
    """Records requests and returns batch responses with given status."""

    def __init__(self, status="completed", failed_prompts=()):
        self.status = status
        self.failed_prompts = failed_prompts
        self.submitted = []
        self.polls = 0

    def __call__(self, method, path, **kwargs):
        if path == "/models":
            return {
                "data": [{"id": "openai/gpt-6-sol"}, {"id": "openai/gpt-6-sol:batch"}]
            }
        if method == "POST":
            self.submitted.append(kwargs["json"])
            return {"id": f"batch_{len(self.submitted)}", "status": "validating"}
        if path == "/batches":
            return {
                "data": [
                    {
                        "id": f"batch_{i + 1}",
                        "model": b["model"],
                        "request_counts": {"total": len(b["requests"])},
                    }
                    for i, b in enumerate(self.submitted)
                ],
                "has_more": False,
            }
        self.polls += 1
        results = None
        if self.status != "in_progress":
            results = [
                self._result(r["custom_id"]) for r in self.submitted[-1]["requests"]
            ]
        return {
            "id": path.split("/")[-1],
            "status": self.status,
            "results": results,
            "usage": {"cost": 0.1},
            "error": {"message": "rejected"} if self.status == "failed" else None,
        }

    def _result(self, custom_id):
        if custom_id in [hash_string(p) for p in self.failed_prompts]:
            return {"custom_id": custom_id, "response": None, "error": {"message": "x"}}
        body = {"choices": [{"message": {"content": f"answer {custom_id}"}}]}
        return {
            "custom_id": custom_id,
            "response": {"status_code": 200, "body": body},
            "error": None,
        }


@pytest.fixture
def fake_api(monkeypatch):
    def _make(**kwargs):
        api = FakeOpenRouter(**kwargs)
        monkeypatch.setattr(modelgen, "_openrouter_request", api)
        monkeypatch.setattr(modelgen.time, "sleep", lambda s: None)
        return api

    return _make


def _read_generations(tmp_path):
    output_file, _ = modelgen._get_batch_paths(tmp_path, MODEL)
    return [json.loads(line) for line in output_file.read_text().splitlines()]


def test_strip_nonbatch_suffix():
    assert modelgen.strip_nonbatch_suffix(MODEL + ":nonbatch") == MODEL
    assert modelgen.strip_nonbatch_suffix(MODEL) == MODEL


def test_get_batch_models(tmp_path, fake_api):
    fake_api(status="in_progress")
    no_variant = "openrouter/mistralai/mistral-large-4-0"
    models = [MODEL, no_variant, "gpt-4o"]
    # new model with batch variant
    assert modelgen.get_batch_models(models, tmp_path) == [MODEL]
    # disabled via suffix
    assert modelgen.get_batch_models([MODEL + ":nonbatch"], tmp_path) == []

    # still batch while batch pending, even if partial generations exist
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    output_file, _ = modelgen._get_batch_paths(tmp_path, MODEL)
    output_file.parent.mkdir(parents=True)
    output_file.write_text("")
    assert modelgen.get_batch_models(models, tmp_path) == [MODEL]

    # existing model with missing prompts uses live API
    modelgen._get_batch_paths(tmp_path, MODEL)[1].unlink()
    assert modelgen.get_batch_models(models, tmp_path) == []


def test_batch_roundtrip(tmp_path, fake_api):
    api = fake_api()
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)

    assert len(api.submitted) == 1
    submitted = api.submitted[0]
    assert submitted["model"] == "openai/gpt-6-sol"
    assert list(submitted.keys())[-1] == "requests"
    assert submitted["requests"][0]["body"]["max_tokens"] == 10000

    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    records = _read_generations(tmp_path)
    assert {r["prompt"] for r in records} == set(PROMPTS)
    assert all(r["model"] == MODEL and r["batch_id"] == "batch_1" for r in records)
    # state removed after results saved
    assert not modelgen._get_batch_paths(tmp_path, MODEL)[1].exists()
    # loadable in same way as live generations
    assert len(modelgen.load_generations(tmp_path, MODEL)) == 3


def test_pending_batch_not_resubmitted(tmp_path, fake_api):
    api = fake_api(status="in_progress")
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert not modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)

    # next run: same batch is polled again, nothing new submitted
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert len(api.submitted) == 1
    api.status = "completed"
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    assert len(_read_generations(tmp_path)) == 3


def test_only_missing_prompts_submitted(tmp_path, fake_api):
    api = fake_api(failed_prompts=["prompt two"])
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    assert len(_read_generations(tmp_path)) == 2
    assert "prompt two" not in [r["prompt"] for r in _read_generations(tmp_path)]

    # failed request is missing, so it would be resubmitted on its own
    api.failed_prompts = ()
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert len(api.submitted[-1]["requests"]) == 1

    # nothing submitted once all prompts processed
    modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert len(api.submitted) == 2
    assert len(_read_generations(tmp_path)) == 3


def test_interrupted_save_does_not_duplicate(tmp_path, fake_api):
    fake_api()
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    _, state_file = modelgen._get_batch_paths(tmp_path, MODEL)
    state = state_file.read_text()
    modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)

    # simulate crash after saving results but before state was removed
    state_file.write_text(state)
    modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    assert len(_read_generations(tmp_path)) == 3


def test_expired_batch_keeps_partial_results(tmp_path, fake_api):
    fake_api(status="expired", failed_prompts=["prompt one", "prompt two"])
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    # not done, but partial results saved and state cleared for resubmission
    assert not modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    assert len(_read_generations(tmp_path)) == 1
    assert not modelgen._get_batch_paths(tmp_path, MODEL)[1].exists()


def test_failed_batch_falls_back_to_live(tmp_path, fake_api):
    # e.g. batch variant listed, but deprecated by provider
    api = fake_api(status="failed")
    api.failed_prompts = PROMPTS  # failed batches have no successful results
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    # done (no batch pending), so caller generates missing prompts via live API
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    assert not modelgen._get_batch_paths(tmp_path, MODEL)[1].exists()


def test_compare_models_skips_pending_batch(tmp_path, fake_api, monkeypatch):
    fake_api(status="in_progress")
    live_calls = []

    async def fake_live(prompts, model_name, output_path, max_concurrent):
        live_calls.append(model_name)

    pairwise_models = []

    def stop_after_stage_1(model_names, *args, **kwargs):
        pairwise_models.extend(model_names)
        raise StopIteration

    monkeypatch.setattr(modelgen, "run_model_on_prompts_async", fake_live)
    monkeypatch.setattr(modelgen, "load_generations", lambda *args: None)
    monkeypatch.setattr(
        model_comparison, "create_pairwise_datasets", stop_after_stage_1
    )

    ref = "openrouter/openai/gpt-4o-2024-11-20"  # no batch variant in fake API
    live_model = "openrouter/openai/gpt-6.1-sol"
    with pytest.raises(StopIteration):
        model_comparison.compare_models(
            PROMPTS,
            [ref, MODEL, live_model + ":nonbatch"],
            [ref],
            output_path=tmp_path,
            batch_max_wait=0,
        )

    # second pass retries failed prompts
    assert live_calls == [ref, ref, live_model, live_model]
    assert pairwise_models == [ref, live_model]


def test_new_batch_not_found_yet_is_pending(tmp_path, fake_api, monkeypatch):
    """OpenRouter briefly returns 404 for newly submitted batches."""
    api = fake_api()

    def not_found_on_first_poll(method, path, **kwargs):
        if method == "GET" and api.polls == 0:
            api.polls += 1
            raise modelgen.OpenRouterError(404, "not found")
        return api(method, path, **kwargs)

    monkeypatch.setattr(modelgen, "_openrouter_request", not_found_on_first_poll)

    # new batch: keeps polling through the 404 until completed
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    deadline = modelgen.time.time() + 600
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=deadline)
    assert len(_read_generations(tmp_path)) == 3

    # old batch: 404 is a real error
    modelgen.submit_batch(["prompt four"], MODEL, tmp_path)
    _, state_file = modelgen._get_batch_paths(tmp_path, MODEL)
    state = json.loads(state_file.read_text())
    state["submitted_at"] = "2020-01-01T00:00:00"
    state_file.write_text(json.dumps(state))
    api.polls = 0
    with pytest.raises(modelgen.OpenRouterError):
        modelgen.collect_batch(["prompt four"], MODEL, tmp_path, deadline=0)


def test_pending_batch_collected_despite_nonbatch_suffix(tmp_path, fake_api):
    """Already paid batch must not be abandoned when model is switched to live."""
    fake_api(status="in_progress")
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert modelgen.get_batch_models([MODEL + ":nonbatch"], tmp_path) == [MODEL]
    assert modelgen.get_pending_batch_models(tmp_path) == [MODEL]


def test_lost_submit_response_is_recovered(tmp_path, fake_api, monkeypatch):
    """Batch created but response lost (e.g. 5xx/timeout): no double submission."""
    api = fake_api(status="completed")

    def lose_post_response(method, path, **kwargs):
        result = api(method, path, **kwargs)
        if method == "POST":
            raise modelgen.OpenRouterError(502, "bad gateway")
        return result

    monkeypatch.setattr(modelgen, "_openrouter_request", lose_post_response)
    with pytest.raises(modelgen.OpenRouterError):
        modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    _, state_file = modelgen._get_batch_paths(tmp_path, MODEL)
    assert json.loads(state_file.read_text())["batch_id"] is None

    # next run: batch found via list, not submitted again
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert len(api.submitted) == 1
    assert json.loads(state_file.read_text())["batch_id"] == "batch_1"
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    assert len(_read_generations(tmp_path)) == 3


def test_interrupted_submit_without_batch_resubmits(tmp_path, fake_api):
    api = fake_api()
    _, state_file = modelgen._get_batch_paths(tmp_path, MODEL)
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        json.dumps(
            {
                "batch_id": None,
                "model": MODEL,
                "num_requests": 3,
                "submitted_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert len(api.submitted) == 1
    assert json.loads(state_file.read_text())["batch_id"] == "batch_1"


def test_rejected_submit_clears_state(tmp_path, fake_api, monkeypatch):
    fake_api()

    def reject(method, path, **kwargs):
        raise modelgen.OpenRouterError(400, "bad request")

    monkeypatch.setattr(modelgen, "_openrouter_request", reject)
    with pytest.raises(modelgen.OpenRouterError):
        modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert not modelgen._get_batch_paths(tmp_path, MODEL)[1].exists()


def test_post_not_retried_on_server_error(monkeypatch):
    calls = []

    class Response:
        status_code, ok, text = 502, False, "bad gateway"

    def fake_request(method, url, **kwargs):
        calls.append(method)
        return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(modelgen.requests, "request", fake_request)
    monkeypatch.setattr(modelgen.time, "sleep", lambda s: None)
    with pytest.raises(modelgen.OpenRouterError):
        modelgen._openrouter_request("POST", "/batches", json={})
    assert calls == ["POST"]
    with pytest.raises(modelgen.OpenRouterError):
        modelgen._openrouter_request("GET", "/batches/x")
    assert calls.count("GET") == 5


def test_unparseable_results_are_kept(tmp_path, fake_api):
    """Malformed result or unknown prompt must not block saving other results."""
    api = fake_api()
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    original_result = api._result

    def result(custom_id):
        if custom_id == hash_string("prompt one"):
            return {
                "custom_id": custom_id,
                "response": {"status_code": 200, "body": {"choices": []}},
            }
        return original_result(custom_id)

    api._result = result
    # 'prompt three' removed from prompts file in the meantime
    assert modelgen.collect_batch(PROMPTS[:2], MODEL, tmp_path, deadline=0)
    assert [r["prompt"] for r in _read_generations(tmp_path)] == ["prompt two"]
    unparsed = json.loads((tmp_path / "batches/unparsed/batch_1.json").read_text())
    assert len(unparsed["results"]) == 2
    assert not modelgen._get_batch_paths(tmp_path, MODEL)[1].exists()


def test_expired_results_fall_back_to_live(tmp_path, fake_api, monkeypatch):
    api = fake_api()
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)

    def gone(method, path, **kwargs):
        raise modelgen.OpenRouterError(410, "gone")

    monkeypatch.setattr(modelgen, "_openrouter_request", gone)
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    assert not modelgen._get_batch_paths(tmp_path, MODEL)[1].exists()


def test_torn_line_does_not_corrupt_generations(tmp_path, fake_api):
    fake_api()
    output_file, _ = modelgen._get_batch_paths(tmp_path, MODEL)
    output_file.parent.mkdir(parents=True)
    output_file.write_text('{"prompt_id": "abc", "prom')  # interrupted write
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    assert len(modelgen.load_generations(tmp_path, MODEL)) == 3


def test_compare_models_collects_removed_model_batch(tmp_path, fake_api, monkeypatch):
    fake_api(status="completed")
    removed = "openrouter/openai/gpt-6-sol"
    modelgen.submit_batch(PROMPTS, removed, tmp_path)

    async def fake_live(*args, **kwargs):
        pass

    def stop_after_stage_1(*args, **kwargs):
        raise StopIteration

    monkeypatch.setattr(modelgen, "run_model_on_prompts_async", fake_live)
    monkeypatch.setattr(modelgen, "load_generations", lambda *args: None)
    monkeypatch.setattr(
        model_comparison, "create_pairwise_datasets", stop_after_stage_1
    )
    ref = "openrouter/openai/gpt-4o-2024-11-20"
    with pytest.raises(StopIteration):
        model_comparison.compare_models(
            PROMPTS, [ref], [ref], output_path=tmp_path, batch_max_wait=0
        )
    assert modelgen.get_pending_batch_models(tmp_path) == []
    assert len(_read_generations(tmp_path)) == 3


@pytest.mark.parametrize(
    "response, finish_reason, expected",
    [
        ("text", "stop", None),
        ("text", None, None),  # some providers report no finish reason
        ("partial", "length", "length"),
        ("", "length", "length"),
        ("text", "error", "error"),
        ("text", "content_filter", "content_filter"),
        ("text", "tool_calls", "tool_calls"),
        ("", "stop", "empty"),
        ("  \n ", None, "empty"),
        (None, "stop", "empty"),
    ],
)
def test_get_failure(response, finish_reason, expected):
    live = {"response_metadata": {"finish_reason": finish_reason}}
    batch = {"choices": [{"finish_reason": finish_reason}]}
    assert modelgen.get_failure(response, live) == expected
    assert modelgen.get_failure(response, batch) == expected


def test_batch_records_generation_params_and_provider(tmp_path, fake_api, monkeypatch):
    api = fake_api()
    original_call = api.__class__.__call__

    def with_provider(self, method, path, **kwargs):
        response = original_call(self, method, path, **kwargs)
        if method == "GET" and path.startswith("/batches/"):
            response.update(provider="OpenAI", model="openai/gpt-6-sol-20260922")
        return response

    monkeypatch.setattr(api.__class__, "__call__", with_provider)
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path, max_tokens=123)
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    record = _read_generations(tmp_path)[0]
    assert record["generation_params"] == {
        "api": "batch",
        "max_tokens": 123,
        "temperature": 0.0,
    }
    assert record["provider"] == "OpenAI"
    assert record["provider_model"] == "openai/gpt-6-sol-20260922"


def test_get_provider_info(monkeypatch):
    monkeypatch.setattr(modelgen.time, "sleep", lambda s: None)
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append(kwargs["params"]["id"])
        if len(calls) == 1:  # not registered yet
            raise modelgen.OpenRouterError(404, "not found")
        return {"data": {"provider_name": "Mistral", "model": "mistralai/m-2026"}}

    monkeypatch.setattr(modelgen, "_openrouter_request", fake_request)
    assert modelgen._get_provider_info("gen-1") == {
        "provider": "Mistral",
        "provider_model": "mistralai/m-2026",
    }
    # missing id or persistent errors: no provider info, but no exception
    assert modelgen._get_provider_info(None) == {
        "provider": None,
        "provider_model": None,
    }
    monkeypatch.setattr(
        modelgen,
        "_openrouter_request",
        lambda *a, **k: (_ for _ in ()).throw(modelgen.OpenRouterError(500, "x")),
    )
    assert modelgen._get_provider_info("gen-2")["provider"] is None


def test_cut_off_batch_result_saved_as_failure(tmp_path, fake_api, monkeypatch):
    api = fake_api()
    original_result = api._result

    def cut_off_prompt_two(custom_id):
        result = original_result(custom_id)
        finish_reason = "length" if custom_id == hash_string("prompt two") else "stop"
        result["response"]["body"]["choices"][0]["finish_reason"] = finish_reason
        return result

    monkeypatch.setattr(api, "_result", cut_off_prompt_two)
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    failures = {r["prompt"]: r["failure"] for r in _read_generations(tmp_path)}
    assert failures == {
        "prompt one": None,
        "prompt two": "length",
        "prompt three": None,
    }

    # failed generation counts as processed: not resubmitted
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    assert len(api.submitted) == 1


def test_load_generations_derives_failure_for_old_records(tmp_path):
    output_file, _ = modelgen._get_batch_paths(tmp_path, MODEL)
    output_file.parent.mkdir(parents=True)
    # old records have no 'failure' field
    records = [
        {
            "prompt": "a",
            "response": "x",
            "full_response": {"response_metadata": {"finish_reason": "stop"}},
        },
        {
            "prompt": "b",
            "response": "",
            "full_response": {"response_metadata": {"finish_reason": "length"}},
        },
    ]
    output_file.write_text("".join(json.dumps(r) + "\n" for r in records))
    generations = modelgen.load_generations(tmp_path, MODEL)
    assert generations["failure"].tolist() == [None, "length"]


def test_pairwise_skips_failed_generations(tmp_path):
    import pandas as pd

    def gens(rows):
        return pd.DataFrame(
            [
                {"prompt": p, "prompt_id": p, "response": r, "failure": f}
                for p, r, f in rows
            ]
        )

    ref = "openrouter/openai/gpt-4o"
    generations = {
        ref: gens(
            [
                ("ok", "ref ok", None),
                ("model cut off", "ref ok", None),
                ("ref cut off", "ref partial", "length"),  # kept: non-empty
                ("ref empty", "", "length"),  # skipped
            ]
        ),
        MODEL: gens(
            [
                ("ok", "model ok", None),
                ("model cut off", "model partial", "length"),  # skipped
                ("ref cut off", "model ok", None),
                ("ref empty", "model ok", None),
                ("ref missing", "model ok", None),  # skipped
            ]
        ),
    }
    model_comparison.create_pairwise_datasets(
        [MODEL, ref], [ref], generations, tmp_path
    )
    df = pd.read_csv(tmp_path / f"{MODEL.replace('/', '_')}.csv")
    assert df["prompt"].tolist() == ["ok", "ref cut off"]


OPENROUTER_USAGE = {
    "prompt_tokens": 12,
    "completion_tokens": 300,
    "completion_tokens_details": {"reasoning_tokens": 250},
    "cost": 0.0042,
}
EXPECTED_USAGE = {
    "input_tokens": 12,
    "output_tokens": 300,
    "reasoning_tokens": 250,
    "cost_usd": 0.0042,
}
NO_USAGE = dict.fromkeys(EXPECTED_USAGE)


@pytest.mark.parametrize(
    "full_response, expected",
    [
        # live (LangChain message) and batch (chat completion) via OpenRouter
        ({"response_metadata": {"token_usage": OPENROUTER_USAGE}}, EXPECTED_USAGE),
        ({"choices": [], "usage": OPENROUTER_USAGE}, EXPECTED_USAGE),
        # cost not reported (e.g. older generations), no reasoning details
        (
            {"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
            {**NO_USAGE, "input_tokens": 1, "output_tokens": 2},
        ),
        # e.g. Anthropic API via LangChain: only LangChain's usage metadata
        (
            {
                "response_metadata": {"usage": {"input_tokens": 5}},
                "usage_metadata": {
                    "input_tokens": 5,
                    "output_tokens": 7,
                    "output_token_details": {"reasoning": 3},
                },
            },
            {**NO_USAGE, "input_tokens": 5, "output_tokens": 7, "reasoning_tokens": 3},
        ),
        ({"choices": []}, NO_USAGE),
    ],
)
def test_get_usage(full_response, expected):
    assert modelgen.get_usage(full_response) == expected


def test_batch_records_usage(tmp_path, fake_api, monkeypatch):
    api = fake_api()
    original_result = api._result

    def with_usage(custom_id):
        result = original_result(custom_id)
        result["response"]["body"]["usage"] = OPENROUTER_USAGE
        return result

    monkeypatch.setattr(api, "_result", with_usage)
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    # cost of each request is requested (as for live generations)
    assert all(
        r["body"]["usage"] == {"include": True} for r in api.submitted[0]["requests"]
    )
    assert modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)
    assert [r["usage"] for r in _read_generations(tmp_path)] == [EXPECTED_USAGE] * 3


def test_live_records_usage(tmp_path, monkeypatch):
    class FakeModel:
        def invoke(self, messages):
            return AIMessage(
                content="answer",
                response_metadata={
                    "finish_reason": "stop",
                    "token_usage": OPENROUTER_USAGE,
                },
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 300,
                    "total_tokens": 312,
                },
            )

    monkeypatch.setattr(
        feedback_forensics.models, "get_model", lambda *args, **kwargs: FakeModel()
    )
    asyncio.run(
        modelgen.run_model_on_prompts_async(PROMPTS, "openai/fake", str(tmp_path))
    )
    output_file, _ = modelgen._get_batch_paths(tmp_path, "openai/fake")
    records = [json.loads(line) for line in output_file.read_text().splitlines()]
    assert len(records) == 3
    assert all(r["usage"] == EXPECTED_USAGE for r in records)
    assert all(r["failure"] is None for r in records)

"""Tests for batch generation in ff_modelgen (OpenRouter API mocked)."""

import json
import pytest

import feedback_forensics.tools.ff_modelgen as modelgen
import feedback_forensics.tools.ff_model_comparison as model_comparison
from inverse_cai.data.annotated_pairs_format import hash_string

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


def test_failed_batch_raises(tmp_path, fake_api):
    fake_api(status="failed")
    modelgen.submit_batch(PROMPTS, MODEL, tmp_path)
    with pytest.raises(RuntimeError, match="rejected"):
        modelgen.collect_batch(PROMPTS, MODEL, tmp_path, deadline=0)


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

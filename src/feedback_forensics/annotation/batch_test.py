"""Tests of annotation via OpenRouter's batch API (batch API faked, live API mocked)."""

import ast
import json

import pandas as pd
import pytest

import feedback_forensics.models.main
from feedback_forensics.annotation import annotate_dataset, batch
from feedback_forensics.models import openrouter
from feedback_forensics.annotation.batch import (
    BatchError,
    BatchFailedError,
    BatchPendingError,
    BatchStateError,
    BatchUnavailableError,
)
from feedback_forensics.annotation.mock_api import mock_response
from feedback_forensics.annotation.testutils import expected_vote, make_pairs_df
from feedback_forensics.annotation.voting import (
    INVALID_VOTE,
    get_preferred_text,
    get_rejected_text,
)
from feedback_forensics.models.openrouter import OpenRouterError
from feedback_forensics.tools import ff_annotate, ff_model_comparison
import feedback_forensics.tools.ff_modelgen as modelgen

MODEL = "openrouter/mock/annotator"
PRINCIPLES = [
    "Select the response that is more concise",
    "Select the response that is more friendly",
    "Select the response that uses more formal language",
]


class FakeBatchAPI:
    """Fake of OpenRouter's batch API: answers requests like the mock LLM server."""

    def __init__(self, status="completed", failed_ids=(), num_results=None):
        """
        Args:
            status: Status of all batches (results unless "in_progress").
            failed_ids: Custom ids of requests that fail within their batch.
            num_results: Only return results of the first requests of a batch.
        """
        self.status = status
        self.failed_ids = set(failed_ids)
        self.num_results = num_results
        self.batches: dict[str, dict] = {}  # batch id -> submitted batch
        # True: batch is created, but the response is lost. "not_created": the
        # request is lost (no batch is created).
        self.lose_submit_response = False
        self.not_found = False  # Batches are not known (404)
        self.polls = 0

    @property
    def requests(self) -> list[dict]:
        return [r for b in self.batches.values() for r in b["requests"]]

    def __call__(self, method, path, **kwargs):
        if path == "/models":
            return {"data": [{"id": "mock/annotator"}, {"id": "mock/annotator:batch"}]}
        if method == "POST":
            if self.lose_submit_response == "not_created":
                raise OpenRouterError(502, "bad gateway")
            batch_id = f"batch_{len(self.batches) + 1}"
            self.batches[batch_id] = kwargs["json"]
            if self.lose_submit_response:
                raise OpenRouterError(502, "bad gateway")
            return {"id": batch_id, "status": "validating"}
        if path == "/batches":
            return {
                "data": [
                    {
                        "id": batch_id,
                        "model": b["model"],
                        "request_counts": {"total": len(b["requests"])},
                    }
                    for batch_id, b in self.batches.items()
                ],
                "has_more": False,
            }
        self.polls += 1
        batch_id = path.split("/")[-1]
        if self.not_found:
            raise OpenRouterError(404, "not found")
        results = None
        if self.status != "in_progress":
            requests = self.batches[batch_id]["requests"][: self.num_results]
            results = [self._result(r) for r in requests]
        return {
            "id": batch_id,
            "status": self.status,
            "results": results,
            "usage": {"cost": 0.1},
            "error": None if self.status == "completed" else {"message": "x"},
        }

    def _result(self, request):
        custom_id = request["custom_id"]
        if custom_id in self.failed_ids:
            return {"custom_id": custom_id, "response": None, "error": {"message": "x"}}
        text, finish_reason = mock_response(request["body"]["messages"])
        body = {
            "model": "mock/annotator-20260101",
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 5},
                "cost": 0.001,
            },
        }
        return {
            "custom_id": custom_id,
            "response": {"status_code": 200, "body": body},
            "error": None,
        }


@pytest.fixture
def fake_api(monkeypatch):
    def _make(**kwargs):
        api = FakeBatchAPI(**kwargs)
        monkeypatch.setattr(openrouter, "openrouter_request", api)
        monkeypatch.setattr(batch, "POLL_INTERVAL", 0)
        return api

    return _make


@pytest.fixture
def live_api(mock_llm_server, monkeypatch):
    """Mock of OpenRouter's live API (and no list prices from the real API)."""
    server = mock_llm_server()
    monkeypatch.setattr(
        feedback_forensics.models.main, "OPENROUTER_API_BASE", server.base_url
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
    monkeypatch.setattr(
        "feedback_forensics.annotation.usage.get_openrouter_pricing", lambda model: None
    )
    return server


@pytest.fixture
def data(tmp_path):
    df = make_pairs_df(8, seed="batch")
    df.to_csv(tmp_path / "data.csv", index=False)
    return df


def annotate(tmp_path, **kwargs):
    kwargs = {"batch": True, "model": MODEL, **kwargs}
    return annotate_dataset(
        tmp_path / "data.csv", PRINCIPLES, output_dir=tmp_path / "out", **kwargs
    )


def read_votes(tmp_path) -> dict:
    df = pd.read_csv(tmp_path / "out" / "results" / "040_votes_per_comparison.csv")
    return {row["index"]: ast.literal_eval(row["votes"]) for _, row in df.iterrows()}


def expected_votes(df, position_debias=False) -> dict:
    return {
        idx: {
            key: expected_vote(
                principle,
                get_preferred_text(row),
                get_rejected_text(row),
                position_debias,
            )
            for key, principle in enumerate(PRINCIPLES)
        }
        for idx, row in df.iterrows()
    }


def read_summary(tmp_path) -> dict:
    path = tmp_path / "out" / "results" / "080_api_usage_summary.json"
    return json.loads(path.read_text())


def read_log(tmp_path) -> list[dict]:
    lines = (tmp_path / "out" / "api_calls.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def state_file(tmp_path, debiased=False):
    name = "01_principle_votes" + ("_debiased" if debiased else "") + "_batches.json"
    return tmp_path / "out" / "results" / "cache" / name


def test_batch_roundtrip(tmp_path, data, fake_api, live_api):
    api = fake_api()
    annotate(tmp_path, max_output_tokens=1234, reasoning_effort="low")

    votes = read_votes(tmp_path)
    assert votes == expected_votes(data)
    # Mock gives unparsable responses for some comparisons: not repeated via live API
    assert INVALID_VOTE in [v for vs in votes.values() for v in vs.values()]
    assert live_api.requests == []
    assert not state_file(tmp_path).exists()

    # One batch, with the same request parameters as live API calls
    (submitted,) = api.batches.values()
    assert submitted["model"] == "mock/annotator"
    assert list(submitted)[-1] == "requests"
    assert len(submitted["requests"]) == len(data)
    assert len({r["custom_id"] for r in submitted["requests"]}) == len(data)
    body = submitted["requests"][0]["body"]
    assert [m["role"] for m in body.pop("messages")] == ["system", "user"]
    assert body == {
        "max_tokens": 1234,
        "temperature": 0.0,
        "usage": {"include": True},
        "reasoning": {"effort": "low"},
    }

    # Usage and cost of each request are recorded
    summary = read_summary(tmp_path)
    assert summary["num_calls"] == len(data)
    assert summary["num_failed_calls"] == summary["num_unfinished_calls"] == 0
    assert summary["input_tokens"] == 100 * len(data)
    assert summary["reasoning_tokens"] == 5 * len(data)
    assert summary["cost_usd"] == pytest.approx(0.001 * len(data))
    assert summary["cost_complete"]
    assert summary["num_truncated_calls"] > 0
    assert summary["this_run"]["num_unparsable_responses"] > 0
    results = [e for e in read_log(tmp_path) if not e.get("started")]
    assert {e["batch_id"] for e in results} == {"batch_1"}
    assert results[0]["provider_model"] == "mock/annotator-20260101"

    # Re-running reuses the cached votes, and repeats the calls without valid vote
    # (here via the batch API again)
    annotate(tmp_path, max_output_tokens=1234, reasoning_effort="low")
    num_invalid = sum(set(vs.values()) == {INVALID_VOTE} for vs in votes.values())
    assert len(api.batches["batch_2"]["requests"]) == num_invalid
    assert read_votes(tmp_path) == votes


def test_same_requests_as_live_api(tmp_path, data, fake_api, live_api):
    api = fake_api()
    annotate(tmp_path)
    annotate_dataset(
        tmp_path / "data.csv", PRINCIPLES, model=MODEL, output_dir=tmp_path / "live"
    )

    def key(messages):
        return json.dumps(messages, sort_keys=True)

    live_messages = sorted(key(r["messages"]) for r in live_api.requests)
    batch_messages = sorted(key(r["body"]["messages"]) for r in api.requests)
    assert batch_messages == live_messages
    live_body = live_api.requests[0]
    batch_body = api.requests[0]["body"]
    assert batch_body["temperature"] == live_body["temperature"]
    assert batch_body["usage"] == live_body["usage"]
    assert batch_body["max_tokens"] == live_body.get(
        "max_completion_tokens", live_body.get("max_tokens")
    )
    live_votes = pd.read_csv(tmp_path / "live/results/040_votes_per_comparison.csv")
    batch_votes = pd.read_csv(tmp_path / "out/results/040_votes_per_comparison.csv")
    pd.testing.assert_frame_equal(batch_votes, live_votes)


def test_pending_batch_is_collected_by_later_run(tmp_path, data, fake_api, live_api):
    api = fake_api(status="in_progress")
    with pytest.raises(BatchPendingError, match="batch_1"):
        annotate(tmp_path, batch_max_wait=0)
    assert state_file(tmp_path).exists()
    assert not (tmp_path / "out" / "results" / "070_annotations_train_ap.json").exists()
    # Requests of pending batches may be billed: logged as unfinished calls
    assert len(read_log(tmp_path)) == len(data)
    assert all(e["started"] and e["batch_id"] == "batch_1" for e in read_log(tmp_path))

    # Still pending: not submitted again
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)
    assert len(api.batches) == 1

    api.status = "completed"
    annotate(tmp_path, batch_max_wait=0)
    assert len(api.batches) == 1
    assert read_votes(tmp_path) == expected_votes(data)
    assert not state_file(tmp_path).exists()
    summary = read_summary(tmp_path)
    assert summary["num_calls"] == summary["this_run"]["num_calls"] == len(data)
    assert summary["num_unfinished_calls"] == 0
    assert live_api.requests == []


def test_waits_for_batch(tmp_path, data, fake_api, live_api, monkeypatch):
    api = fake_api(status="in_progress")
    waits = []

    async def wait(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            api.status = "completed"

    monkeypatch.setattr(batch, "POLL_INTERVAL", 60)
    monkeypatch.setattr(batch, "_wait", wait)
    annotate(tmp_path)
    assert waits == [60, 60, 60]
    assert read_votes(tmp_path) == expected_votes(data)


def test_waiting_can_be_interrupted(tmp_path, data, fake_api, live_api, monkeypatch):
    """Regression: the wait ran in a thread that an interrupt (Ctrl-C) did not stop."""
    api = fake_api(status="in_progress")

    async def wait(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "_wait", wait)
    with pytest.raises(KeyboardInterrupt):
        annotate(tmp_path)
    assert api.polls == 1
    assert state_file(tmp_path).exists()

    api.status = "completed"
    annotate(tmp_path)
    assert len(api.batches) == 1
    assert read_votes(tmp_path) == expected_votes(data)


@pytest.mark.parametrize("position_debias", [False, True])
def test_several_batches(
    tmp_path, data, fake_api, live_api, monkeypatch, position_debias
):
    api = fake_api()
    monkeypatch.setattr(batch, "MAX_REQUESTS_PER_BATCH", 3)
    # Two passes over the data (two prompts per comparison and response order)
    annotate(tmp_path, position_debias=position_debias, max_principles_per_prompt=2)

    num_requests = len(data) * 2 * (2 if position_debias else 1)
    assert len(api.requests) == num_requests
    assert len(api.batches) == -(-num_requests // 3)
    assert all(len(b["requests"]) <= 3 for b in api.batches.values())
    assert read_votes(tmp_path) == expected_votes(data, position_debias)
    assert live_api.requests == []
    assert not state_file(tmp_path, debiased=position_debias).exists()


def test_failed_requests_are_repeated_via_live_api(tmp_path, data, fake_api, live_api):
    api = fake_api()
    with pytest.raises(BatchPendingError):
        api.status = "in_progress"
        annotate(tmp_path, batch_max_wait=0)
    failed = api.requests[:3]
    api.failed_ids = {r["custom_id"] for r in failed}
    api.status = "completed"
    annotate(tmp_path)

    assert sorted(json.dumps(r["messages"]) for r in live_api.requests) == sorted(
        json.dumps(r["body"]["messages"]) for r in failed
    )
    assert read_votes(tmp_path) == expected_votes(data)
    summary = read_summary(tmp_path)
    assert summary["num_calls"] == len(data) + 3
    assert summary["num_failed_calls"] == 3
    assert summary["num_unfinished_calls"] == 0


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_incomplete_batch_keeps_partial_results(
    tmp_path, data, fake_api, live_api, status
):
    """Not silently annotated via the live API (at full price) instead."""
    api = fake_api(status=status, num_results=4)
    with pytest.raises(BatchFailedError, match=f"batch_1 \\({status}"):
        annotate(tmp_path)
    assert live_api.requests == []
    assert not state_file(tmp_path).exists()

    # Remaining requests: via live API if requested, else as a new batch
    annotate(tmp_path, batch=False)
    num_cached = len(data) - len(live_api.requests)
    assert 0 < num_cached <= 4
    assert len(api.batches) == 1
    assert read_votes(tmp_path) == expected_votes(data)
    # Requests without result in the batch are not left as unfinished calls
    summary = read_summary(tmp_path)
    assert summary["num_unfinished_calls"] == 0
    assert summary["num_failed_calls"] == len(data) - 4
    assert summary["num_calls"] == len(data) + len(live_api.requests)


def test_lost_submit_response_is_recovered(tmp_path, data, fake_api, live_api):
    api = fake_api(status="in_progress")
    api.lose_submit_response = True
    with pytest.raises(BatchError, match="bad gateway"):
        annotate(tmp_path, batch_max_wait=0)
    state = json.loads(state_file(tmp_path).read_text())
    assert [b["batch_id"] for b in state["batches"]] == [None]

    # Batch without results yet: not known whether it is the lost batch (it may
    # be another batch of the same model and size). Not submitted again.
    api.lose_submit_response = False
    with pytest.raises(BatchPendingError, match="id not known yet"):
        annotate(tmp_path, batch_max_wait=0)
    assert len(api.batches) == 1

    api.status = "completed"
    annotate(tmp_path)
    assert len(api.batches) == 1  # not submitted again
    assert read_votes(tmp_path) == expected_votes(data)
    assert live_api.requests == []
    assert read_summary(tmp_path)["num_calls"] == len(data)


def test_batch_of_other_dataset_is_not_taken_for_lost_batch(
    tmp_path, data, fake_api, live_api
):
    """Regression: any batch of the same model and size was taken for the lost
    batch, e.g. the batch of another dataset of ff-model-comparison. Its results
    matched no request, so everything was annotated via the live API instead."""
    api = fake_api()
    api.lose_submit_response = "not_created"
    with pytest.raises(BatchError, match="bad gateway"):
        annotate(tmp_path)
    messages = [{"role": "user", "content": "Request of another dataset"}]
    api.batches["batch_1"] = {
        "model": "mock/annotator",
        "requests": [
            {"custom_id": f"other-{i}", "body": {"messages": messages}}
            for i in range(len(data))
        ],
    }

    api.lose_submit_response = False
    annotate(tmp_path)
    assert list(api.batches) == ["batch_1", "batch_2"]
    assert len(api.batches["batch_2"]["requests"]) == len(data)
    assert live_api.requests == []
    assert read_votes(tmp_path) == expected_votes(data)
    cache_dir = tmp_path / "out" / "results" / "cache"
    assert not list(cache_dir.glob("*unmatched*"))


def test_interrupted_submit_without_batch_resubmits(tmp_path, data, fake_api, live_api):
    api = fake_api(status="in_progress")
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)
    # As if interrupted before the batch was created on OpenRouter
    state = json.loads(state_file(tmp_path).read_text())
    state["batches"][0]["batch_id"] = None
    state_file(tmp_path).write_text(json.dumps(state))
    api.batches.clear()

    api.status = "completed"
    annotate(tmp_path)
    assert len(api.batches) == 1
    assert read_votes(tmp_path) == expected_votes(data)


def test_rejected_submit_clears_state(tmp_path, data, fake_api, live_api, monkeypatch):
    api = fake_api()

    def reject(method, path, **kwargs):
        if method == "POST":
            raise OpenRouterError(400, "invalid request")
        return api(method, path, **kwargs)

    monkeypatch.setattr(openrouter, "openrouter_request", reject)
    with pytest.raises(BatchError, match="invalid request"):
        annotate(tmp_path)
    assert not state_file(tmp_path).exists()


def test_batch_unknown_to_openrouter_is_kept(tmp_path, data, fake_api, live_api):
    """Regression: the batch was dropped, and its requests submitted (and paid)
    again, although it may be running (other API key, temporary error)."""
    api = fake_api(status="in_progress")
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)
    api.not_found = True
    # Newly submitted batches may briefly not be found
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)

    state = json.loads(state_file(tmp_path).read_text())
    state["batches"][0]["submitted_at"] = "2020-01-01T00:00:00+00:00"
    state_file(tmp_path).write_text(json.dumps(state))
    with pytest.raises(BatchError, match="batch_1.*not known to OpenRouter") as e:
        annotate(tmp_path, batch_max_wait=0)
    assert str(state_file(tmp_path)) in str(e.value)
    assert state_file(tmp_path).exists()

    api.not_found = False
    api.status = "completed"
    annotate(tmp_path)
    assert list(api.batches) == ["batch_1"]
    assert live_api.requests == []
    assert read_votes(tmp_path) == expected_votes(data)


def test_failed_batch_is_reported_while_others_are_pending(
    tmp_path, data, fake_api, live_api, monkeypatch
):
    """Regression: only the pending batches were reported, the next run then
    submitted the requests of the failed batch again without notice."""
    monkeypatch.setattr(batch, "MAX_REQUESTS_PER_BATCH", 5)
    api = fake_api(status="in_progress")
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)

    def first_batch_failed(method, path, **kwargs):
        result = api(method, path, **kwargs)
        if path == "/batches/batch_1":
            result.update(status="failed", results=None, error={"message": "x"})
        return result

    monkeypatch.setattr(openrouter, "openrouter_request", first_batch_failed)
    with pytest.raises(BatchFailedError, match="batch_1.*Also:.*batch_2.*pending"):
        annotate(tmp_path, batch_max_wait=0)
    state = json.loads(state_file(tmp_path).read_text())
    assert [entry["batch_id"] for entry in state["batches"]] == ["batch_2", "batch_3"]


def test_pending_batches_are_collected_without_batch_variant(
    tmp_path, data, fake_api, live_api, monkeypatch
):
    """Regression: if the batch variant was removed after submission, neither
    batch runs (no variant) nor other runs (batches pending) could continue."""
    monkeypatch.setattr(batch, "MAX_REQUESTS_PER_BATCH", 5)
    api = fake_api(status="in_progress")
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)
    assert len(api.batches) == 3

    def without_variant_and_first_batch(method, path, **kwargs):
        if path == "/models":
            return {"data": [{"id": "mock/annotator"}]}
        result = api(method, path, **kwargs)
        if path == "/batches/batch_1":
            # Its requests are to be submitted again
            result.update(status="expired", results=None)
        return result

    monkeypatch.setattr(
        openrouter, "openrouter_request", without_variant_and_first_batch
    )
    api.status = "completed"
    with pytest.raises(BatchFailedError, match="batch_1"):
        annotate(tmp_path)
    assert not state_file(tmp_path).exists()
    with pytest.raises(BatchUnavailableError, match="mock/annotator:batch"):
        annotate(tmp_path)
    assert len(api.batches) == 3
    assert live_api.requests == []
    # The results of the collected batches are saved: only the rest is sent live
    # (and requests answered without any valid vote, as in every resumed run)
    annotate(tmp_path, batch=False)
    assert 5 <= len(live_api.requests) < len(api.requests)
    assert read_votes(tmp_path) == expected_votes(data)


def test_malformed_results_do_not_block_batch(
    tmp_path, data, fake_api, live_api, monkeypatch
):
    """Regression: a result that is not an object crashed every (re-)run."""
    api = fake_api()

    def with_malformed_results(method, path, **kwargs):
        result = api(method, path, **kwargs)
        if path.startswith("/batches/"):
            results = result["results"]
            results[0] = "not an object"
            results[1] = {"custom_id": results[1]["custom_id"], "response": "text"}
            results[2]["response"]["body"]["choices"][0] = None
        return result

    monkeypatch.setattr(openrouter, "openrouter_request", with_malformed_results)
    annotate(tmp_path)
    assert not state_file(tmp_path).exists()
    # Their requests are repeated via the live API
    assert len(live_api.requests) == 3
    assert read_votes(tmp_path) == expected_votes(data)


def test_annotating_without_batch_api_while_batches_are_pending_is_refused(
    tmp_path, data, fake_api, live_api
):
    """Regression: all requests of the pending batches were sent again, via the
    live API."""
    api = fake_api(status="in_progress")
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)
    with pytest.raises(BatchStateError, match="Re-run with batch API"):
        annotate(tmp_path, batch=False)
    assert live_api.requests == []
    assert ff_annotate.run(cli_args(tmp_path)) == 1

    # Abandoning the batches
    state_file(tmp_path).unlink()
    annotate(tmp_path, batch=False)
    assert len(live_api.requests) == len(data)
    assert len(api.batches) == 1


def test_results_are_recorded_once_if_saving_is_interrupted(
    tmp_path, data, fake_api, live_api
):
    api = fake_api(status="in_progress")
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)
    state = state_file(tmp_path).read_text()
    api.status = "completed"
    annotate(tmp_path)
    summary = read_summary(tmp_path)

    # As if interrupted after saving the results, before updating the state
    state_file(tmp_path).write_text(state)
    annotate(tmp_path)
    assert len(api.batches) == 1
    assert live_api.requests == []
    assert read_votes(tmp_path) == expected_votes(data)
    rerun_summary = read_summary(tmp_path)
    assert rerun_summary["this_run"]["num_calls"] == 0
    for key in ("num_calls", "cost_usd", "input_tokens", "num_unfinished_calls"):
        assert rerun_summary[key] == summary[key]


def test_usage_of_requests_with_the_same_votes(tmp_path, fake_api, live_api):
    """Regression: of requests for the same comparison (duplicate rows, shown in
    different orders), only the usage of the first was recorded, the others were
    left as unfinished calls."""
    api = fake_api()
    df = make_pairs_df(8, seed="batch")
    df = pd.concat([df] * 6, ignore_index=True)
    df.to_csv(tmp_path / "data.csv", index=False)
    annotate(tmp_path)

    num_requests = len(api.requests)
    assert len(df) // 6 < num_requests <= 2 * len(df) // 6
    summary = read_summary(tmp_path)
    assert summary["num_calls"] == num_requests
    assert summary["num_unfinished_calls"] == 0
    assert summary["cost_usd"] == pytest.approx(0.001 * num_requests)
    assert live_api.requests == []


def test_results_not_matching_the_data_are_kept_but_not_used(
    tmp_path, data, fake_api, live_api
):
    api = fake_api(status="in_progress")
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)

    # Data changes while the batch is pending
    new_data = make_pairs_df(8, seed="other")
    new_data.to_csv(tmp_path / "data.csv", index=False)
    api.status = "completed"
    annotate(tmp_path)

    assert read_votes(tmp_path) == expected_votes(new_data)
    assert len(api.batches) == 2
    unmatched = tmp_path / "out/results/cache/01_principle_votes_unmatched_batch_1.json"
    assert len(json.loads(unmatched.read_text())["results"]) == len(data)


def test_pending_batch_with_other_settings_is_not_mixed_up(
    tmp_path, data, fake_api, live_api
):
    api = fake_api(status="in_progress")
    with pytest.raises(BatchPendingError):
        annotate(tmp_path, batch_max_wait=0)
    with pytest.raises(
        BatchStateError, match="max_output_tokens=3000 \\(requested: 500\\)"
    ):
        annotate(tmp_path, max_output_tokens=500)
    assert len(api.batches) == 1
    assert state_file(tmp_path).exists()


def test_model_without_batch_variant(tmp_path, data, fake_api, live_api):
    api = fake_api()
    with pytest.raises(BatchUnavailableError, match="mock/other:batch"):
        annotate(tmp_path, model="openrouter/mock/other")
    assert api.batches == {}
    assert live_api.requests == []


def test_non_openrouter_model_rejected_before_writing_files(tmp_path, data):
    with pytest.raises(ValueError, match="only supported for 'openrouter/' models"):
        annotate(tmp_path, model="openai/gpt-4o")
    assert not (tmp_path / "out").exists()


def cli_args(tmp_path) -> list[str]:
    args = ["-d", str(tmp_path / "data.csv"), "-v", "", "-p", json.dumps(PRINCIPLES)]
    return args + ["-o", str(tmp_path / "out"), "-m", MODEL]


def test_cli(tmp_path, data, fake_api, live_api, capsys):
    api = fake_api(status="in_progress")
    args = cli_args(tmp_path) + ["--batch"]
    assert ff_annotate.run(args + ["--batch-max-wait", "0"]) == 3
    api.status = "completed"
    assert ff_annotate.run(args) == 0
    assert len(api.batches) == 1
    assert read_votes(tmp_path) == expected_votes(data)

    with pytest.raises(SystemExit):
        ff_annotate.run(["-d", "data.csv", "-m", "openai/gpt-4o", "--batch"])
    assert "only supported for 'openrouter/' models" in capsys.readouterr().err


@pytest.fixture
def compare(tmp_path, monkeypatch):
    """Run ff-model-comparison with batch annotation (generation stubbed)."""
    prompts = [f"Prompt {i}: please explain topic {i}." for i in range(4)]
    models = ["openai/reference", "openai/model-x", "openai/model-y"]

    async def run_model_on_prompts_async(*args, **kwargs):
        pass

    def load_generations(output_path, model_name):
        responses = [
            f"{model_name}: {p}" + " Detail." * i for i, p in enumerate(prompts)
        ]
        return pd.DataFrame({"prompt": prompts, "response": responses})

    monkeypatch.setattr(
        modelgen, "run_model_on_prompts_async", run_model_on_prompts_async
    )
    monkeypatch.setattr(modelgen, "load_generations", load_generations)

    def _compare(**kwargs):
        kwargs = {"annotator_model": MODEL, "batch_annotation": True, **kwargs}
        return ff_model_comparison.compare_models(
            prompts=prompts,
            model_names=models,
            reference_models=models[:1],
            output_path=str(tmp_path),
            **kwargs,
        )

    return _compare


def annotation_files(tmp_path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "annotations").glob("*.json"))


def test_model_comparison_with_batch_annotation(tmp_path, fake_api, live_api, compare):
    api = fake_api(status="in_progress")
    not_annotated = compare(batch_max_wait=0)
    # Batches of both datasets are submitted, nothing is annotated yet
    assert len(api.batches) == 2
    assert annotation_files(tmp_path) == []
    assert sorted(not_annotated["pending"]) == [
        "openai_model-x.csv",
        "openai_model-y.csv",
    ]
    assert not_annotated["failed"] == []

    api.status = "completed"
    assert compare() == {"pending": [], "failed": []}
    assert len(api.batches) == 2
    assert live_api.requests == []
    assert annotation_files(tmp_path) == [
        "combined_ap.json",
        "openai_model-x_ap.json",
        "openai_model-y_ap.json",
    ]


def test_model_comparison_continues_if_batch_of_a_dataset_fails(
    tmp_path, fake_api, live_api, compare, monkeypatch
):
    """Regression: the error of one dataset stopped the comparison (the batches of
    the other datasets were not submitted or collected, nothing was merged)."""
    api = fake_api(status="in_progress")
    compare(batch_max_wait=0)

    def first_batch_failed(method, path, **kwargs):
        batch = api(method, path, **kwargs)
        if path == "/batches/batch_1":
            batch.update(status="failed", results=None, error={"message": "x"})
        return batch

    monkeypatch.setattr(openrouter, "openrouter_request", first_batch_failed)
    api.status = "completed"
    not_annotated = compare()
    assert not_annotated["pending"] == []
    (failed,) = not_annotated["failed"]
    annotated = {"openai_model-x.csv", "openai_model-y.csv"} - {failed}
    assert annotation_files(tmp_path) == sorted(
        ["combined_ap.json", annotated.pop().replace(".csv", "_ap.json")]
    )
    assert live_api.requests == []

    # CLI: incomplete comparisons are reported by the exit code
    args = ["ff-model-comparison", "-p", "Prompt", "-o", str(tmp_path / "cli")]
    args += ["-m", "openai/reference", "openai/model-x", "-r", "openai/reference"]
    args += ["--annotator-model", MODEL, "--batch-annotation"]
    monkeypatch.setattr("sys.argv", args + ["--batch-max-wait", "0"])
    monkeypatch.setattr(openrouter, "openrouter_request", api)
    api.status = "in_progress"
    assert ff_model_comparison.run() == 3
    api.status = "completed"
    assert ff_model_comparison.run() == 0


def test_model_comparison_without_batch_variant_uses_live_api(
    tmp_path, fake_api, live_api, compare
):
    api = fake_api()
    for annotator_model in ("openrouter/mock/other", "openai/mock-annotator"):
        output_path = tmp_path / annotator_model.split("/")[0]
        ff_model_comparison.compare_models(
            prompts=["Prompt"],
            model_names=["openai/reference", "openai/model-x"],
            reference_models=["openai/reference"],
            output_path=str(output_path),
            annotator_model=annotator_model,
            batch_annotation=True,
        )
        assert annotation_files(output_path) == [
            "combined_ap.json",
            "openai_model-x_ap.json",
        ]
    assert api.batches == {}
    assert len(live_api.requests) >= 2

"""Tests of the vote cache (resuming runs) and of API usage/cost accounting."""

import ast
import itertools
import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from feedback_forensics.annotation import (
    annotate_dataset,
    get_default_principles,
    pipeline,
    usage,
    voting,
)
from feedback_forensics.annotation.cache import get_vote_hash
from feedback_forensics.annotation.mock_api import (
    MALFORMED_RANGE,
    MARKDOWN_RANGE,
    TRUNCATED_RANGE,
    _h,
    pair_key,
)
from feedback_forensics.annotation.testutils import (
    MOCK_MODEL,
    expected_vote,
    make_pairs_df,
)
from feedback_forensics.annotation.usage import UsageTracker, summarise_usage
from feedback_forensics.annotation.voting import (
    INVALID_VOTE,
    SHOWN_FIRST_VALUES,
    VotingConfig,
    get_preferred_text,
    get_rejected_text,
)

PRINCIPLES = get_default_principles()


def modify_responses(monkeypatch, server, modify):
    """Change the mock's responses: `modify(body, status, payload) -> (status, payload)`."""
    handle = server._handle
    monkeypatch.setattr(server, "_handle", lambda body: modify(body, *handle(body)))


def without_unparsable_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Comparisons for which the mock's response is parsable."""
    unparsable = df.apply(
        lambda row: _h(pair_key(row["text_a"], row["text_b"])) % 100
        in (*MALFORMED_RANGE, *MARKDOWN_RANGE, *TRUNCATED_RANGE),
        axis=1,
    )
    return df[~unparsable].reset_index(drop=True)


def read_votes(output_dir) -> dict:
    df = pd.read_csv(output_dir / "results" / "040_votes_per_comparison.csv")
    return {row["index"]: ast.literal_eval(row["votes"]) for _, row in df.iterrows()}


def expected_votes(df, principles, position_debias=False) -> dict:
    return {
        idx: {
            key: expected_vote(
                principle,
                get_preferred_text(row),
                get_rejected_text(row),
                position_debias,
            )
            for key, principle in enumerate(principles)
        }
        for idx, row in df.iterrows()
    }


def read_summary(output_dir) -> dict:
    return json.load(open(output_dir / "results" / "080_api_usage_summary.json"))


def read_log(output_dir) -> list[dict]:
    """Entries of the API call log (start and result of each call)."""
    return [json.loads(line) for line in open(output_dir / "api_calls.jsonl")]


# --- Vote cache ---------------------------------------------------------------


def test_vote_hash():
    args = dict(
        preferred="a",
        rejected="b",
        principle="p",
        model_name="openai/m",
        max_output_tokens=3000,
        reasoning_effort=None,
    )
    vote_hash = get_vote_hash(**args)
    assert len(vote_hash) == 32  # 128 bits, no truncation
    assert get_vote_hash(**args) == vote_hash
    # Every input the vote depends on is part of the key
    for name, value in [
        ("preferred", "x"),
        ("rejected", "x"),
        ("principle", "x"),
        ("model_name", "openai/x"),
        ("max_output_tokens", 8000),
        ("reasoning_effort", "low"),
    ]:
        assert get_vote_hash(**{**args, name: value}) != vote_hash, name
    keys = {
        get_vote_hash(**args, shown_first=s) for s in (None, "preferred", "rejected")
    }
    assert len(keys) == 3


@pytest.mark.parametrize(
    "position_debias, preferred_texts",
    [
        (False, ("Candidate answer number 46429.", "Candidate answer number 165383.")),
        (True, ("Candidate answer number 47209.", "Candidate answer number 116135.")),
    ],
)
def test_comparisons_with_colliding_32bit_hashes_get_own_votes(
    tmp_path, mock_llm_server, position_debias, preferred_texts
):
    """Regression: vote keys were truncated to 32 bits, so comparisons whose keys
    collided got the same vote (also when resuming). The keys of these two
    comparisons share their first 32 bits."""
    server = mock_llm_server()
    principle = "Select the response that is more concise"
    rejected = "A fixed rejected answer that is fairly long and wordy."
    settings = dict(max_output_tokens=3000, reasoning_effort=None)

    def key_prefixes(preferred: str) -> set[str]:
        return {
            get_vote_hash(
                preferred=preferred,
                rejected=rejected,
                principle=principle,
                model_name=MOCK_MODEL,
                shown_first=shown_first,
                **settings,
            )[:8]
            for shown_first in (SHOWN_FIRST_VALUES if position_debias else [None])
        }

    # The test data (found by trying "Candidate answer number <i>." for i up to
    # 200k) must be updated if the keys change, e.g. if they cover new settings.
    assert key_prefixes(preferred_texts[0]) & key_prefixes(preferred_texts[1])
    df = pd.DataFrame(
        {"text_a": preferred_texts, "text_b": rejected, "preferred_text": "text_a"}
    )
    df.to_csv(tmp_path / "data.csv", index=False)
    expected = expected_votes(df, [principle], position_debias)
    assert expected[0] != expected[1]

    for _ in range(2):  # 2nd run: votes from the cache
        server.reset()
        annotate_dataset(
            tmp_path / "data.csv",
            [principle],
            MOCK_MODEL,
            output_dir=tmp_path / "out",
            position_debias=position_debias,
            **settings,
        )
        assert read_votes(tmp_path / "out") == expected
    assert server.requests == []


def test_votes_are_only_reused_for_the_same_settings(tmp_path, mock_llm_server):
    """Regression: the cache ignored max_output_tokens and reasoning_effort, so
    re-running with other settings returned the old votes (and the new settings
    were recorded in the output's metadata)."""
    server = mock_llm_server()
    df = without_unparsable_pairs(make_pairs_df(8, seed="s"))
    df.to_csv(tmp_path / "data.csv", index=False)

    def run(**settings) -> list[dict]:
        server.reset()
        ap_path = annotate_dataset(
            tmp_path / "data.csv",
            PRINCIPLES,
            MOCK_MODEL,
            output_dir=tmp_path / "out",
            **settings,
        )
        metadata = json.load(open(ap_path))["metadata"]["annotation"]
        assert metadata["max_output_tokens"] == settings["max_output_tokens"]
        assert metadata["reasoning_effort"] == settings.get("reasoning_effort")
        assert read_votes(tmp_path / "out") == expected_votes(df, PRINCIPLES)
        return server.requests

    assert len(run(max_output_tokens=2000)) == len(df)
    assert run(max_output_tokens=2000) == []  # all votes from the cache
    requests = run(max_output_tokens=4000)
    assert [r["max_completion_tokens"] for r in requests] == [4000] * len(df)
    requests = run(max_output_tokens=4000, reasoning_effort="low")
    assert [r["reasoning_effort"] for r in requests] == ["low"] * len(df)
    assert run(max_output_tokens=4000, reasoning_effort="low") == []
    assert run(max_output_tokens=2000) == []


def test_rerun_with_more_output_tokens_repeats_truncated_calls(
    tmp_path, mock_llm_server, monkeypatch, log_messages
):
    """Regression: votes of truncated responses were cached as invalid, so
    following the advice to re-run with a larger --max-output-tokens made no API
    calls and returned the same invalid votes."""
    server = mock_llm_server()

    def truncate_below_1000_tokens(body, status, payload):
        # Like a reasoning model running out of output tokens while reasoning
        if body["max_completion_tokens"] < 1000:
            payload["choices"][0]["message"]["content"] = ""
            payload["choices"][0]["finish_reason"] = "length"
        return status, payload

    modify_responses(monkeypatch, server, truncate_below_1000_tokens)
    df = make_pairs_df(8, seed="t")
    df.to_csv(tmp_path / "data.csv", index=False)
    args = (tmp_path / "data.csv", PRINCIPLES, MOCK_MODEL)

    annotate_dataset(*args, output_dir=tmp_path / "out", max_output_tokens=500)
    votes = [v for vs in read_votes(tmp_path / "out").values() for v in vs.values()]
    assert set(votes) == {INVALID_VOTE}
    warnings = [m for level, m in log_messages if level == "WARNING"]

    server.reset()
    annotate_dataset(*args, output_dir=tmp_path / "out", max_output_tokens=2000)
    assert len(server.requests) == len(df)
    assert read_votes(tmp_path / "out") == expected_votes(df, PRINCIPLES)
    summary = read_summary(tmp_path / "out")
    assert summary["num_calls"] == 2 * len(df)  # both runs
    assert (
        summary["num_truncated_calls"]
        == len(df) + summary["this_run"]["num_truncated_calls"]
    )
    # The first run warned about the truncated calls
    assert any(
        f"{len(votes)}/{len(votes)} votes in the output are invalid" in m
        and f"{len(df)} hit the max output tokens" in m
        for m in warnings
    )


def test_failed_calls_are_repeated_when_resuming(
    tmp_path, mock_llm_server, monkeypatch, retry_waits, log_messages
):
    """Regression: votes of failed API calls (e.g. out of credits) were cached as
    invalid, so re-running never repeated these calls."""
    failing_server = mock_llm_server()
    modify_responses(
        monkeypatch,
        failing_server,
        lambda body, status, payload: (
            402,
            {"error": {"message": "Insufficient credits", "code": 402}},
        ),
    )
    df = make_pairs_df(8, seed="f")
    df.to_csv(tmp_path / "data.csv", index=False)
    args = (tmp_path / "data.csv", PRINCIPLES, MOCK_MODEL)

    annotate_dataset(*args, output_dir=tmp_path / "out")
    votes = [v for vs in read_votes(tmp_path / "out").values() for v in vs.values()]
    assert set(votes) == {INVALID_VOTE}
    assert read_summary(tmp_path / "out")["num_failed_calls"] == len(df)
    assert retry_waits == []  # not retried (would fail again)
    assert any(
        "Cost: none reported (no successful calls)" in m for _, m in log_messages
    )

    server = mock_llm_server()  # provider works again
    annotate_dataset(*args, output_dir=tmp_path / "out")
    assert len(server.requests) == len(df)
    assert read_votes(tmp_path / "out") == expected_votes(df, PRINCIPLES)


@pytest.mark.parametrize("position_debias", [False, True])
def test_rerun_without_position_debias_removes_per_order_votes(
    tmp_path, mock_llm_server, position_debias
):
    """Regression: a run without position debiasing left the per-order votes of
    an earlier debiased run in the same output directory (not matching 040/070)."""
    mock_llm_server()
    make_pairs_df(4, seed="o").to_csv(tmp_path / "data.csv", index=False)
    per_order_path = tmp_path / "out/results/041_votes_per_comparison_per_order.csv"
    for debias in (True, position_debias):
        annotate_dataset(
            tmp_path / "data.csv",
            PRINCIPLES[:3],
            MOCK_MODEL,
            output_dir=tmp_path / "out",
            position_debias=debias,
        )
        assert per_order_path.exists() == debias


# --- Settings validation --------------------------------------------------------


@pytest.fixture
def no_annotation(monkeypatch):
    """Fail the test if annotation starts."""

    def get_votes_for_principles(**kwargs):
        pytest.fail("Annotation started with invalid settings")

    monkeypatch.setattr(pipeline, "get_votes_for_principles", get_votes_for_principles)


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize(
    "setting", ["max_concurrent_tasks", "max_principles_per_prompt"]
)
def test_invalid_settings_rejected_before_writing_files(
    tmp_path, no_annotation, setting, value
):
    """Regression: max_concurrent_tasks=0 made all API calls wait forever, other
    invalid values failed only after output files had been written."""
    with pytest.raises(ValueError, match=setting):
        VotingConfig(model=MOCK_MODEL, **{setting: value})

    make_pairs_df(2).to_csv(tmp_path / "data.csv", index=False)
    with pytest.raises(ValueError, match=setting):
        annotate_dataset(
            tmp_path / "data.csv",
            PRINCIPLES,
            MOCK_MODEL,
            output_dir=tmp_path / "out",
            **{setting: value},
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "principles",
    [
        "Select the response that is more concise",
        # E.g. ICAI's 030_distilled_principles_per_cluster.json
        {"0": "Select the response that is more concise"},
        ["Select the response that is more concise", 1],
    ],
    ids=["str", "dict", "not_str"],
)
def test_invalid_principles_rejected_before_writing_files(
    tmp_path, no_annotation, principles
):
    """Regression: a string was annotated with each of its characters as a
    principle, a dict with its keys."""
    make_pairs_df(2).to_csv(tmp_path / "data.csv", index=False)
    with pytest.raises(TypeError, match="principles"):
        annotate_dataset(
            tmp_path / "data.csv", principles, MOCK_MODEL, output_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


# --- API usage and cost ---------------------------------------------------------


def assert_usage_as_reported(usage: dict, responses: list[tuple[int, dict]]) -> None:
    """Usage totals equal the sums of what the (mock) provider reported.

    Assumes one response per API call (no retried errors): errors are failed calls.
    """
    reported = [body["usage"] for status, body in responses if status == 200]
    assert usage["num_calls"] == len(responses)
    assert usage["num_failed_calls"] == len(responses) - len(reported)
    assert usage["input_tokens"] == sum(u["prompt_tokens"] for u in reported)
    assert usage["output_tokens"] == sum(u["completion_tokens"] for u in reported)
    assert usage["reasoning_tokens"] == sum(
        u["completion_tokens_details"]["reasoning_tokens"] for u in reported
    )
    costs = [u["cost"] for u in reported if "cost" in u]
    assert usage["num_calls_with_cost"] == len(costs)
    assert usage["cost_usd"] == (pytest.approx(sum(costs)) if costs else None)
    assert usage["cost_complete"] == (0 < len(costs) == len(reported))


class Interrupted(Exception):
    """Simulated interruption of a run (e.g. Ctrl-C)."""


def test_usage_summary_covers_interrupted_and_resumed_run(
    tmp_path, mock_llm_server, monkeypatch, log_messages
):
    """Regression: the usage summary only covered the last run, so after resuming
    an interrupted run it under-reported the cost (while claiming it complete)."""
    server = mock_llm_server()
    df = make_pairs_df(16, seed="i")
    df.to_csv(tmp_path / "data.csv", index=False)
    args = (tmp_path / "data.csv", PRINCIPLES, MOCK_MODEL)

    get_vote = voting.get_vote_for_single_comparison
    calls = itertools.count()

    async def get_vote_or_interrupt(*args, **kwargs):
        if next(calls) >= 5:
            raise Interrupted()
        return await get_vote(*args, **kwargs)

    monkeypatch.setattr(voting, "get_vote_for_single_comparison", get_vote_or_interrupt)
    with pytest.raises(Interrupted):
        annotate_dataset(*args, output_dir=tmp_path / "out", max_concurrent_tasks=1)
    monkeypatch.setattr(voting, "get_vote_for_single_comparison", get_vote)
    assert len(server.responses) == 5
    assert not (tmp_path / "out/results/080_api_usage_summary.json").exists()

    annotate_dataset(*args, output_dir=tmp_path / "out")
    summary = read_summary(tmp_path / "out")
    assert_usage_as_reported(summary, server.responses)  # both runs
    assert_usage_as_reported(summary["this_run"], server.responses[5:])
    assert summary["num_calls"] > summary["this_run"]["num_calls"] >= len(df) - 5
    assert summary["cost_complete"] is True
    assert summary["num_unfinished_calls"] == 0
    results = [e for e in read_log(tmp_path / "out") if not e.get("started")]
    assert summary["num_calls"] == len(results)
    messages = [m for _, m in log_messages]
    assert any(
        m.startswith(
            f"API usage of all runs with this output directory: "
            f"{summary['num_calls']} calls"
        )
        for m in messages
    )


class InterruptedInFlight(BaseException):
    """Simulated interruption (e.g. Ctrl-C) while an API call is in flight: not an
    error of the call (like the cancellation of the call's task by asyncio)."""


def test_usage_summary_counts_calls_interrupted_in_flight(
    tmp_path, mock_llm_server, monkeypatch, log_messages
):
    """Regression: calls in flight when a run was interrupted were not logged, so
    after resuming, the summary claimed a complete cost although the provider had
    billed more."""
    server = mock_llm_server()
    df = make_pairs_df(16, seed="i")
    df.to_csv(tmp_path / "data.csv", index=False)
    args = (tmp_path / "data.csv", PRINCIPLES, MOCK_MODEL)

    run_call = voting.run_with_http_retries
    calls = itertools.count()

    async def interrupt_before_response_arrives(*args, **kwargs):
        response = await run_call(*args, **kwargs)  # sent, and billed
        # The last call (so that no further call starts before the interruption)
        if next(calls) == len(df) - 1:
            raise InterruptedInFlight()
        return response

    monkeypatch.setattr(
        voting, "run_with_http_retries", interrupt_before_response_arrives
    )
    with pytest.raises(InterruptedInFlight):
        annotate_dataset(*args, output_dir=tmp_path / "out", max_concurrent_tasks=1)
    monkeypatch.setattr(voting, "run_with_http_retries", run_call)
    assert len(server.responses) == len(df)
    unlogged_cost = server.responses[-1][1]["usage"]["cost"]

    annotate_dataset(*args, output_dir=tmp_path / "out")
    summary = read_summary(tmp_path / "out")
    assert summary["num_unfinished_calls"] == 1
    assert summary["num_calls"] == len(server.responses) - 1
    billed_cost = sum(body["usage"]["cost"] for _, body in server.responses)
    assert summary["cost_usd"] == pytest.approx(billed_cost - unlogged_cost)
    assert summary["cost_complete"] is False  # a lower bound
    assert summary["this_run"]["num_unfinished_calls"] == 0
    assert summary["this_run"]["cost_complete"] is True
    assert any(
        m.startswith(
            f"API usage of all runs with this output directory: "
            f"{summary['num_calls']} calls (+ 1 unfinished call of interrupted runs"
        )
        and "(lower bound)" in m
        for _, m in log_messages
    )


def test_usage_summary_of_rerun_without_api_calls(
    tmp_path, mock_llm_server, log_messages
):
    """Regression: re-running a finished run replaced the usage totals with zeros,
    claimed that the provider reported no cost and did not warn about the invalid
    votes still in the output."""
    server = mock_llm_server()
    df = without_unparsable_pairs(make_pairs_df(8, seed="z"))
    df.to_csv(tmp_path / "data.csv", index=False)
    args = (tmp_path / "data.csv", PRINCIPLES, MOCK_MODEL)

    annotate_dataset(*args, output_dir=tmp_path / "out")
    first = read_summary(tmp_path / "out")
    server.reset()
    log_messages.clear()
    annotate_dataset(*args, output_dir=tmp_path / "out")
    second = read_summary(tmp_path / "out")

    assert server.responses == []
    this_run = second.pop("this_run", None)
    assert second == {k: v for k, v in first.items() if k != "this_run"}
    assert second["num_calls"] == len(df) and second["cost_complete"] is True
    assert this_run["num_calls"] == 0 and this_run["cost_usd"] is None
    messages = [m for _, m in log_messages]
    assert (
        "API usage of this run: no API calls (all votes were loaded from the vote "
        "cache)." in messages
    )
    assert any(
        m.startswith(
            f"API usage of all runs with this output directory: {len(df)} calls"
        )
        and f"Cost: ${second['cost_usd']:.4f}" in m
        for m in messages
    )
    assert not any("not reported" in m for m in messages)
    # Invalid votes (per principle) are still in the output
    assert second["num_invalid_votes"] > 0
    warnings = [m for level, m in log_messages if level == "WARNING"]
    assert any(
        f"{second['num_invalid_votes']}/{second['num_votes']} votes in the output "
        "are invalid" in m
        for m in warnings
    )


def test_usage_with_failed_calls_and_partially_reported_cost(
    tmp_path, mock_llm_server, monkeypatch, retry_waits, log_messages
):
    server = mock_llm_server()

    def fail_or_omit_cost(body, status, payload):
        behaviour = _h("usage", json.dumps(body["messages"])) % 4
        if behaviour == 0:  # rejected (not retried)
            return 400, {"error": {"message": "Mock bad request", "code": 400}}
        if behaviour == 1:
            del payload["usage"]["cost"]
        return status, payload

    modify_responses(monkeypatch, server, fail_or_omit_cost)
    df = make_pairs_df(16, seed="p")
    df.to_csv(tmp_path / "data.csv", index=False)
    annotate_dataset(
        tmp_path / "data.csv", PRINCIPLES, MOCK_MODEL, output_dir=tmp_path / "out"
    )

    summary = read_summary(tmp_path / "out")
    statuses = [status for status, _ in server.responses]
    reported = [body["usage"] for status, body in server.responses if status == 200]
    num_with_cost = sum("cost" in u for u in reported)
    assert 0 < num_with_cost < len(reported) and 400 in statuses
    assert summary["num_calls"] == len(df)
    assert_usage_as_reported(summary, server.responses)
    assert summary["num_failed_calls"] == statuses.count(400)
    assert retry_waits == []  # not retried (would fail again)
    assert summary["cost_complete"] is False
    messages = [m for _, m in log_messages]
    assert any(
        f"(lower bound: reported for only {num_with_cost}/{len(reported)} "
        "successful calls)" in m
        for m in messages
    )
    warnings = [m for level, m in log_messages if level == "WARNING"]
    assert any(f"{statuses.count(400)} failed (no response" in m for m in warnings)


def test_usage_with_cost_not_reported(
    tmp_path, mock_llm_server, monkeypatch, log_messages
):
    server = mock_llm_server()

    def omit_cost(body, status, payload):
        del payload["usage"]["cost"]
        return status, payload

    modify_responses(monkeypatch, server, omit_cost)
    make_pairs_df(4, seed="n").to_csv(tmp_path / "data.csv", index=False)
    annotate_dataset(
        tmp_path / "data.csv", PRINCIPLES, MOCK_MODEL, output_dir=tmp_path / "out"
    )

    summary = read_summary(tmp_path / "out")
    assert summary["num_calls"] > 0
    assert_usage_as_reported(summary, server.responses)
    assert summary["cost_usd"] is None and summary["cost_complete"] is False
    assert any("Cost: not reported by provider" in m for _, m in log_messages)


def test_usage_summary_of_run_without_votes(tmp_path, mock_llm_server, log_messages):
    server = mock_llm_server()
    make_pairs_df(2, seed="w").to_csv(tmp_path / "data.csv", index=False)
    annotate_dataset(tmp_path / "data.csv", [], MOCK_MODEL, output_dir=tmp_path / "out")

    assert server.requests == []
    # Not: "(all votes were loaded from the vote cache)"
    assert "API usage of this run: no API calls." in [m for _, m in log_messages]


def test_unparsable_responses_counted(tmp_path, mock_llm_server, monkeypatch):
    """Also responses that are valid JSON, but whose votes can't be assigned to the
    principles."""
    server = mock_llm_server()

    def votes_by_name(body, status, payload):
        payload["choices"][0]["message"]["content"] = '{"concise": "A"}'
        return status, payload

    modify_responses(monkeypatch, server, votes_by_name)
    make_pairs_df(4, seed="k").to_csv(tmp_path / "data.csv", index=False)
    annotate_dataset(
        tmp_path / "data.csv", PRINCIPLES, MOCK_MODEL, output_dir=tmp_path / "out"
    )

    this_run = read_summary(tmp_path / "out")["this_run"]
    assert this_run["num_unparsable_responses"] == len(server.responses) > 0


def ai_message(input_tokens: int, output_tokens: int) -> AIMessage:
    """A model response (as returned by LangChain) with the given token usage."""
    return AIMessage(
        content="{}",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"finish_reason": "stop"},
    )


def test_cost_at_list_prices(tmp_path, monkeypatch):
    """Computed with each model's list price, if known for all models."""
    prices = {
        "a/model": {"prompt": 1e-6, "completion": 2e-6},
        "b/model": {"prompt": 3e-6, "completion": 5e-6},
    }
    monkeypatch.setattr(usage, "get_openrouter_pricing", prices.get)
    tracker = UsageTracker(tmp_path / "api_calls.jsonl")
    for model, input_tokens, output_tokens in [
        ("a/model", 100, 10),
        ("b/model", 200, 20),
        ("a/model", 300, 30),
    ]:
        tracker.record(
            f"openrouter/{model}", response=ai_message(input_tokens, output_tokens)
        )

    summary = summarise_usage(tracker, num_votes=0, num_invalid_votes=0)
    assert summary["list_price_cost_usd"] == pytest.approx(
        400 * 1e-6 + 40 * 2e-6 + 200 * 3e-6 + 20 * 5e-6
    )
    assert summary["list_price_per_token"] == {
        f"openrouter/{model}": price for model, price in prices.items()
    }

    tracker.record("openrouter/unknown/model", response=ai_message(1, 1))
    summary = summarise_usage(tracker, num_votes=0, num_invalid_votes=0)
    assert summary["list_price_cost_usd"] is None


def test_usage_log_with_incomplete_last_line(tmp_path, log_messages):
    """Regression: if the log's last line was incomplete (e.g. the process was killed
    while writing it), the first call of the next run was appended to that line,
    so it was missing from the totals. Reading the log even failed if the line
    ended within a multi-byte character."""
    log_path = tmp_path / "api_calls.jsonl"
    UsageTracker(log_path).record("openai/m", response=ai_message(100, 10))
    with open(log_path, "ab") as f:
        f.write('{"model": "openai/m", "error": "Ü'.encode("utf-8")[:-1])

    UsageTracker(log_path).record("openai/m", response=ai_message(200, 20))

    total = UsageTracker.from_log(log_path)
    assert (total.num_calls, total.input_tokens) == (2, 300)
    warnings = [m for level, m in log_messages if level == "WARNING"]
    assert any(m.startswith("Skipping malformed line") for m in warnings)

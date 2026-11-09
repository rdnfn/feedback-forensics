"""Tests for native principle-following annotation (against a mock LLM API)."""

import ast
import hashlib
import json
from collections import Counter

import pandas as pd
import pytest

from feedback_forensics.annotation import annotate_dataset, get_default_principles
from feedback_forensics.annotation.mock_api import (
    MALFORMED_RANGE,
    MARKDOWN_RANGE,
    PRINCIPLE_POSITIONAL,
    TRUNCATED_RANGE,
)
from feedback_forensics.annotation.mock_api import _h, pair_key, principle_behaviour
from feedback_forensics.annotation.prompts import VOTING_PROMPT
from feedback_forensics.annotation.testutils import (
    MOCK_MODEL,
    expected_order_vote,
    expected_vote,
    make_pairs_df,
)
from feedback_forensics.annotation.voting import (
    INVALID_VOTE,
    POSITION_INCONSISTENT_VOTE,
    combine_position_votes,
    get_preferred_text,
    get_rejected_text,
)
from feedback_forensics.data.ap_format import hash_comparison, hash_string
from feedback_forensics.tools import ff_annotate

EXTRA_PRINCIPLES = [
    "Select the response that is more confident",
    "Select the response that sounds more like a pirate",
]


def read_votes(output_dir) -> dict:
    df = pd.read_csv(output_dir / "results" / "040_votes_per_comparison.csv")
    return {row["index"]: ast.literal_eval(row["votes"]) for _, row in df.iterrows()}


def assert_votes_match_mock(df, output_dir, principles, position_debias, bias=False):
    """Votes must equal those implied by the (content-deterministic) mock."""
    votes = read_votes(output_dir)
    assert set(votes) == set(df.index)
    for idx, row in df.iterrows():
        preferred, rejected = get_preferred_text(row), get_rejected_text(row)
        for key, principle in enumerate(principles):
            expected = expected_vote(
                principle, preferred, rejected, position_debias, position_bias=bias
            )
            assert votes[idx][key] == expected, (idx, principle)


def test_voting_prompt_unchanged():
    """The voting prompt must not change (comparability with ICAI/published data).

    Note: the prompt intentionally contains a trailing space.
    """
    assert (
        hashlib.md5(VOTING_PROMPT.encode("utf-8")).hexdigest()
        == "7c9c899821aa75c26019d4afb4e58a54"
    )


def test_default_principles_and_hashes_match_icai():
    icai_ap = pytest.importorskip("inverse_cai.data.annotated_pairs_format")
    icai_principles = pytest.importorskip(
        "inverse_cai.experiment.config.default_principles"
    )
    icai_prompts = pytest.importorskip("inverse_cai.experiment.config.prompts")

    assert get_default_principles("v1") == icai_principles.DEFAULT_PRINCIPLES["v4"]
    assert VOTING_PROMPT == icai_prompts.DEFAULT_VOTING_PROMPT
    for s in ["", "abc", "Select the response that is more concise", "ünïcödé 🎭"]:
        assert hash_string(s) == icai_ap.hash_string(s)
    for prompt in [None, "A prompt"]:
        args = ({"text": "a", "model": "m1"}, {"text": "b"}, prompt)
        assert hash_comparison(*args) == icai_ap.hash_comparison(*args)


def test_default_principles_versions():
    assert len(get_default_principles()) == 40
    assert get_default_principles("") == []
    with pytest.raises(ValueError):
        get_default_principles("v999")


@pytest.mark.parametrize(
    "first, second, expected",
    [
        (True, True, True),
        (False, False, False),
        (None, None, None),
        ("Both", "Both", "Both"),
        (True, False, POSITION_INCONSISTENT_VOTE),
        (True, None, POSITION_INCONSISTENT_VOTE),
        ("Both", True, POSITION_INCONSISTENT_VOTE),
        (INVALID_VOTE, True, INVALID_VOTE),
        (None, INVALID_VOTE, INVALID_VOTE),
        (INVALID_VOTE, INVALID_VOTE, INVALID_VOTE),
    ],
)
def test_combine_position_votes(first, second, expected):
    assert combine_position_votes(first, second) == expected


def test_repeated_runs_in_one_process(tmp_path, mock_llm_server):
    """Regression test: multiple runs (and passes) in one process with keep-alive
    connections and transient errors must not lose any votes.

    LangChain's shared HTTP client fails here ("Event loop is closed") on the
    2nd `asyncio.run()` in a process; transient errors must be retried.
    """
    server = mock_llm_server(keep_alive=True, fail_first_attempts=True)
    df = make_pairs_df(30, seed="r")
    df.to_csv(tmp_path / "data.csv", index=False)
    principles = get_default_principles() + EXTRA_PRINCIPLES  # two passes

    for run in range(3):
        output_dir = tmp_path / f"run{run}"
        annotate_dataset(
            tmp_path / "data.csv", principles, MOCK_MODEL, output_dir=output_dir
        )
        usage = json.load(open(output_dir / "results" / "080_api_usage_summary.json"))
        assert usage["num_failed_calls"] == 0
        assert usage["num_calls"] == 2 * len(df)
        assert_votes_match_mock(df, output_dir, principles, position_debias=False)
    assert server.num_failed_attempts > 0  # retries were exercised


def test_runs_are_reproducible(tmp_path, mock_llm_server):
    """Unlike ICAI, the A/B order (and thus every prompt) is reproducible."""
    server = mock_llm_server()
    make_pairs_df(30, seed="s").to_csv(tmp_path / "data.csv", index=False)
    requests_per_run = []
    for run in range(2):
        server.reset()
        annotate_dataset(
            tmp_path / "data.csv",
            get_default_principles(),
            MOCK_MODEL,
            output_dir=tmp_path / f"run{run}",
        )
        requests_per_run.append(
            sorted(json.dumps(r, sort_keys=True) for r in server.requests)
        )
    assert requests_per_run[0] == requests_per_run[1]
    # Both orders are used (roughly balanced)
    flipped = Counter(
        json.loads(line)["flipped"] for line in open(tmp_path / "run0/api_calls.jsonl")
    )
    assert flipped[True] > 5 and flipped[False] > 5


def test_position_debias(tmp_path, mock_llm_server):
    """With a position-biased annotator, biased votes are discarded and all other
    votes are kept, with per-order votes saved separately."""
    server = mock_llm_server(position_bias=True)
    df = make_pairs_df(40, seed="d")
    df.to_csv(tmp_path / "data.csv", index=False)
    principles = get_default_principles()

    ap_path = annotate_dataset(
        tmp_path / "data.csv",
        principles,
        MOCK_MODEL,
        output_dir=tmp_path / "out",
        position_debias=True,
    )
    assert len(server.requests) == 2 * len(df)  # both orders for each pair
    assert_votes_match_mock(
        df, tmp_path / "out", principles, position_debias=True, bias=True
    )

    # Positionally biased (always "A") votes are inconsistent across orders,
    # unless the pair response is invalid anyway.
    votes = read_votes(tmp_path / "out")
    num_positional = 0
    for idx, row in df.iterrows():
        preferred, rejected = get_preferred_text(row), get_rejected_text(row)
        pair_behaviour = _h(pair_key(preferred, rejected)) % 100
        if pair_behaviour in (*MALFORMED_RANGE, *MARKDOWN_RANGE, *TRUNCATED_RANGE):
            continue
        for key, principle in enumerate(principles):
            if (
                principle_behaviour(principle, preferred, rejected)
                in PRINCIPLE_POSITIONAL
            ):
                assert votes[idx][key] == POSITION_INCONSISTENT_VOTE
                num_positional += 1
    assert num_positional > 0

    # per-order votes
    per_order = pd.read_csv(
        tmp_path / "out/results/041_votes_per_comparison_per_order.csv"
    )
    assert list(per_order.columns) == ["index", "preferred_first", "rejected_first"]
    for _, order_row in per_order.iterrows():
        row = df.loc[order_row["index"]]
        preferred, rejected = get_preferred_text(row), get_rejected_text(row)
        for col, preferred_first in [
            ("preferred_first", True),
            ("rejected_first", False),
        ]:
            order_votes = ast.literal_eval(order_row[col])
            for key, principle in enumerate(principles):
                assert order_votes[key] == expected_order_vote(
                    principle, preferred, rejected, preferred_first, position_bias=True
                )

    # AP output: inconsistent votes stored without preference
    ap = json.load(open(ap_path))
    reasons = Counter(
        a.get("no_pref_reason")
        for c in ap["comparisons"]
        for a in c["annotations"].values()
    )
    assert reasons["position_inconsistent"] == sum(
        v == POSITION_INCONSISTENT_VOTE for vs in votes.values() for v in vs.values()
    )
    assert ap["metadata"]["annotation"]["position_debias"] is True


def test_usage_and_cost_logging(tmp_path, mock_llm_server):
    server = mock_llm_server()
    df = make_pairs_df(30, seed="u")
    df.to_csv(tmp_path / "data.csv", index=False)
    annotate_dataset(
        tmp_path / "data.csv",
        get_default_principles(),
        MOCK_MODEL,
        output_dir=tmp_path / "out",
        log_api_calls=True,
    )

    entries = [json.loads(line) for line in open(tmp_path / "out/api_calls.jsonl")]
    assert len(entries) == len(server.requests) == len(df)

    # Totals match what the (mock) provider reported
    summary = json.load(open(tmp_path / "out/results/080_api_usage_summary.json"))
    num_truncated = sum(
        _h(pair_key(*(get_preferred_text(r), get_rejected_text(r)))) % 100
        in TRUNCATED_RANGE
        for _, r in df.iterrows()
    )
    assert num_truncated > 0
    assert summary["num_calls"] == len(df)
    assert summary["num_failed_calls"] == 0
    assert summary["num_truncated_calls"] == num_truncated
    assert summary["num_unparsable_responses"] > 0
    assert summary["input_tokens"] == sum(e["input_tokens"] for e in entries) > 0
    assert summary["output_tokens"] == sum(e["output_tokens"] for e in entries) > 0
    assert summary["reasoning_tokens"] == 7 * len(df)
    assert summary["cost_usd"] == pytest.approx(sum(e["cost_usd"] for e in entries))
    assert summary["cost_usd"] > 0 and summary["cost_complete"] is True
    assert summary["list_price_cost_usd"] is None  # only for openrouter models

    # Full request/response content logged with log_api_calls=True
    sent_messages = sorted(json.dumps(r["messages"]) for r in server.requests)
    logged_messages = sorted(json.dumps(e["request_messages"]) for e in entries)
    assert sent_messages == logged_messages
    assert all(isinstance(e["response_text"], str) for e in entries)
    assert all(e["latency_s"] is not None for e in entries)


def test_resume_from_cache(tmp_path, mock_llm_server):
    server = mock_llm_server()
    make_pairs_df(20, seed="c").to_csv(tmp_path / "data.csv", index=False)
    args = (tmp_path / "data.csv", get_default_principles(), MOCK_MODEL)

    first = json.load(open(annotate_dataset(*args, output_dir=tmp_path / "out")))
    num_requests = len(server.requests)
    second = json.load(open(annotate_dataset(*args, output_dir=tmp_path / "out")))
    assert len(server.requests) == num_requests  # all votes from cache
    assert first["comparisons"] == second["comparisons"]


def test_cli(tmp_path, mock_llm_server):
    server = mock_llm_server()
    df = make_pairs_df(10, with_prompt=True, seed="x")
    df.to_csv(tmp_path / "data.csv", index=False)

    exit_code = ff_annotate.run(
        [
            "-d",
            str(tmp_path / "data.csv"),
            "-o",
            str(tmp_path / "out"),
            "-m",
            MOCK_MODEL,
            "-v",
            "",
            "-p",
            json.dumps(EXTRA_PRINCIPLES),
            "--position-debias",
            "--max-output-tokens",
            "123",
        ]
    )
    assert exit_code == 0
    ap = json.load(open(tmp_path / "out/results/070_annotations_train_ap.json"))
    principle_annotators = [
        a["description"] for a in ap["annotators"].values() if a["type"] == "principle"
    ]
    assert principle_annotators == EXTRA_PRINCIPLES
    assert len(server.requests) == 2 * len(df)
    assert all(r["max_completion_tokens"] == 123 for r in server.requests)

    assert ff_annotate.run(["-d", str(tmp_path / "data.csv"), "-v", ""]) == 1

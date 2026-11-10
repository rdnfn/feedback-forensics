"""Helpers for annotation tests: fixtures (mock LLM server, log messages), test
data, expected votes."""

import asyncio
import contextlib
import itertools
from types import SimpleNamespace

import backoff
import pandas as pd
import pytest
from loguru import logger

from feedback_forensics.annotation.mock_api import (
    FAIL_ONCE_RANGE,
    MALFORMED_RANGE,
    MARKDOWN_RANGE,
    MISSING,
    TRUNCATED_RANGE,
    MockLLMServer,
    _h,
    mock_vote_value,
    pair_key,
)
from feedback_forensics.annotation.voting import combine_position_votes

MOCK_MODEL = "openai/mock-annotator"


@pytest.fixture
def mock_llm_server(monkeypatch):
    """Factory fixture: start a mock LLM server and point the OpenAI client to it."""

    with contextlib.ExitStack() as stack:

        def start(**kwargs) -> MockLLMServer:
            server = stack.enter_context(MockLLMServer(**kwargs))
            monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
            monkeypatch.setenv("OPENAI_API_BASE", server.base_url)
            monkeypatch.setenv("OPENAI_API_KEY", "mock-key")
            return server

        yield start


@pytest.fixture
def retry_waits(monkeypatch) -> list[float]:
    """Skip the waits between FF's retries (`backoff`), returns the waits."""
    waits = []

    async def sleep(delay, *args, **kwargs):
        waits.append(delay)
        await asyncio.sleep(0)

    # Only for `backoff`: other libraries (e.g. anyio) may import `asyncio.sleep`
    monkeypatch.setattr(
        backoff._async, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": sleep})
    )
    return waits


@pytest.fixture
def log_messages():
    """(level, message) of everything logged (INFO and above) during the test."""
    messages = []
    handler_id = logger.add(
        lambda m: messages.append((m.record["level"].name, m.record["message"])),
        level="INFO",
    )
    yield messages
    logger.remove(handler_id)


def make_pairs_df(
    n: int,
    with_prompt: bool = False,
    with_preferred: bool = True,
    with_extra_columns: bool = False,
    seed: str = "",
) -> pd.DataFrame:
    """Create a synthetic pairwise dataset.

    Contains at least `n` rows. Further rows are added until every pair-level
    behaviour of the mock API (transient error, malformed, markdown-wrapped and
    truncated response) occurs at least once.
    """
    required_ranges = [
        FAIL_ONCE_RANGE,
        MALFORMED_RANGE,
        MARKDOWN_RANGE,
        TRUNCATED_RANGE,
    ]
    covered = set()
    rows = []
    for i in itertools.count():
        row = _make_row(i, with_prompt, with_preferred, with_extra_columns, seed)
        # texts as sent to the API (prompt merged into texts if available)
        texts = [
            (
                str(
                    [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row[col]},
                    ]
                )
                if with_prompt
                else row[col]
            )
            for col in ("text_a", "text_b")
        ]
        pair_behaviour = _h(pair_key(*texts)) % 100
        newly_covered = {
            j for j, r in enumerate(required_ranges) if pair_behaviour in r
        } - covered
        if i < n or newly_covered:
            rows.append(row)
            covered |= newly_covered
        if i >= n - 1 and len(covered) == len(required_ranges):
            break
    return pd.DataFrame(rows)


def _make_row(i, with_prompt, with_preferred, with_extra_columns, seed) -> dict:
    row = {
        "text_a": f"Response {seed}{i} from model A." + " More detail." * (i % 4),
        "text_b": f"Answer {seed}{i} by model B!" + " Extra." * (i % 3),
    }
    if with_prompt:
        row["prompt"] = f"Prompt number {seed}{i}: please explain."
    if with_preferred:
        row["preferred_text"] = "text_a" if (i * 7) % 3 else "text_b"
    if with_extra_columns:
        row["model_a"] = "model-x"
        row["model_b"] = f"model-{i % 2}"
        row["human_label"] = "text_b" if i % 5 else "text_a"
        row["topic"] = f"topic {i % 4}"
    return row


def expected_order_vote(
    principle: str,
    preferred: str,
    rejected: str,
    preferred_shown_first: bool,
    position_bias: bool = False,
):
    """Vote (relative to preferred) that a correct pipeline must derive from the mock."""
    if _h(pair_key(preferred, rejected)) % 100 in (
        *MALFORMED_RANGE,
        *MARKDOWN_RANGE,
        *TRUNCATED_RANGE,
    ):
        return "invalid"
    sample_a, sample_b = (
        (preferred, rejected) if preferred_shown_first else (rejected, preferred)
    )
    raw = mock_vote_value(principle, sample_a, sample_b, position_bias)
    if raw is MISSING or raw == "C":
        return "invalid"
    mapping = {"None": None, "Both": "Both", "neither": "Neither"}
    if raw in mapping:
        return mapping[raw]
    selected = sample_a if raw == "A" else sample_b
    return selected == preferred


def expected_vote(
    principle: str,
    preferred: str,
    rejected: str,
    position_debias: bool,
    position_bias: bool = False,
):
    """Expected final vote. Without position bias in the mock, order is irrelevant."""
    if position_debias:
        return combine_position_votes(
            expected_order_vote(principle, preferred, rejected, True, position_bias),
            expected_order_vote(principle, preferred, rejected, False, position_bias),
        )
    assert not position_bias, "Without debiasing, the vote depends on the random order."
    return expected_order_vote(principle, preferred, rejected, True)

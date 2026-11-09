"""Principle-following annotation ("voting") of pairwise comparisons.

For each comparison, an LLM is asked which of the two responses should be selected
according to each principle. Votes are expressed relative to the reference
(`preferred_text`) annotation:

- True: the principle selects the preferred response
- False: the principle selects the rejected response
- None: the principle is not applicable
- "Both" / "Neither": the principle selects both / neither
- "invalid": no valid vote could be obtained
- "position_inconsistent": (position debiasing only) the vote changed when the
  order of the two responses was swapped

Ported from `inverse_cai.algorithm.voting` (inverse-cai v0.4.8). With
`position_debias=False` the same prompts (incl. principle order and A/B order) are
sent as by ICAI with FF's former `ff-annotate` configuration (one seed, seed 1).
Differences to ICAI: all API calls of a run share one event loop and HTTP client,
network errors are retried, and A/B orders are drawn upfront (reproducible).
"""

import ast
import asyncio
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import backoff
import httpx
import openai
import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm

from feedback_forensics.annotation.cache import VoteCache, get_vote_hash
from feedback_forensics.annotation.prompts import VOTING_PROMPT, parse_prompt
from feedback_forensics.annotation.usage import UsageTracker

INVALID_VOTE = "invalid"
POSITION_INCONSISTENT_VOTE = "position_inconsistent"
SHOWN_FIRST_VALUES = ("preferred", "rejected")

VALID_VOTE_VALUES = {
    "A": True,
    "B": False,
    "Both": "Both",
    "Neither": "Neither",
    "None": None,
    None: None,
}


@dataclass
class VotingConfig:
    """Configuration of principle-following annotation."""

    model: str
    max_principles_per_prompt: int = 40
    # Includes reasoning tokens for reasoning models
    max_output_tokens: int = 3000
    max_concurrent_tasks: int = 400
    seed: int = 1
    # Reasoning effort for reasoning models (None: provider default)
    reasoning_effort: str | None = None
    # Annotate each comparison in both orders (A/B and B/A) and only keep votes
    # that agree across orders. Doubles the number of API calls.
    position_debias: bool = False


@dataclass
class VotingResult:
    """Votes per comparison: df index -> {principle id -> vote}.

    With position debiasing, `votes_per_order` additionally contains the votes
    for each response order before they were combined:
    df index -> {"preferred_first": {...}, "rejected_first": {...}}.
    """

    votes: dict
    votes_per_order: dict | None = None


@dataclass
class _VoteRequest:
    """A single LLM call: one comparison, one set of principles, one order."""

    preferred: str
    rejected: str
    flipped: bool  # True: rejected response shown as sample A
    hashes: dict[str, str]  # principle -> vote hash


def get_preferred_text(row) -> str:
    if row["preferred_text"] == "text_a":
        return row["text_a"]
    elif row["preferred_text"] == "text_b":
        return row["text_b"]
    raise ValueError(f"Invalid preferred_text value: {row['preferred_text']}")


def get_rejected_text(row) -> str:
    if row["preferred_text"] == "text_a":
        return row["text_b"]
    elif row["preferred_text"] == "text_b":
        return row["text_a"]
    raise ValueError(f"Invalid preferred_text value: {row['preferred_text']}")


def get_votes_for_principles(
    feedback_df: pd.DataFrame,
    principles: dict[int, str],
    config: VotingConfig,
    cache_path: Path | None = None,
    usage_tracker: UsageTracker | None = None,
    model=None,
) -> VotingResult:
    """Annotate each comparison in `feedback_df` according to each principle.

    Runs all API calls in a single event loop (use
    `get_votes_for_principles_async` inside an already running loop).

    Args:
        feedback_df: Data with `text_a`, `text_b` and `preferred_text` columns.
        principles: Mapping principle id -> principle text.
        config: Voting configuration.
        cache_path: Base path for the vote cache (None to disable caching).
        usage_tracker: Tracker to record API usage and cost.
        model: LangChain chat model (default: created from `config.model`
            inside the event loop).
    """
    return asyncio.run(
        get_votes_for_principles_async(
            feedback_df=feedback_df,
            principles=principles,
            config=config,
            cache_path=cache_path,
            usage_tracker=usage_tracker,
            model=model,
        )
    )


async def get_votes_for_principles_async(
    feedback_df: pd.DataFrame,
    principles: dict[int, str],
    config: VotingConfig,
    cache_path: Path | None = None,
    usage_tracker: UsageTracker | None = None,
    model=None,
) -> VotingResult:
    """Async version of `get_votes_for_principles`."""
    if usage_tracker is None:
        usage_tracker = UsageTracker()

    args = dict(
        feedback_df=feedback_df,
        principles=principles,
        config=config,
        cache_path=cache_path,
        usage_tracker=usage_tracker,
    )

    if model is not None or len(principles) == 0:
        return await _get_votes(**args, model=model)

    from feedback_forensics.models import get_model

    if config.model.startswith("anthropic/"):
        model = get_model(config.model, max_tokens=config.max_output_tokens)
        return await _get_votes(**args, model=model)

    # Use an HTTP client bound to the current event loop instead of LangChain's
    # process-wide cached client: the cached client's connections break
    # ("Event loop is closed") when used from a later `asyncio.run()`.
    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max(config.max_concurrent_tasks, 10),
            max_keepalive_connections=100,
        ),
        timeout=httpx.Timeout(600.0, connect=30.0),
    ) as http_client:
        model = get_model(
            config.model,
            max_tokens=config.max_output_tokens,
            reasoning_effort=config.reasoning_effort,
            http_async_client=http_client,
        )
        return await _get_votes(**args, model=model)


async def _get_votes(
    feedback_df: pd.DataFrame,
    principles: dict[int, str],
    config: VotingConfig,
    cache_path: Path | None,
    usage_tracker: UsageTracker,
    model,
) -> VotingResult:
    if len(principles) == 0:
        logger.warning("No principles to vote on, skipping voting.")
        return VotingResult(votes={idx: {} for idx in feedback_df.index})

    vote_cache = VoteCache(cache_path) if cache_path is not None else None

    # Same random sequence as ICAI with a single seed: first shuffle the
    # principle order, then (per pass) draw one A/B flip per comparison, in row
    # order. Drawing all flips upfront makes runs reproducible (independent of
    # the scheduling of async API calls and of cache hits).
    rng = random.Random(config.seed)
    principle_keys = list(principles.keys())
    rng.shuffle(principle_keys)
    n = config.max_principles_per_prompt
    principle_parts = [
        [principles[k] for k in principle_keys[i : i + n]]
        for i in range(0, len(principle_keys), n)
    ]
    logger.info(
        f"Annotating {len(feedback_df)} comparisons with {len(principles)} principles "
        f"using {config.model} ({len(principle_parts)} pass(es) over the data, "
        f"position debiasing {'ON' if config.position_debias else 'OFF'}). "
        f"Running up to {config.max_concurrent_tasks} API calls concurrently."
    )

    rows = [
        (idx, get_preferred_text(row), get_rejected_text(row))
        for idx, row in feedback_df.iterrows()
    ]

    def vote_hash(preferred, rejected, principle, shown_first):
        return get_vote_hash(
            preferred=preferred,
            rejected=rejected,
            principle=principle,
            model_name=config.model,
            shown_first=shown_first,
        )

    hashed_votes = {}
    for i, part in enumerate(principle_parts):
        logger.info(f"Starting pass {i+1}/{len(principle_parts)}")
        requests = []
        for _, preferred, rejected in rows:
            if config.position_debias:
                orders = [(False, "preferred"), (True, "rejected")]
            else:
                orders = [(rng.choice([True, False]), None)]
            for flipped, shown_first in orders:
                hashes = {
                    principle: vote_hash(preferred, rejected, principle, shown_first)
                    for principle in part
                }
                requests.append(_VoteRequest(preferred, rejected, flipped, hashes))

        hashed_votes.update(
            await _run_pass(
                requests=requests,
                principles=part,
                model=model,
                config=config,
                vote_cache=vote_cache,
                usage_tracker=usage_tracker,
            )
        )

    # Collect per-comparison votes
    votes_per_comparison = {}
    votes_per_order = {} if config.position_debias else None
    for idx, preferred, rejected in rows:
        if config.position_debias:
            per_order = {
                f"{shown_first}_first": {
                    key: hashed_votes[
                        vote_hash(preferred, rejected, principle, shown_first)
                    ]
                    for key, principle in principles.items()
                }
                for shown_first in SHOWN_FIRST_VALUES
            }
            votes_per_order[idx] = per_order
            votes_per_comparison[idx] = {
                key: combine_position_votes(
                    per_order["preferred_first"][key],
                    per_order["rejected_first"][key],
                )
                for key in principles
            }
        else:
            votes_per_comparison[idx] = {
                key: hashed_votes[vote_hash(preferred, rejected, principle, None)]
                for key, principle in principles.items()
            }

    if config.position_debias:
        all_votes = [
            v for votes in votes_per_comparison.values() for v in votes.values()
        ]
        num_inconsistent = sum(v == POSITION_INCONSISTENT_VOTE for v in all_votes)
        logger.info(
            f"Position debiasing: {num_inconsistent}/{len(all_votes)} votes "
            f"({100 * num_inconsistent / max(len(all_votes), 1):.1f}%) were "
            "inconsistent across the two response orders and were discarded."
        )

    return VotingResult(votes=votes_per_comparison, votes_per_order=votes_per_order)


def combine_position_votes(vote_preferred_first, vote_rejected_first):
    """Combine votes obtained with both response orders (position debiasing).

    Votes are kept only if both orders agree. If either is invalid, the vote is
    invalid, otherwise disagreeing votes are marked as position-inconsistent.
    """
    if vote_preferred_first == vote_rejected_first:
        return vote_preferred_first
    if INVALID_VOTE in (vote_preferred_first, vote_rejected_first):
        return INVALID_VOTE
    return POSITION_INCONSISTENT_VOTE


async def _run_pass(
    requests: list[_VoteRequest],
    principles: list[str],
    model,
    config: VotingConfig,
    vote_cache: VoteCache | None,
    usage_tracker: UsageTracker,
) -> dict:
    """Run all vote requests (single pass over the data), returns hash -> vote."""
    cached_votes = vote_cache.get_cached_votes() if vote_cache is not None else {}
    semaphore = asyncio.Semaphore(config.max_concurrent_tasks)

    async def process(request: _VoteRequest) -> dict:
        if all(h in cached_votes for h in request.hashes.values()):
            return {h: cached_votes[h] for h in request.hashes.values()}
        async with semaphore:
            vote = await get_vote_for_single_comparison(
                preferred=request.preferred,
                rejected=request.rejected,
                principles=principles,
                flipped=request.flipped,
                model=model,
                model_name=config.model,
                usage_tracker=usage_tracker,
            )
        hashed_vote = {
            hash_str: vote.get(principle, INVALID_VOTE)
            for principle, hash_str in request.hashes.items()
        }
        if vote_cache is not None:
            for hash_str, vote_value in hashed_vote.items():
                vote_cache.update_cache(hash_str, vote_value)
        return hashed_vote

    results = await tqdm.gather(*[process(r) for r in requests])
    return {k: v for part in results for k, v in part.items()}


async def get_vote_for_single_comparison(
    preferred: str,
    rejected: str,
    principles: list[str],
    flipped: bool,
    model,
    model_name: str,
    usage_tracker: UsageTracker,
) -> dict:
    """Get votes (relative to the preferred response) for a single comparison.

    Returns:
        Mapping principle text -> vote.
    """
    if flipped:
        sample_a, sample_b = rejected, preferred
    else:
        sample_a, sample_b = preferred, rejected

    numbered_principles = {i: v for i, v in enumerate(principles)}

    messages = parse_prompt(
        prompt_str=VOTING_PROMPT,
        prompt_kwargs=dict(
            sample_a=sample_a,
            sample_b=sample_b,
            summaries=numbered_principles,
        ),
    )

    vote = None
    start = time.monotonic()
    try:
        response = await run_with_http_retries(model.ainvoke, messages)
        usage_tracker.record(
            model_name,
            response=response,
            latency_s=time.monotonic() - start,
            messages=messages,
            num_principles=len(principles),
            flipped=flipped,
        )
        if response.response_metadata.get("finish_reason") != "stop":
            logger.warning(
                f"Vote did not finish with 'stop' reason, instead with "
                f"'{response.response_metadata.get('finish_reason')}' reason. "
                "This will likely lead to invalid votes. For reasoning models, "
                "consider increasing max output tokens or lowering reasoning effort."
            )
        vote = response.content
        vote = parse_individual_pref_vote(
            vote,
            num_principles=len(numbered_principles),
            valid_values=VALID_VOTE_VALUES,
        )
        if all(v == INVALID_VOTE for v in vote.values()):
            usage_tracker.record_unparsable_response()
    except Exception as e:  # pylint: disable=broad-except
        if vote is None:
            logger.error(f"Failed to generate votes: {type(e).__name__}: {e}")
            usage_tracker.record(
                model_name,
                latency_s=time.monotonic() - start,
                error=e,
                messages=messages,
            )
        else:
            logger.error(f"Failed to parse votes: {vote} ({e})")
            usage_tracker.record_unparsable_response()
        vote = {i: INVALID_VOTE for i in range(len(numbered_principles))}

    vote = {
        numbered_principles[k]: v for k, v in vote.items() if k in numbered_principles
    }

    if flipped:
        vote = {k: (not v) if isinstance(v, bool) else v for k, v in vote.items()}

    return vote


def parse_individual_pref_vote(vote: str, num_principles: int, valid_values: dict):
    """Parse the raw LLM vote output into a dict of principle index -> vote."""
    try:
        vote_json = clean_vote_json(vote, num_principles)
        vote_dict = ast.literal_eval(vote_json)
    except Exception as e:  # pylint: disable=broad-except
        vote_dict = {i: INVALID_VOTE for i in range(num_principles)}
        logger.error(f"Failed to parse vote: {vote}")
        logger.error(e)

    vote_dict = {int(k): v for k, v in vote_dict.items()}

    if len(vote_dict) != num_principles:
        logger.error(
            f"Vote length {len(vote_dict)} does not match number of principles {num_principles}"
        )

    for key, value in vote_dict.items():
        if value in valid_values:
            vote_dict[key] = valid_values[value]
        else:
            logger.error(f"Vote value {value} is not in {list(valid_values.keys())}")
            vote_dict[key] = INVALID_VOTE

    return vote_dict


def clean_vote_json(vote_json: str, summaries_len: int) -> str:
    """Normalise the LLM's JSON-ish vote output into a Python literal."""
    vote_json = (
        vote_json.replace("\n", "")
        .replace(" ", "")
        .replace("true", "True")
        .replace("false", "False")
        .replace("both", "Both")
        .replace("neither", "Neither")
        .replace("null", "None")
    )
    # replace string keys with int keys
    for i in list(range(summaries_len + 10)) + ["True", "False", "None"]:
        vote_json = vote_json.replace(f'"{i}"', f"{i}")
        vote_json = vote_json.replace(f"'{i}'", f"{i}")

    for letter in ["A", "B", "Both", "Neither"]:
        vote_json = vote_json.replace(f"'{letter}'", f'"{letter}"')

    return vote_json


# --- Retries ------------------------------------------------------------------

RETRYABLE_HTTP_CODES = (403, 408, 409, 420, 429, 444, 498, 499)


def _get_http_code(err) -> int | None:
    if isinstance(err, openai.APIStatusError):
        return err.status_code
    if len(err.args) > 0 and isinstance(err.args[0], dict) and "code" in err.args[0]:
        return err.args[0]["code"]
    try:
        return int(re.sub(r".*Error code: (\d*).*", r"\1", str(err)))
    except ValueError:
        return None


def _fatal_model_error(err) -> bool:
    """Whether to give up retrying (True) for a given error.

    Retries network errors/timeouts, rate limits and server errors. Gives up on
    other client errors (e.g. invalid request or API key) and on non-API errors
    (e.g. bugs), which would fail again.
    """
    if isinstance(
        err,
        (openai.APIConnectionError, httpx.TransportError, asyncio.TimeoutError),
    ):
        logger.warning(f"Connection error, retrying: {err}")
        return False
    code = _get_http_code(err)
    if not isinstance(code, int):
        return True
    if code in RETRYABLE_HTTP_CODES or code // 100 == 5:
        logger.warning(f"Got HTTP error {code}, retrying: {err}")
        return False
    return True


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=8,
    max_value=60,
    giveup=_fatal_model_error,
)
async def run_with_http_retries(fn, *args, **kwargs):
    return await fn(*args, **kwargs)

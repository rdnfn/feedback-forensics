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
`position_debias=False`, the prompts are the same as ICAI's with FF's former
`ff-annotate` configuration (one seed, seed 1), except for the order of the two
responses: same system prompt, principles and principle order, and model
parameters. Differences to ICAI:

- A/B order: the random A/B orders (same random sequence as ICAI's) are drawn
  upfront and assigned to the comparisons in row order, making runs
  reproducible. ICAI assigned them in the order in which its async tasks
  happened to start (and backoff jitter of retries consumed random numbers
  too), so the A/B order of a given comparison (in a given pass) matches ICAI's
  only by chance, about half of the time. This only matters to the extent that
  the annotator is position-biased.
- HTTP: all API calls of a run share one event loop and an HTTP client bound to
  it, with request timeouts and TCP keepalive (see
  `feedback_forensics.models.main.create_async_http_client`).
- Retries: connection errors, timeouts and HTTP 409 (conflict, also retried by
  the OpenAI and Anthropic API clients) are retried like rate limits and server
  errors (ICAI only retried the latter), with the same retry budget as ICAI (see
  `run_with_http_retries`).
- OpenRouter requests also ask for the cost to be reported (`usage`), and
  identify FF instead of ICAI in their headers.
- Vote cache (resuming runs): requests without any valid vote are not cached,
  so that they are repeated when a run is resumed, and votes are only reused for
  the same generation settings (see `feedback_forensics.annotation.cache`).
- Batch API: optionally, requests are sent via OpenRouter's batch API (see
  `feedback_forensics.annotation.batch`).
"""

import ast
import asyncio
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
import backoff
import httpx
import openai
import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm

from feedback_forensics.annotation.cache import VoteCache, get_vote_hash
from feedback_forensics.annotation.prompts import VOTING_PROMPT, parse_prompt
from feedback_forensics.annotation.usage import UsageTracker
from feedback_forensics.models.openrouter import check_batch_model

INVALID_VOTE = "invalid"
POSITION_INCONSISTENT_VOTE = "position_inconsistent"
SHOWN_FIRST_VALUES = ("preferred", "rejected")
# Finish reasons of complete responses: "stop" (OpenAI-compatible APIs), "end_turn"
# and "stop_sequence" (Anthropic)
COMPLETE_FINISH_REASONS = ("stop", "end_turn", "stop_sequence")

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
    # Annotate via OpenRouter's batch API (at reduced price, but results may take
    # up to 24 h; `openrouter/` models only, needs a vote cache). See
    # `feedback_forensics.annotation.batch`.
    batch: bool = False
    # Minutes to wait for batches, `BatchPendingError` is raised if still pending
    batch_max_wait: float = 24 * 60

    def __post_init__(self):
        # Fail early: e.g. 0 concurrent tasks would make every API call wait forever
        for name in ("max_principles_per_prompt", "max_concurrent_tasks"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.batch:
            check_batch_model(self.model)
        if self.batch_max_wait < 0:
            raise ValueError(
                f"batch_max_wait must not be negative, got {self.batch_max_wait}"
            )


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

    @property
    def principles(self) -> list[str]:
        return list(self.hashes)

    @property
    def request_id(self) -> str:
        """Hash identifying the request (comparison, principles, response order,
        model and generation settings): its `custom_id` in a batch."""
        content = json.dumps([self.flipped, list(self.hashes.values())])
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @property
    def messages(self) -> list[dict]:
        """Messages of the LLM call (OpenAI format)."""
        return _get_messages(
            self.preferred, self.rejected, self.principles, self.flipped
        )


def _get_messages(
    preferred: str, rejected: str, principles: list[str], flipped: bool
) -> list[dict]:
    """Messages of the LLM call (OpenAI format) voting on a comparison."""
    sample_a, sample_b = (rejected, preferred) if flipped else (preferred, rejected)
    return parse_prompt(
        prompt_str=VOTING_PROMPT,
        prompt_kwargs=dict(
            sample_a=sample_a,
            sample_b=sample_b,
            summaries=dict(enumerate(principles)),
        ),
    )


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
            inside the event loop). Its own generation settings apply, not
            `config.max_output_tokens` and `config.reasoning_effort` (which,
            with `config.model`, still key its votes in the vote cache).
            Not supported with `config.batch`.

    Raises:
        BatchError: with `config.batch` (e.g. `BatchPendingError` if batches are
            still pending), or without if batches of an earlier run are pending,
            see `feedback_forensics.annotation.batch`.
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
    if config.batch and model is not None:
        raise ValueError("Batch annotation is not supported with a custom `model`.")
    if config.batch and cache_path is None:
        # Batch results are saved to the vote cache (possibly by a later run)
        raise ValueError("Batch annotation requires a vote cache (`cache_path`).")

    args = dict(
        feedback_df=feedback_df,
        principles=principles,
        config=config,
        cache_path=cache_path,
        usage_tracker=usage_tracker,
    )

    if model is not None or len(principles) == 0:
        return await _get_votes(**args, model=model)

    from feedback_forensics.models.main import create_async_http_client, get_model

    # Use an HTTP client bound to the current event loop instead of LangChain's
    # process-wide cached client: the cached client's connections break
    # ("Event loop is closed") when used from a later `asyncio.run()`.
    async with create_async_http_client(
        config.model, max_connections=max(config.max_concurrent_tasks, 10)
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
            max_output_tokens=config.max_output_tokens,
            reasoning_effort=config.reasoning_effort,
            shown_first=shown_first,
        )

    requests_per_pass = []
    for part in principle_parts:
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
        requests_per_pass.append(requests)

    # Ids of requests answered by the batch API without any valid vote
    answered_request_ids = set()
    if cache_path is not None:
        batch_state_file = cache_path.with_name(cache_path.name + "_batches.json")
    if config.batch:
        from feedback_forensics.annotation.batch import run_batches

        answered_request_ids = await run_batches(
            requests=[r for requests in requests_per_pass for r in requests],
            config=config,
            vote_cache=vote_cache,
            state_file=batch_state_file,
            usage_tracker=usage_tracker,
        )
    elif cache_path is not None and batch_state_file.exists():
        from feedback_forensics.annotation.batch import check_no_pending_batches

        check_no_pending_batches(batch_state_file)

    hashed_votes = {}
    for i, requests in enumerate(requests_per_pass):
        logger.info(f"Starting pass {i+1}/{len(principle_parts)}")
        hashed_votes.update(
            await _run_pass(
                requests=requests,
                model=model,
                config=config,
                vote_cache=vote_cache,
                usage_tracker=usage_tracker,
                answered_request_ids=answered_request_ids,
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
    model,
    config: VotingConfig,
    vote_cache: VoteCache | None,
    usage_tracker: UsageTracker,
    answered_request_ids: frozenset | set = frozenset(),
) -> dict:
    """Run all vote requests (single pass over the data), returns hash -> vote.

    Requests without any valid vote (failed API call, truncated or unparsable
    response) are not cached, so that they are repeated when the run is resumed.

    Args:
        answered_request_ids: Ids of requests already answered in this run (by
            the batch API) without any valid vote: their votes are invalid, they
            are not sent again.
    """
    cached_votes = vote_cache.get_cached_votes() if vote_cache is not None else {}
    semaphore = asyncio.Semaphore(config.max_concurrent_tasks)

    async def process(request: _VoteRequest) -> dict:
        if all(h in cached_votes for h in request.hashes.values()):
            return {h: cached_votes[h] for h in request.hashes.values()}
        if request.request_id in answered_request_ids:
            return {h: INVALID_VOTE for h in request.hashes.values()}
        async with semaphore:
            vote = await get_vote_for_single_comparison(
                preferred=request.preferred,
                rejected=request.rejected,
                principles=request.principles,
                flipped=request.flipped,
                model=model,
                model_name=config.model,
                usage_tracker=usage_tracker,
            )
        hashed_vote = {
            hash_str: vote.get(principle, INVALID_VOTE)
            for principle, hash_str in request.hashes.items()
        }
        if vote_cache is not None and any(
            v != INVALID_VOTE for v in hashed_vote.values()
        ):
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
    """Get votes (relative to the preferred response) for a single comparison,
    with a (live) API call.

    Returns:
        Mapping principle text -> vote.
    """
    messages = _get_messages(preferred, rejected, principles, flipped)

    # Logged before the call: calls in flight when the run is interrupted (e.g.
    # with Ctrl-C) remain without result, but may still be billed
    call_id = usage_tracker.record_start(model_name)
    start = time.monotonic()
    try:
        response = await run_with_http_retries(model.ainvoke, messages)
        usage_tracker.record(
            model_name,
            response=response,
            latency_s=time.monotonic() - start,
            messages=messages,
            call_id=call_id,
            num_principles=len(principles),
            flipped=flipped,
        )
        # OpenAI-compatible APIs report a finish reason, Anthropic a stop reason
        metadata = response.response_metadata
        warn_if_incomplete(metadata.get("finish_reason") or metadata.get("stop_reason"))
        content = response.content
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Failed to generate votes: {type(e).__name__}: {e}")
        usage_tracker.record(
            model_name,
            latency_s=time.monotonic() - start,
            error=e,
            messages=messages,
            call_id=call_id,
        )
        return {principle: INVALID_VOTE for principle in principles}

    return parse_vote_response(content, principles, flipped, usage_tracker)


def warn_if_incomplete(finish_reason) -> None:
    if finish_reason not in COMPLETE_FINISH_REASONS:
        logger.warning(
            f"Vote did not finish with 'stop' reason, instead with "
            f"'{finish_reason}' reason. "
            "This will likely lead to invalid votes. For reasoning models, "
            "consider increasing max output tokens or lowering reasoning effort."
        )


def parse_vote_response(
    content, principles: list[str], flipped: bool, usage_tracker: UsageTracker
) -> dict:
    """Parse the response to a vote request into votes relative to the preferred
    response (principle text -> vote).

    Args:
        content: Response text.
        principles: Principles of the request, in the order of the prompt.
        flipped: Whether the rejected response was shown as sample A.
        usage_tracker: Tracker to count unparsable responses.
    """
    try:
        vote = parse_individual_pref_vote(
            content,
            num_principles=len(principles),
            valid_values=VALID_VOTE_VALUES,
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Failed to parse votes: {content} ({e})")
        vote = {i: INVALID_VOTE for i in range(len(principles))}
    if all(v == INVALID_VOTE for v in vote.values()):
        usage_tracker.record_unparsable_response()

    numbered_principles = dict(enumerate(principles))
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

# As ICAI (403: OpenAI moderation), plus 409 (conflict, e.g. lock timeouts)
RETRYABLE_HTTP_CODES = (403, 408, 409, 420, 429, 444, 498, 499)
CONNECTION_ERRORS = (
    openai.APIConnectionError,  # incl. timeouts (openai.APITimeoutError)
    anthropic.APIConnectionError,  # incl. timeouts (anthropic.APITimeoutError)
    httpx.TransportError,
    asyncio.TimeoutError,
)


def _get_http_code(err) -> int | None:
    if isinstance(err, (openai.APIStatusError, anthropic.APIStatusError)):
        return err.status_code
    if len(err.args) > 0 and isinstance(err.args[0], dict) and "code" in err.args[0]:
        return err.args[0]["code"]
    try:
        return int(re.sub(r".*Error code: (\d*).*", r"\1", str(err)))
    except ValueError:
        return None


def _fatal_model_error(err) -> bool:
    """Whether to give up retrying (True) for a given error.

    Retries network errors/timeouts, rate limits, server errors and other
    transient HTTP errors (`RETRYABLE_HTTP_CODES`). Gives up on other client
    errors (e.g. invalid request or API key) and on non-API errors (e.g. bugs),
    which would fail again.
    """
    if isinstance(err, CONNECTION_ERRORS):
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
    max_tries=10,
    giveup=_fatal_model_error,
)
async def run_with_http_retries(fn, *args, **kwargs):
    """Call `fn`, retrying failed API calls (see `_fatal_model_error`).

    Same retry budget as ICAI (`inverse_cai.algorithm.utils`): up to 10 tries,
    waiting a random time of up to 1, 2, 4, ..., 256 s (exponential backoff with
    full jitter, about 4 min in total on average) between them. Each try includes
    the API client's own retries (2 by default for the OpenAI and Anthropic SDKs).
    """
    return await fn(*args, **kwargs)

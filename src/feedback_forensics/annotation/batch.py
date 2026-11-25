"""Annotation via OpenRouter's batch API (at reduced price, results within ~24 h).

With `VotingConfig.batch`, the vote requests without cached votes are submitted
as batches instead of being sent one by one (`openrouter/` models with a batch
variant only). Each request is the same as the live API call (messages, model,
generation settings). The results are saved to the vote cache, from which the
votes are then read as usual:

- Batches are tracked in a state file next to the vote cache
  (`<cache name>_batches.json`), written before submitting and updated with the
  batch id after: a batch is never submitted twice, and a batch whose submission
  response was lost is recovered from OpenRouter's batch list (once its results
  show that it is this batch, see `feedback_forensics.models.openrouter.recover_batch`).
  While the state file exists, annotating without `VotingConfig.batch` is refused
  (`BatchStateError`), as it would repeat the requests of the pending batches via
  the live API.
- If batches are still pending after `VotingConfig.batch_max_wait`,
  `BatchPendingError` is raised: re-running (same output directory and settings)
  collects them, without submitting their requests again.
- Requests are matched to results by a hash of their content (see
  `_VoteRequest.request_id`): results that match no request of the run (e.g. the
  data changed since submission) are never used as votes, but saved as raw
  results (`<cache name>_unmatched_<batch id>.json`), like results that cannot be
  read, so that no paid result is lost.
- Requests that failed within a completed batch (no response) are repeated via
  the live API in the same run. Responses without any valid vote (truncated or
  unparsable) give invalid votes, as with the live API: they are not cached, and
  thus repeated when the run is resumed.
- If a batch fails, expires or is cancelled, its partial results are saved and
  `BatchFailedError` is raised (instead of silently annotating the rest at the
  full price of the live API), also if other batches are still pending.
- A batch that OpenRouter does not know (long after its submission) is kept in
  the state file, and `BatchError` is raised: it may be running nonetheless
  (e.g. submitted with another API key), so it is only abandoned by the user
  (by deleting the state file).
- If the model has no batch variant (anymore), pending batches are still
  collected, before `BatchUnavailableError` is raised for the other requests.
- API usage and cost of each request are logged like live API calls (with its
  `batch_id`). Requests of pending batches count as unfinished calls, requests
  without a result when their batch ended as failed calls.
- Waiting for batches can be interrupted (e.g. with Ctrl-C) at any time.
"""

import asyncio
import json
import time
from pathlib import Path

from loguru import logger
from requests import RequestException

from feedback_forensics.models import openrouter
from feedback_forensics.annotation.cache import VoteCache
from feedback_forensics.annotation.usage import UsageTracker
from feedback_forensics.annotation.voting import (
    INVALID_VOTE,
    VotingConfig,
    _VoteRequest,
    parse_vote_response,
    warn_if_incomplete,
)
from feedback_forensics.models.main import DEFAULT_TEMPERATURE
from feedback_forensics.models.openrouter import (
    BATCH_SUFFIX,
    BatchNotFoundError,
    OpenRouterError,
)

# Requests per batch (more are split into several batches). Not a limit
# verified with OpenRouter: chosen to keep the size of submissions manageable.
MAX_REQUESTS_PER_BATCH = 10_000
POLL_INTERVAL = 60  # Seconds between status checks of pending batches


class BatchError(RuntimeError):
    """Batch annotation could not be completed."""


class BatchPendingError(BatchError):
    """Batches are still pending: re-run later to collect their results."""


class BatchFailedError(BatchError):
    """A batch did not complete (partial results are saved)."""


class BatchUnavailableError(BatchError):
    """The model has no batch variant on OpenRouter."""


class BatchStateError(BatchError):
    """Pending batches do not fit the requested annotation (other settings, or
    annotation without batch API)."""


class BatchRequestError(RuntimeError):
    """A request failed within a batch (for the API call log)."""


def check_no_pending_batches(state_file: Path) -> None:
    """Raise `BatchStateError` if batches are pending (annotation without batch API)."""
    if state_file.exists():
        raise BatchStateError(
            f"Annotation batches are pending for this output directory ({state_file})"
            ". Annotating without batch API would send their requests again, via "
            "the live API. Re-run with batch API to collect their results, or "
            "delete the file to abandon them (they may still be billed)."
        )


async def run_batches(
    requests: list[_VoteRequest],
    config: VotingConfig,
    vote_cache: VoteCache,
    state_file: Path,
    usage_tracker: UsageTracker,
) -> set[str]:
    """Get the votes of all requests without cached votes via the batch API, and
    save them to the vote cache.

    Submits the requests that are neither cached nor part of a pending batch,
    then waits for all pending batches (see module docstring).

    Returns:
        Ids of requests answered without any valid vote (not cached).

    Raises:
        BatchPendingError: Batches are still pending after `config.batch_max_wait`.
        BatchFailedError: A batch did not complete.
        BatchUnavailableError: The model has no batch variant (raised once no
            batch is pending anymore).
        BatchStateError: Pending batches were submitted with other settings.
        BatchError: A request to the batch API failed, or a batch is not known
            to OpenRouter.
    """
    run = _BatchRun(
        requests_by_id={r.request_id: r for r in requests},
        config=config,
        vote_cache=vote_cache,
        state_file=state_file,
        usage_tracker=usage_tracker,
    )
    deadline = time.time() + config.batch_max_wait * 60

    # Blocking steps run in threads, one at a time: when interrupted (e.g. with
    # Ctrl-C) while waiting, the run stops at once; a step in progress (e.g.
    # submitting a batch and saving its id) is still completed.
    not_submitted = False  # Requests are left, as there is no batch variant

    async def submit(requests):
        nonlocal not_submitted
        if not requests:
            return
        if not await asyncio.to_thread(run.batch_available):
            # Pending batches are still collected (as they would otherwise be
            # stuck: annotating without batch API is refused while they are)
            not_submitted = True
            return
        for i in range(0, len(requests), MAX_REQUESTS_PER_BATCH):
            chunk = requests[i : i + MAX_REQUESTS_PER_BATCH]
            await asyncio.to_thread(run.submit, chunk)

    def error(error_class, message: str) -> BatchError:
        """The error to raise: batches that did not complete are always reported
        by the run that found them, as later runs submit their requests again."""
        if not run.failures:
            return error_class(message)
        return BatchFailedError(
            f"Annotation batch(es) for {config.model} did not complete: "
            f"{', '.join(run.failures)}. Results received so far are saved. Re-run "
            "to submit the remaining requests as a new batch"
            + (f". Also: {message}" if message else ".")
        )

    try:
        await submit(await asyncio.to_thread(run.get_requests_to_submit))
        while run.state["batches"]:
            # (Batches found not to have been created are submitted again)
            await submit(await asyncio.to_thread(run.poll))
            if run.not_found:
                raise error(
                    BatchError,
                    f"Annotation batch(es) {run.not_found} for {config.model} are "
                    "not known to OpenRouter. They may have been submitted with "
                    "another API key (OPENROUTER_API_KEY), or the error may be "
                    "temporary: re-run to try again. To abandon the batches (they "
                    f"may still be billed), delete {state_file}.",
                )
            if not run.state["batches"]:
                break
            if time.time() + POLL_INTERVAL > deadline:
                batch_ids = [
                    entry["batch_id"] or "(submission interrupted, id not known yet)"
                    for entry in run.state["batches"]
                ]
                raise error(
                    BatchPendingError,
                    f"Annotation batch(es) {batch_ids} for {config.model} still "
                    "pending. Re-run later with the same output directory and "
                    "settings to collect the results (nothing is submitted again).",
                )
            await _wait(POLL_INTERVAL)
    except (OpenRouterError, RequestException) as e:
        raise error(
            BatchError,
            f"Request to OpenRouter's batch API failed: {e}. Re-run with the same "
            "output directory and settings to continue (submitted batches are "
            f"tracked in {state_file}).",
        ) from e

    unavailable = (
        f"OpenRouter lists no batch variant of {config.model} "
        f"('{config.model.split('/', 1)[1]}{BATCH_SUFFIX}'). Annotate without "
        "batch API instead."
    )
    if run.failures:
        raise error(BatchFailedError, unavailable if not_submitted else "")
    if not_submitted:
        raise BatchUnavailableError(unavailable)
    return run.answered_request_ids


async def _wait(seconds: float) -> None:
    await asyncio.sleep(seconds)


class _BatchRun:
    """State and (blocking) steps of `run_batches`."""

    def __init__(
        self,
        requests_by_id: dict[str, _VoteRequest],
        config: VotingConfig,
        vote_cache: VoteCache,
        state_file: Path,
        usage_tracker: UsageTracker,
    ):
        self.requests_by_id = requests_by_id
        self.config = config
        self.vote_cache = vote_cache
        self.state_file = state_file
        self.usage_tracker = usage_tracker
        self.state = self._load_state()
        # Ids of requests answered without any valid vote
        self.answered_request_ids: set[str] = set()
        # Descriptions of batches that did not complete
        self.failures: list[str] = []
        # Ids of batches not known to OpenRouter in the last poll (kept in state)
        self.not_found: list[str] = []
        self._batch_available: bool | None = None

    def batch_available(self) -> bool:
        """Whether OpenRouter lists a batch variant of the model."""
        if self._batch_available is None:
            self._batch_available = bool(
                openrouter.get_models_with_batch_variant([self.config.model])
            )
            if not self._batch_available and self.state["batches"]:
                logger.warning(
                    f"OpenRouter lists no batch variant of {self.config.model} "
                    "(anymore): only collecting the pending batches."
                )
        return self._batch_available

    def _load_state(self) -> dict:
        settings = {
            "model": self.config.model,
            "max_output_tokens": self.config.max_output_tokens,
            "reasoning_effort": self.config.reasoning_effort,
            # Request ids depend on the annotation prompt
            "without_separate_prompt": self.config.without_separate_prompt,
        }
        if not self.state_file.exists():
            return {**settings, "batches": []}
        with open(self.state_file, encoding="utf-8") as f:
            state = json.load(f)
        # Batches submitted before prompts were stated separately
        state.setdefault("without_separate_prompt", True)
        differences = [
            f"{key}={state.get(key)!r} (requested: {value!r})"
            for key, value in settings.items()
            if state.get(key) != value
        ]
        if differences:
            raise BatchStateError(
                f"Pending annotation batches ({self.state_file}) were submitted with "
                f"other settings: {', '.join(differences)}. Re-run with those "
                "settings to collect their results, or delete the file to abandon "
                "them (they may still be billed)."
            )
        return state

    def _save_state(self) -> None:
        if self.state["batches"]:
            openrouter.write_json_atomic(self.state_file, self.state)
        else:
            self.state_file.unlink(missing_ok=True)

    def _remove(self, entry: dict) -> None:
        self.state["batches"].remove(entry)
        self._save_state()

    def _uncached(self, request_ids) -> list[_VoteRequest]:
        """Requests of this run (of the given ids) without cached votes."""
        cached_votes = self.vote_cache.get_cached_votes()
        requests = [self.requests_by_id.get(request_id) for request_id in request_ids]
        return [
            request
            for request in requests
            if request is not None
            and not all(h in cached_votes for h in request.hashes.values())
        ]

    def get_requests_to_submit(self) -> list[_VoteRequest]:
        """Requests that are neither cached nor part of a pending batch."""
        self._recover_lost_submissions()
        pending_ids = {i for e in self.state["batches"] for i in e["custom_ids"]}
        to_submit = self._uncached(
            i for i in self.requests_by_id if i not in pending_ids
        )
        if not to_submit and not self.state["batches"]:
            logger.info("Batch annotation: all votes are cached, nothing to submit.")
        return to_submit

    def _recover_lost_submissions(self) -> list[_VoteRequest]:
        """Find the ids of batches whose submission was interrupted before the id
        was saved (kept without id while not known yet whether they were created).

        Returns:
            Requests (without cached votes) of batches that were not created.
        """
        not_submitted = []
        for entry in list(self.state["batches"]):
            if entry["batch_id"]:
                continue
            resolved, batch_id = openrouter.recover_batch(
                self.config.model,
                entry,
                custom_ids=entry["custom_ids"],
                exclude_ids=[e["batch_id"] for e in self.state["batches"]],
            )
            if batch_id:
                entry["batch_id"] = batch_id
                self._save_state()
                logger.info(f"Recovered batch {batch_id} (submission was interrupted).")
            elif resolved:
                self._remove(entry)
                logger.info(
                    "A batch submission was interrupted before the batch was "
                    "created: submitting its requests again."
                )
                not_submitted += self._uncached(entry["custom_ids"])
            else:
                self._save_state()
                logger.info(
                    "A batch submission was interrupted: not known yet whether the "
                    "batch was created (batches that may be it have no results yet)."
                )
        return not_submitted

    def submit(self, requests: list[_VoteRequest]) -> None:
        """Submit requests as one batch."""
        config = self.config
        entry = {"custom_ids": [request.request_id for request in requests]}
        self.state["batches"].append(entry)
        batch = openrouter.submit_batch(
            config.model,
            [
                {"custom_id": r.request_id, "body": _request_body(r, config)}
                for r in requests
            ],
            entry=entry,
            save=self._save_state,
            discard=lambda: self._remove(entry),
        )
        logger.info(
            f"Submitted batch {batch['id']} with {len(requests)} annotation requests "
            f"for {config.model}."
        )
        for request in requests:
            self.usage_tracker.record_start(
                config.model,
                call_id=_call_id(batch["id"], request.request_id),
                batch_id=batch["id"],
            )

    def poll(self) -> list[_VoteRequest]:
        """Check all pending batches once, and save the results of those that ended.

        Returns:
            Requests to submit again (see `_recover_lost_submissions`).
        """
        not_submitted = self._recover_lost_submissions()
        self.not_found = []
        for entry in list(self.state["batches"]):
            if not entry["batch_id"]:
                continue
            try:
                batch, status = openrouter.poll_batch(
                    entry["batch_id"], entry["submitted_at"]
                )
            except BatchNotFoundError:
                # Kept in state (the other batches are still collected)
                self.not_found.append(entry["batch_id"])
                continue
            if batch is None:
                continue
            self._save_results(batch, status, entry)
            # Only remove batch from state once its results are safely saved
            self._remove(entry)
            if status != "completed":
                self.failures.append(f"{batch['id']} ({status}: {batch.get('error')})")
        return not_submitted

    def _save_results(self, batch: dict, status: str, entry: dict) -> None:
        """Save the results of a batch (also partial results of batches that did
        not complete) to the vote cache, and record the result of each request of
        the batch in the API call log."""
        config, usage_tracker = self.config, self.usage_tracker
        batch_id = batch["id"]
        cached_votes = self.vote_cache.get_cached_votes()
        # Results already recorded by a run interrupted while saving the results
        recorded = usage_tracker.get_recorded_call_ids(batch_id)

        def record(custom_id, **kwargs):
            call_id = _call_id(batch_id, custom_id)
            if call_id not in recorded:
                usage_tracker.record(
                    config.model, call_id=call_id, batch_id=batch_id, **kwargs
                )
                recorded.add(call_id)

        unmatched = []
        num_saved = num_invalid = num_failed = 0
        for result in batch.get("results") or []:
            custom_id = result.get("custom_id") if isinstance(result, dict) else None
            if not isinstance(custom_id, str):
                # Cannot be matched to a request (recorded as without result)
                logger.warning(f"Result without id in batch {batch_id}.")
                unmatched.append(result)
                continue
            request = self.requests_by_id.get(custom_id)
            try:
                body = openrouter.get_result_body(result)
                if body is not None:
                    choice = body["choices"][0]
                    content = choice["message"]["content"]
                    finish_reason = choice.get("finish_reason")
                    if not isinstance(content, str):
                        raise TypeError(f"content is not a string: {content!r}")
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                logger.warning(f"Could not read result in batch {batch_id}: {e!r}")
                unmatched.append(result)
                record(custom_id, error=BatchRequestError(f"unreadable result: {e!r}"))
                continue
            if body is None:
                error = result.get("error") or result.get("response")
                record(custom_id, error=BatchRequestError(json.dumps(error)))
                num_failed += 1
                continue

            request_info = {}
            if request is not None:
                request_info = dict(
                    messages=request.messages if usage_tracker.log_content else None,
                    num_principles=len(request.hashes),
                    flipped=request.flipped,
                )
            record(custom_id, response=body, **request_info)
            if request is None:
                unmatched.append(result)
                continue
            warn_if_incomplete(finish_reason)
            vote = parse_vote_response(
                content, request.principles, request.flipped, usage_tracker
            )
            hashed_vote = {
                hash_str: vote.get(principle, INVALID_VOTE)
                for principle, hash_str in request.hashes.items()
            }
            if any(v != INVALID_VOTE for v in hashed_vote.values()):
                for hash_str, vote_value in hashed_vote.items():
                    self.vote_cache.update_cache(hash_str, vote_value)
                cached_votes.update(hashed_vote)
                num_saved += 1
            elif not all(h in cached_votes for h in request.hashes.values()):
                # (Else its votes are cached, from another request for the same
                # comparison and principles)
                self.answered_request_ids.add(request.request_id)
                num_invalid += 1

        # Requests without result, e.g. of batches that did not complete
        num_missing = 0
        for custom_id in entry["custom_ids"]:
            if _call_id(batch_id, custom_id) not in recorded:
                record(
                    custom_id,
                    error=BatchRequestError(f"no result (batch {status})"),
                )
                num_missing += 1

        if unmatched:
            unmatched_file = self.state_file.with_name(
                self.state_file.stem.removesuffix("_batches")
                + f"_unmatched_{batch_id}.json"
            )
            openrouter.write_json_atomic(
                unmatched_file, {"model": config.model, "results": unmatched}
            )
            logger.warning(
                f"{len(unmatched)} results of batch {batch_id} could not be read or "
                "match no annotation request of this run (did the data, principles "
                f"or settings change?). Saved as raw results to {unmatched_file}."
            )

        completed = status == "completed"
        log = logger.info if completed else logger.error
        log(
            f"Batch {batch_id} for {config.model} {status}: saved votes of "
            f"{num_saved} requests, {num_invalid} responses without valid vote, "
            f"{num_failed} failed requests, {num_missing} requests without result"
            + (
                " (repeated via live API)"
                if completed and (num_failed or num_missing)
                else ""
            )
            + f". Cost: {(batch.get('usage') or {}).get('cost')}"
        )


def _request_body(request: _VoteRequest, config: VotingConfig) -> dict:
    """Same parameters as `feedback_forensics.models.main.get_model` for live calls."""
    body = {
        "messages": request.messages,
        "max_tokens": config.max_output_tokens,
        "temperature": DEFAULT_TEMPERATURE,
        # Report the cost of each request
        "usage": {"include": True},
    }
    if config.reasoning_effort is not None:
        body["reasoning"] = {"effort": config.reasoning_effort}
    return body


def _call_id(batch_id: str, request_id: str) -> str:
    """Id of a batch request in the API call log: known to the run collecting it."""
    return f"{batch_id}:{request_id}"

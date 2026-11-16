"""Access to the OpenRouter API beyond chat completions (models, batches).

Shared by batch generation (`feedback_forensics.tools.ff_modelgen`) and batch
annotation (`feedback_forensics.annotation.batch`).
"""

import json
import os
import pathlib
import time
from datetime import datetime, timezone

import requests
from loguru import logger

from feedback_forensics.models.main import OPENROUTER_API_BASE, OPENROUTER_HEADERS

BATCH_SUFFIX = ":batch"  # OpenRouter suffix of batch model variants
BATCH_NOT_FOUND_GRACE = 600  # Seconds after submission to treat 404 as pending
# Statuses of batches that ended without completing (may have partial results):
# as reported by OpenRouter, and "gone" (results deleted), see `poll_batch`
INCOMPLETE_STATUSES = ("failed", "expired", "cancelled", "gone")


class OpenRouterError(RuntimeError):
    """Failed OpenRouter request, with a message legible without traceback."""

    def __init__(self, status_code: int, text: str, fix: str | None = None):
        message = f"OpenRouter API error {status_code}: {text}"
        if fix:
            message += f"\nTo fix: {fix}"
        super().__init__(message)
        self.status_code = status_code


class OpenRouterAccountError(OpenRouterError):
    """Request rejected due to the API key or account (not the request itself).

    Retrying cannot succeed until the user fixes the key or account.
    """

    def __init__(self, status_code: int, problem: str, fix: str):
        RuntimeError.__init__(self, f"{problem}\nTo fix: {fix}")
        self.status_code = status_code


def get_openrouter_error(status_code: int, text: str) -> OpenRouterError:
    """Create legible error (problem and, if known, fix) for failed OpenRouter response.

    Args:
        status_code: HTTP status code of the response.
        text: Body of the response (OpenRouter's JSON error, or anything else).
    """
    try:
        error = json.loads(text)["error"]
        message, metadata = error.get("message"), error.get("metadata") or {}
    except (ValueError, KeyError, TypeError, AttributeError):
        message, metadata = None, {}
    message = message if isinstance(message, str) else ""
    metadata = metadata if isinstance(metadata, dict) else {}
    rerun = "Then re-run: completed work and submitted batches are resumed."

    key_limit = (
        metadata.get("limit_source") == "openrouter_key_limit"
        or "key limit exceeded" in message.lower()
    )
    if status_code == 403 and key_limit:
        limit, spend = metadata.get("limit_usd"), metadata.get("spend_usd")
        if isinstance(limit, (int, float)) and isinstance(spend, (int, float)):
            details = f"${spend:,.2f} used of ${limit:,.2f} limit"
        else:
            details = message
        return OpenRouterAccountError(
            status_code,
            f"OpenRouter API key has reached its spending limit ({details}).",
            "raise or remove the limit of the key at "
            "https://openrouter.ai/settings/keys, or set OPENROUTER_API_KEY to a "
            f"key with enough remaining limit. {rerun}",
        )
    if status_code == 402:
        return OpenRouterAccountError(
            status_code,
            f"OpenRouter account has insufficient credits ({message or text}).",
            f"add credits at https://openrouter.ai/settings/credits. {rerun}",
        )
    if status_code == 401:
        return OpenRouterAccountError(
            status_code,
            f"OpenRouter rejected the API key ({message or text}).",
            "set OPENROUTER_API_KEY to a valid key "
            f"(see https://openrouter.ai/settings/keys). {rerun}",
        )
    # Other errors: OpenRouter's message instead of raw JSON (or e.g. an HTML page)
    text = " ".join(text.split())
    details = message or (text[:300] + ("..." if len(text) > 300 else ""))
    if status_code == 429:
        fix = "rate limit still hit after retries, wait a few minutes. " + rerun
    elif status_code >= 500:
        fix = (
            "likely a temporary problem at OpenRouter (see "
            f"https://status.openrouter.ai), wait a few minutes. {rerun}"
        )
    elif status_code == 404:
        fix = (
            "check that the model name is correct and still available "
            "(see https://openrouter.ai/models)."
        )
    else:
        fix = None
    return OpenRouterError(status_code, details or "(empty response)", fix)


class BatchNotFoundError(OpenRouterError):
    """A batch is not known to OpenRouter, long after its submission.

    Not treated as the end of the batch: it may be running (and billed)
    nonetheless, e.g. if it was submitted with another API key, or if the error
    is temporary. Its state is to be kept, until it is abandoned by the user.
    """

    def __init__(self, batch_id: str, submitted_at: str):
        super().__init__(
            404,
            f"batch {batch_id} (submitted {submitted_at}) not found. It may have "
            "been submitted with another API key (OPENROUTER_API_KEY), or the "
            "error may be temporary",
        )
        self.batch_id = batch_id


def openrouter_request(method: str, path: str, **kwargs) -> dict:
    """Send request to OpenRouter API, retrying on rate limits and server errors.

    Non-GET requests are only retried on rate limits (429), since other errors
    (5xx, timeouts) are ambiguous: the request may have been processed, and
    retrying could e.g. submit (and pay for) a batch twice.
    """
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        **OPENROUTER_HEADERS,
    }
    retry_all = method == "GET"
    for attempt in range(5):
        last_attempt = attempt == 4
        try:
            response = requests.request(
                method,
                OPENROUTER_API_BASE + path,
                headers=headers,
                timeout=300,
                **kwargs,
            )
        except requests.RequestException as e:
            if not retry_all or last_attempt:
                raise
            logger.warning(f"OpenRouter request failed ({e}), retrying...")
            time.sleep(60)
            continue
        retryable = response.status_code == 429 or (
            retry_all and response.status_code >= 500
        )
        if not retryable or last_attempt:
            break
        logger.warning(f"OpenRouter returned {response.status_code}, retrying...")
        time.sleep(60)
    if not response.ok:
        raise get_openrouter_error(response.status_code, response.text)
    return response.json()


def check_batch_model(model_name: str) -> None:
    """Raise a ValueError if the batch API cannot be used for a model."""
    if not model_name.startswith("openrouter/"):
        raise ValueError(
            "The batch API is only supported for 'openrouter/' models, "
            f"got '{model_name}'."
        )


def get_models_with_batch_variant(model_names) -> set[str]:
    """The models ('openrouter/provider/model') for which OpenRouter lists a batch
    variant. Never includes models of other providers."""
    candidates = [m for m in model_names if m.startswith("openrouter/")]
    if not candidates:
        return set()
    available = {m["id"] for m in openrouter_request("GET", "/models")["data"]}
    return {m for m in candidates if m.split("/", 1)[1] + BATCH_SUFFIX in available}


def submit_batch(
    model_name: str, requests_list: list[dict], entry: dict, save, discard
):
    """Submit chat completion requests as a batch, tracked in `entry`.

    The entry is saved before submitting (without batch id) and again with the
    batch id after, so that a batch is never lost or submitted twice: if the
    response gets lost (crash, timeout, 5xx), `recover_batch` finds the batch
    instead of submitting again.

    Args:
        model_name: Model ('openrouter/provider/model').
        requests_list: Requests ({"custom_id": ..., "body": ...}).
        entry: State of the batch, set here: `batch_id`, `num_requests`,
            `submitted_at`.
        save: Function (without arguments) persisting the entry.
        discard: Function (without arguments) removing the persisted entry,
            called if the batch was definitely not created.

    Returns:
        The created batch.
    """
    check_batch_model(model_name)
    entry.update(
        batch_id=None,
        num_requests=len(requests_list),
        submitted_at=now().isoformat(),
    )
    save()
    # Note: OpenRouter requires 'requests' to be the last key
    try:
        batch = openrouter_request(
            "POST",
            "/batches",
            json={
                "endpoint": "/v1/chat/completions",
                "model": model_name.split("/", 1)[1],
                "requests": requests_list,
            },
        )
    except OpenRouterError as e:
        if e.status_code < 500:
            # Definitely rejected (not created), safe to submit again later
            discard()
        raise
    entry["batch_id"] = batch["id"]
    save()
    return batch


def recover_batch(
    model_name: str, entry: dict, custom_ids, exclude_ids=()
) -> tuple[bool, str | None]:
    """Find the batch of an entry whose submission response was lost (entry
    without batch id, see `submit_batch`).

    Candidates are the batches of the model with the same number of requests
    created since the submission. As this may also match other batches (e.g. of
    another dataset submitted at the same time), a candidate is only accepted
    once its results show the custom ids of the entry's requests. Candidates
    found to be other batches are remembered in the entry (`not_ours`).

    Args:
        model_name: Model ('openrouter/provider/model').
        entry: State of the batch (`num_requests`, `submitted_at`).
        custom_ids: Custom ids of the requests of the batch.
        exclude_ids: Ids of batches known to be other batches.

    Returns:
        (resolved, batch id): (True, id) if the batch was found, (True, None) if
        no such batch exists (it can be submitted again), (False, None) if
        candidates cannot be verified yet (without results): try again later.
    """
    created_after = int(parse_time(entry["submitted_at"]).timestamp()) - 300
    model = model_name.split("/", 1)[1]
    not_ours = entry.setdefault("not_ours", [])
    candidates, after = [], None
    while True:
        params = {"limit": 100, "created_after": created_after}
        if after:
            params["after"] = after
        page = openrouter_request("GET", "/batches", params=params)
        for batch in page.get("data") or []:
            total = (batch.get("request_counts") or {}).get("total")
            if (
                batch.get("model") == model
                and total in (None, entry["num_requests"])
                and batch["id"] not in exclude_ids
                and batch["id"] not in not_ours
            ):
                candidates.append(batch["id"])
        if not page.get("has_more") or not page.get("last_id"):
            break
        after = page["last_id"]

    custom_ids = set(custom_ids)
    resolved = True
    for batch_id in candidates:
        batch, status = poll_batch(batch_id, entry["submitted_at"])
        if batch is None:
            resolved = False  # without results yet
        elif any(r.get("custom_id") in custom_ids for r in batch.get("results") or []):
            return True, batch_id
        else:
            # Results of other requests, or ended without any results (if ours,
            # nothing is lost by submitting again)
            not_ours.append(batch_id)
    return resolved, None


def poll_batch(batch_id: str, submitted_at: str) -> tuple[dict | None, str]:
    """Get a batch, if it is no longer pending.

    Returns:
        (batch, status), with batch None while pending. Statuses of batches that
        ended: "completed", or one of `INCOMPLETE_STATUSES`. Batches without
        "results" have none (anymore).

    Raises:
        BatchNotFoundError: The batch is not known to OpenRouter.
    """
    try:
        batch = openrouter_request("GET", f"/batches/{batch_id}")
    except OpenRouterError as e:
        if e.status_code == 404:
            age = (now() - parse_time(submitted_at)).total_seconds()
            if age <= BATCH_NOT_FOUND_GRACE:
                # Newly submitted batches may briefly not be found
                return None, "not_found_yet"
            raise BatchNotFoundError(batch_id, submitted_at) from e
        if e.status_code == 410:
            # Results deleted by OpenRouter (after 30 days)
            return {"id": batch_id, "error": "results expired on OpenRouter"}, "gone"
        raise
    status = batch["status"]
    if status in INCOMPLETE_STATUSES:
        return batch, status
    if status == "completed" and batch.get("results") is not None:
        return batch, status
    return None, status


def get_result_body(result: dict) -> dict | None:
    """Chat completion of a result of a batch, None if the request failed.

    Raises:
        KeyError, TypeError: The result cannot be read.
    """
    if not isinstance(result, dict):
        raise TypeError(f"result is not an object: {result!r}")
    response = result.get("response")
    if result.get("error") or not response:
        return None
    if not isinstance(response, dict):
        raise TypeError(f"response is not an object: {response!r}")
    if response.get("status_code") != 200:
        return None
    return response["body"]


def write_json_atomic(path: pathlib.Path, data: dict):
    """Write JSON so that path never contains a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str) -> datetime:
    """Parse ISO time (naive values are interpreted as local time)."""
    return datetime.fromisoformat(value).astimezone(timezone.utc)

"""Logging of LLM API calls: token usage, cost and failures.

Every API call made during annotation is logged as JSON lines in `api_calls.jsonl`:
a short entry when the call starts (time, model, `call_id` and `"started": true`),
and an entry with its result (same `call_id`; token usage, cost, finish reason,
duration; with `log_content=True` also the request messages, response text and
full response metadata). At the end of a run, totals are summarised (see
`summarise_usage`) for all calls in the log, i.e. for all runs with the same
output directory (such as an interrupted run and its resumption), and for the
calls of the run itself. Calls without a result (`num_unfinished_calls`) were in
flight when a run was interrupted (e.g. with Ctrl-C or killed): the provider may
still have billed them.

Cost:
- `cost_usd`: cost as reported by the provider. OpenRouter reports it (requested
  via `usage: {include: true}`, see `feedback_forensics.models`), other providers
  do not (null). Note that providers sometimes report a cost of 0 (e.g. free
  promotions), hence also:
- `list_price_cost_usd` (summary only): tokens x OpenRouter's public list price
  for the model (`openrouter/` models only, best effort).
- `cost_complete` (summary only): false if `cost_usd` is only a lower bound,
  because the cost was not reported for all successful calls or some calls are
  unfinished.
"""

import datetime
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _extract_usage(response: Any) -> dict:
    """Extract token usage and cost from a LangChain chat model response."""
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    # OpenAI-compatible APIs (incl. OpenRouter) use "token_usage",
    # Anthropic uses "usage".
    raw_usage = (
        response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    )
    output_details = usage_metadata.get("output_token_details") or {}

    cost = raw_usage.get("cost") if isinstance(raw_usage, dict) else None
    return {
        "input_tokens": usage_metadata.get("input_tokens"),
        "output_tokens": usage_metadata.get("output_tokens"),
        "reasoning_tokens": output_details.get("reasoning"),
        "cost_usd": float(cost) if cost is not None else None,
        "finish_reason": response_metadata.get("finish_reason")
        or response_metadata.get("stop_reason"),
        "provider_model": response_metadata.get("model_name")
        or response_metadata.get("model"),
    }


class UsageTracker:
    """Collects (and optionally logs to a JSONL file) usage of LLM API calls."""

    def __init__(self, log_path: str | Path | None = None, log_content: bool = False):
        """
        Args:
            log_path: JSONL file to append the start and result of each API call
                to (None: no log).
            log_content: Also log request messages, response text and full
                response metadata (larger log files).
        """
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            _end_incomplete_last_line(self.log_path)
        self.log_content = log_content
        self._lock = threading.Lock()
        self.num_calls = 0
        self.num_failed_calls = 0
        self.num_truncated_calls = 0
        # None if unknown (not logged in api_calls.jsonl, see `from_log`)
        self.num_unparsable_responses: int | None = 0
        self.num_calls_with_cost = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0
        # model -> [input tokens, output tokens] (for list prices)
        self.tokens_per_model: dict[str, list[int]] = {}
        # Ids of calls started but without result (see `record_start`)
        self.unfinished_calls: set[str] = set()

    @classmethod
    def from_log(cls, log_path: str | Path) -> "UsageTracker":
        """Tracker with the totals of all calls logged in `log_path` (if it exists).

        The number of unparsable responses is not logged and hence unknown.
        Calls whose start, but no result, was logged are unfinished.
        """
        tracker = cls()
        tracker.num_unparsable_responses = None
        log_path = Path(log_path)
        if not log_path.exists():
            return tracker
        # An incomplete line (see `_end_incomplete_last_line`) may end within a
        # multi-byte character
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Skipping malformed line in {log_path}: {line.strip()[:200]}"
                    )
                    continue
                tracker._add(entry)
        return tracker

    def record_start(self, model: str) -> str:
        """Record the start of an API call, returns its id (pass it to `record`).

        Calls without a result were in flight when the run was interrupted: they
        are counted as unfinished, as the provider may have billed them.
        """
        call_id = uuid.uuid4().hex
        self._log({"time": _now(), "model": model, "call_id": call_id, "started": True})
        return call_id

    def record(
        self,
        model: str,
        response: Any = None,
        latency_s: float | None = None,
        error: BaseException | None = None,
        messages: list | None = None,
        call_id: str | None = None,
        **extra,
    ) -> dict:
        """Record the result of a single API call (successful if `response` given,
        else failed). `call_id`: from `record_start`, if the start was recorded."""
        entry = {
            "time": _now(),
            "model": model,
            "success": error is None and response is not None,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
        }
        if call_id is not None:
            entry["call_id"] = call_id
        if response is not None:
            entry.update(_extract_usage(response))
        if error is not None:
            entry["error"] = f"{type(error).__name__}: {error}"[:2000]
        entry.update(extra)
        if self.log_content:
            entry["request_messages"] = messages
            if response is not None:
                entry["response_text"] = getattr(response, "content", None)
                entry["response_metadata"] = getattr(
                    response, "response_metadata", None
                )
        self._log(entry)
        return entry

    def _log(self, entry: dict) -> None:
        """Add an entry to the totals and append it to the log file."""
        with self._lock:
            self._add(entry)
            if self.log_path is not None:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")

    def _add(self, entry: dict) -> None:
        """Add a log entry (start or result of a call) to the totals."""
        if entry.get("started"):
            self.unfinished_calls.add(entry.get("call_id"))
            return
        self.unfinished_calls.discard(entry.get("call_id"))
        self.num_calls += 1
        if not entry.get("success"):
            self.num_failed_calls += 1
        if entry.get("finish_reason") in ("length", "max_tokens"):
            self.num_truncated_calls += 1
        input_tokens = entry.get("input_tokens") or 0
        output_tokens = entry.get("output_tokens") or 0
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.reasoning_tokens += entry.get("reasoning_tokens") or 0
        model_tokens = self.tokens_per_model.setdefault(entry.get("model"), [0, 0])
        model_tokens[0] += input_tokens
        model_tokens[1] += output_tokens
        if entry.get("cost_usd") is not None:
            self.num_calls_with_cost += 1
            self.cost_usd += entry["cost_usd"]

    def record_unparsable_response(self) -> None:
        with self._lock:
            self.num_unparsable_responses += 1

    def summary(self, list_prices: dict[str, dict] | None = None) -> dict:
        """Totals across all recorded calls.

        Args:
            list_prices: Model -> list price per token (see `get_list_prices`).
                The cost at list price is computed if known for all models.
        """
        num_successful = self.num_calls - self.num_failed_calls
        summary = {
            "num_calls": self.num_calls,
            "num_failed_calls": self.num_failed_calls,
            "num_truncated_calls": self.num_truncated_calls,
            # Started, but without result (interrupted runs), not in num_calls
            "num_unfinished_calls": len(self.unfinished_calls),
        }
        if self.num_unparsable_responses is not None:
            summary["num_unparsable_responses"] = self.num_unparsable_responses
        summary.update(
            {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                # None if the provider did not report cost for any call
                "cost_usd": (self.cost_usd if self.num_calls_with_cost > 0 else None),
                "num_calls_with_cost": self.num_calls_with_cost,
                "cost_complete": (
                    self.num_calls_with_cost > 0
                    and self.num_calls_with_cost == num_successful
                    and not self.unfinished_calls
                ),
                "list_price_cost_usd": None,
            }
        )
        list_prices = list_prices or {}
        if self.tokens_per_model and all(
            model in list_prices for model in self.tokens_per_model
        ):
            summary["list_price_cost_usd"] = sum(
                input_tokens * list_prices[model]["prompt"]
                + output_tokens * list_prices[model]["completion"]
                for model, (input_tokens, output_tokens) in (
                    self.tokens_per_model.items()
                )
            )
            summary["list_price_per_token"] = {
                model: list_prices[model] for model in self.tokens_per_model
            }
        return summary


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _end_incomplete_last_line(path: Path) -> None:
    """End the last line of a file with a newline, if it has none: e.g. if a run was
    killed while writing to the log, so that the next entry gets its own line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
    with open(path, "ab") as f:
        f.write(b"\n")


def summarise_usage(
    tracker: UsageTracker, num_votes: int, num_invalid_votes: int
) -> dict:
    """Summary of the API usage of all runs with the tracker's log file.

    Top-level totals cover all calls in the tracker's log (`api_calls.jsonl`),
    i.e. of all runs with the same output directory (e.g. an interrupted run and
    its resumption). `this_run` covers the calls recorded by `tracker` only.
    `num_votes` and `num_invalid_votes` describe the votes in the output.

    The cost at OpenRouter list price is computed for `openrouter/` models
    (needs network access).
    """
    total = UsageTracker.from_log(tracker.log_path) if tracker.log_path else tracker
    list_prices = get_list_prices(total.tokens_per_model)
    summary = total.summary(list_prices)
    summary["num_votes"] = num_votes
    summary["num_invalid_votes"] = num_invalid_votes
    summary["this_run"] = tracker.summary(list_prices)
    return summary


def combine_usage_summaries(summaries: list[dict | None]) -> dict:
    """Totals of the usage summaries (see `summarise_usage`) of several annotation
    runs, e.g. of the annotation files merged into one dataset.

    A summary may be None if unknown (e.g. annotated before usage was recorded):
    the totals then only cover the other runs, `cost_complete` is false and the
    cost at list price is null.
    """
    known = [s for s in summaries if s is not None]
    all_known = bool(known) and len(known) == len(summaries)

    def total(field):
        return sum(s.get(field) or 0 for s in known)

    reported_costs = [s["cost_usd"] for s in known if s.get("cost_usd") is not None]
    list_price_costs = [s.get("list_price_cost_usd") for s in known]
    list_prices = {}
    for s in known:
        list_prices.update(s.get("list_price_per_token") or {})
    combined = {
        "num_annotation_runs": len(summaries),
        "num_annotation_runs_without_usage": len(summaries) - len(known),
        **{
            field: total(field)
            for field in (
                "num_calls",
                "num_failed_calls",
                "num_truncated_calls",
                "num_unfinished_calls",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
            )
        },
        "cost_usd": sum(reported_costs) if reported_costs else None,
        "num_calls_with_cost": total("num_calls_with_cost"),
        "cost_complete": all_known and all(s.get("cost_complete") for s in known),
        "list_price_cost_usd": (
            sum(list_price_costs)
            if all_known and None not in list_price_costs
            else None
        ),
    }
    if list_prices:
        combined["list_price_per_token"] = list_prices
    combined["num_votes"] = total("num_votes")
    combined["num_invalid_votes"] = total("num_invalid_votes")
    return combined


def log_usage_summary(summary: dict) -> None:
    """Log a summary from `summarise_usage` and warn about invalid votes."""
    this_run = summary["this_run"]
    if this_run["num_calls"] == 0:
        logger.info(
            "API usage of this run: no API calls"
            + (
                " (all votes were loaded from the vote cache)."
                if summary["num_votes"]
                else "."
            )
        )
    else:
        logger.info(f"API usage of this run: {_format_usage(this_run)}")
    if summary["num_calls"] != this_run["num_calls"] or summary["num_unfinished_calls"]:
        logger.info(
            "API usage of all runs with this output directory: "
            + _format_usage(summary)
        )

    failed = this_run["num_failed_calls"]
    truncated = this_run["num_truncated_calls"]
    unparsable = this_run["num_unparsable_responses"]
    if not (summary["num_invalid_votes"] or failed or truncated or unparsable):
        return
    message = (
        f"{summary['num_invalid_votes']}/{summary['num_votes']} votes in the output "
        "are invalid (stored as 'invalid')."
    )
    if failed or truncated or unparsable:
        message += (
            f" Of the {this_run['num_calls']} API calls of this run, {failed} failed "
            f"(no response after retries), {truncated} hit the max output tokens "
            f"(truncated) and {unparsable} had an unparsable response. "
            "Re-running with the same output directory repeats calls that produced "
            "no valid vote. For truncated calls of reasoning models, consider "
            "increasing --max-output-tokens or lowering --reasoning-effort (votes "
            "are only reused from the cache for the same settings)."
        )
    else:
        message += (
            " The annotator's responses contained no valid vote for these "
            "principles."
        )
    logger.warning(message)


def _format_usage(s: dict) -> str:
    num_successful = s["num_calls"] - s["num_failed_calls"]
    if s["cost_usd"] is None:
        cost_str = (
            "not reported by provider"
            if num_successful > 0
            else "none reported (no successful calls)"
        )
    else:
        cost_str = f"${s['cost_usd']:.4f}"
        if s["num_calls_with_cost"] < num_successful:
            cost_str += (
                f" (lower bound: reported for only "
                f"{s['num_calls_with_cost']}/{num_successful} successful calls)"
            )
        elif not s["cost_complete"]:
            cost_str += " (lower bound)"
    if s.get("list_price_cost_usd") is not None:
        cost_str += f", at list price: ${s['list_price_cost_usd']:.4f}"
    calls_str = f"{s['num_calls']} calls"
    num_unfinished = s["num_unfinished_calls"]
    if num_unfinished:
        calls_str += (
            f" (+ {num_unfinished} unfinished call{'s' if num_unfinished > 1 else ''}"
            " of interrupted runs, possibly billed but not included)"
        )
    return (
        f"{calls_str}, {s['input_tokens']:,} input tokens, "
        f"{s['output_tokens']:,} output tokens "
        f"({s['reasoning_tokens']:,} reasoning). Cost: {cost_str}"
    )


def get_list_prices(models) -> dict[str, dict]:
    """OpenRouter list prices (USD per token) of the `openrouter/` models given.

    Returns:
        Model -> {"prompt": ..., "completion": ...} for the models with known price.
    """
    list_prices = {}
    for model in models:
        if model is not None and model.startswith("openrouter/"):
            pricing = get_openrouter_pricing(model.split("/", 1)[1])
            if pricing is not None:
                list_prices[model] = pricing
    return list_prices


def get_openrouter_pricing(model_id: str) -> dict | None:
    """Get OpenRouter list price (USD per token) for a model, None if unavailable."""
    try:
        import requests

        response = requests.get(OPENROUTER_MODELS_URL, timeout=15)
        response.raise_for_status()
        for model in response.json()["data"]:
            if model.get("id") == model_id or model.get("canonical_slug") == model_id:
                return {
                    "prompt": float(model["pricing"]["prompt"]),
                    "completion": float(model["pricing"]["completion"]),
                }
        logger.warning(f"Model '{model_id}' not found in OpenRouter price list.")
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(f"Could not get OpenRouter list price for '{model_id}': {e}")
    return None

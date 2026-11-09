"""Logging of LLM API calls: token usage, cost and failures.

Every API call made during annotation is appended as one JSON line to
`api_calls.jsonl` (token usage, cost, finish reason, duration; with
`log_content=True` also the request messages, response text and full response
metadata), and totals are summarised at the end of a run.

Cost:
- `cost_usd`: cost as reported by the provider. OpenRouter reports it (requested
  via `usage: {include: true}`, see `feedback_forensics.models`), other providers
  do not (null). Note that providers sometimes report a cost of 0 (e.g. free
  promotions), hence also:
- `list_price_cost_usd` (summary only): tokens x OpenRouter's public list price
  for the model (`openrouter/` models only, best effort).
"""

import datetime
import json
import threading
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
            log_path: JSONL file to append one entry per API call to (None: no log).
            log_content: Also log request messages, response text and full
                response metadata (larger log files).
        """
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_content = log_content
        self._lock = threading.Lock()
        self.num_calls = 0
        self.num_failed_calls = 0
        self.num_truncated_calls = 0
        self.num_unparsable_responses = 0
        self.num_calls_with_cost = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0

    def record(
        self,
        model: str,
        response: Any = None,
        latency_s: float | None = None,
        error: BaseException | None = None,
        messages: list | None = None,
        **extra,
    ) -> dict:
        """Record a single API call (successful if `response` given, else failed)."""
        entry = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "success": error is None and response is not None,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
        }
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

        with self._lock:
            self.num_calls += 1
            if not entry["success"]:
                self.num_failed_calls += 1
            if entry.get("finish_reason") in ("length", "max_tokens"):
                self.num_truncated_calls += 1
            self.input_tokens += entry.get("input_tokens") or 0
            self.output_tokens += entry.get("output_tokens") or 0
            self.reasoning_tokens += entry.get("reasoning_tokens") or 0
            if entry.get("cost_usd") is not None:
                self.num_calls_with_cost += 1
                self.cost_usd += entry["cost_usd"]
            if self.log_path is not None:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        return entry

    def record_unparsable_response(self) -> None:
        with self._lock:
            self.num_unparsable_responses += 1

    def summary(self, model: str | None = None) -> dict:
        """Totals across all recorded calls.

        Args:
            model: If an `openrouter/` model name is given, the list-price cost
                is computed from OpenRouter's public pricing (needs network).
        """
        num_successful = self.num_calls - self.num_failed_calls
        summary = {
            "num_calls": self.num_calls,
            "num_failed_calls": self.num_failed_calls,
            "num_truncated_calls": self.num_truncated_calls,
            "num_unparsable_responses": self.num_unparsable_responses,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            # None if the provider did not report cost for any call
            "cost_usd": (self.cost_usd if self.num_calls_with_cost > 0 else None),
            "num_calls_with_cost": self.num_calls_with_cost,
            "cost_complete": (
                self.num_calls_with_cost > 0
                and self.num_calls_with_cost == num_successful
            ),
            "list_price_cost_usd": None,
        }
        if model is not None and model.startswith("openrouter/"):
            pricing = get_openrouter_pricing(model.split("/", 1)[1])
            if pricing is not None:
                summary["list_price_cost_usd"] = (
                    self.input_tokens * pricing["prompt"]
                    + self.output_tokens * pricing["completion"]
                )
                summary["list_price_per_token"] = pricing
        return summary

    def log_summary(self, summary: dict | None = None) -> None:
        s = summary if summary is not None else self.summary()
        if s["cost_usd"] is None:
            cost_str = "not reported by provider"
        else:
            cost_str = f"${s['cost_usd']:.4f}"
            if not s["cost_complete"]:
                cost_str += (
                    f" (lower bound: reported for only "
                    f"{s['num_calls_with_cost']}/{s['num_calls']} calls)"
                )
        if s.get("list_price_cost_usd") is not None:
            cost_str += f", at list price: ${s['list_price_cost_usd']:.4f}"
        logger.info(
            f"API usage: {s['num_calls']} calls, "
            f"{s['input_tokens']:,} input tokens, {s['output_tokens']:,} output tokens "
            f"({s['reasoning_tokens']:,} reasoning). Cost: {cost_str}"
        )
        problems = {
            "failed (no response after retries)": s["num_failed_calls"],
            "truncated (hit max output tokens)": s["num_truncated_calls"],
            "with unparsable response": s["num_unparsable_responses"],
        }
        if any(problems.values()):
            logger.warning(
                "Some API calls did not produce valid votes (affected votes are "
                "stored as 'invalid'): "
                + ", ".join(f"{v}/{s['num_calls']} {k}" for k, v in problems.items())
                + ". For truncated calls of reasoning models, consider increasing "
                "--max-output-tokens or lowering --reasoning-effort."
            )


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

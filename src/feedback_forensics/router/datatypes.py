"""Plain data types of the model-call router (see the package README)."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

LIVE, BATCH = "live", "batch"
# Cache lifetimes a breakpoint can ask for (`Step.cache_ttl`)
TTL_5M, TTL_1H = "5m", "1h"


@dataclass(frozen=True)
class Message:
    """An ordinary chat message."""

    role: str
    text: str

    def __post_init__(self):
        if not self.role or any(c in self.role for c in "\n\x1e\x1f"):
            raise ValueError(f"Invalid message role: {self.role!r}")


@dataclass(frozen=True)
class Params:
    """Generation settings of a call (None: provider default)."""

    max_tokens: int | None = None
    reasoning_effort: str | None = None
    temperature: float | None = 0.0

    def as_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            # (0 and 0.0 are the same setting, and must give the same key)
            "temperature": (
                None if self.temperature is None else float(self.temperature)
            ),
        }


def call_key(model: str, params: Params, messages: tuple[Message, ...]) -> str:
    """Key of a call: SHA-256 (hex) of the JSON of its model, params and messages
    (sorted keys, compact separators, UTF-8, non-ASCII characters not escaped)."""
    content = {
        "model": model,
        "params": params.as_dict(),
        "messages": [[m.role, m.text] for m in messages],
    }
    text = json.dumps(
        content, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Call:
    """One LLM call.

    `model` uses the repository's naming ('openrouter/<provider>/<model>').
    `tag` is opaque to the router and returned untouched with the result (it is
    also written to the store, so it should be JSON data). `key` identifies the
    call (see `call_key`): it depends on model, params and messages only, not on
    the tag, and not on where the prompt is later cut for caching.
    """

    model: str
    messages: tuple[Message, ...]
    params: Params = Params()
    tag: Any = field(default=None, compare=False)
    key: str = field(init=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ValueError("A call needs at least one message.")
        object.__setattr__(
            self, "key", call_key(self.model, self.params, self.messages)
        )

    @property
    def text(self) -> str:
        """The prompt as one text: character offsets (`PrefixNode.end`,
        `Step.breakpoints`) count characters of this text."""
        return "".join(m.text for m in self.messages)


@dataclass(frozen=True)
class Usage:
    """Token usage and cost of one call, as reported by the provider (None: not
    reported). `input_tokens` includes the cached and cache-written tokens,
    `output_tokens` the reasoning tokens."""

    input_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class Result:
    """The answer to a call, or why there is none (`error`).

    `route`: "store" (answered before this run), "live" or "batch"; None if the
    call has no answer. `raw` is the provider's response (None for answers taken
    from the store).
    """

    key: str
    tag: Any = None
    text: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    usage: Usage = Usage()
    route: str | None = None
    raw: Any = field(default=None, compare=False, repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Policy:
    """What the caller needs from a run.

    Args:
        deadline: How long the run may take. None: results are needed now, so
            nothing is sent in batches. A run also stops waiting for pending
            batches after this time (running again continues).
        budget_usd: Stop sending once this run's spending (reported cost of
            finished calls plus the estimated cost of calls under way) would
            exceed it. None: no limit.
        max_concurrency: Live calls in flight at the same time.
        retries: Further attempts per call (on the live lane) after a failed
            call or an answer rejected by `accept`.
        accept: The caller's check that an answer is usable. Rejected answers
            are still stored (they are paid for), retried, and reported if
            still rejected. Also applied to stored answers.
    """

    deadline: timedelta | None = None
    budget_usd: float | None = None
    max_concurrency: int = 50
    retries: int = 2
    accept: Callable[[Result], bool] | None = None

    def __post_init__(self):
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        if self.retries < 0:
            raise ValueError("retries must not be negative.")


@dataclass(frozen=True)
class ModelProfile:
    """Prices and caching behaviour of a model: everything the planner knows
    about it. Prices are in USD per million tokens.

    Args:
        model: Model id on OpenRouter ('<provider>/<model>').
        cache_read_multiplier: Price of input tokens read from the cache,
            relative to `input_price`.
        cache_write_multiplier_5m, cache_write_multiplier_1h: Price of input
            tokens written to the cache (with a lifetime of 5 minutes / 1 hour).
        min_cacheable_tokens: Shorter prefixes are not cached.
        cache_lifetime: A cache entry expires this long after its last use
            (default lifetime).
        max_breakpoints: Cache breakpoints per call. 0: the model does not
            cache at breakpoints (none are sent).
        chars_per_token: To estimate token counts.
        batch_model: Model id of the batch variant, None if there is none.
        batch_price_multiplier: Batch prices relative to the prices above.
        cache_in_batch: Whether breakpoints lead to cache reads in batches
            (None: not known).
        observed_hit_rate: Of the batch requests planned to read a prefix from
            the cache, the share that did, in the last run that had any (None:
            not measured). Recorded by the router.
    """

    model: str
    input_price: float
    output_price: float
    cache_read_multiplier: float = 1.0
    cache_write_multiplier_5m: float = 1.0
    cache_write_multiplier_1h: float = 1.0
    min_cacheable_tokens: int = 1024
    cache_lifetime: timedelta = timedelta(minutes=5)
    max_breakpoints: int = 0
    chars_per_token: float = 4.0
    batch_model: str | None = None
    batch_price_multiplier: float = 0.5
    cache_in_batch: bool | None = None
    observed_hit_rate: float | None = None

    def cache_write_multiplier(self, ttl: str) -> float:
        if ttl == TTL_1H:
            return self.cache_write_multiplier_1h
        return self.cache_write_multiplier_5m


@dataclass(frozen=True)
class PrefixNode:
    """A prompt prefix shared by two or more calls: the first `end` characters
    of the prompt (`Call.text`) of every call in `keys`. The root has `end` 0."""

    end: int
    tokens: int  # estimated
    keys: tuple[str, ...]  # of all calls below, in tree order
    children: tuple["PrefixNode", ...] = ()


@dataclass(frozen=True)
class PrefixTree:
    """The shared prefixes of the calls of one model. `calls` are in tree order
    (prompts sorted as text): calls sharing a prefix are next to each other."""

    model: str
    root: PrefixNode
    calls: tuple[Call, ...]


@dataclass(frozen=True)
class Step:
    """How one call is sent.

    Args:
        lane: "live" or "batch".
        breakpoints: Character offsets (into `Call.text`) of the cache
            breakpoints, ascending.
        after: Key of the one call that must have finished first (it writes the
            cache entry this call reads), or None.
        cache_ttl: Lifetime asked for at the breakpoints ("5m" or "1h").
        estimated_usd: Estimated cost of the call as planned.
    """

    call: Call
    lane: str = LIVE
    breakpoints: tuple[int, ...] = ()
    after: str | None = None
    cache_ttl: str = TTL_5M
    estimated_usd: float = 0.0


@dataclass(frozen=True)
class Stage:
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Estimate:
    """Estimated cost of a plan. `as_given_usd`: every call sent live at list
    price without any caching. `planned_usd`: as planned."""

    as_given_usd: float = 0.0
    planned_usd: float = 0.0
    input_tokens: int = 0
    cached_tokens: float = 0.0  # expected
    output_tokens: int = 0
    duration: timedelta = timedelta(0)


@dataclass(frozen=True)
class Plan:
    """Stages are run one after another. `strategy` is the strategy that made
    the plan ("mixed" if it differs between models, see `choices`: pairs of
    model and strategy)."""

    strategy: str
    stages: tuple[Stage, ...] = ()
    estimate: Estimate = Estimate()
    choices: tuple[tuple[str, str], ...] = ()

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(step for stage in self.stages for step in stage.steps)


@dataclass(frozen=True)
class RunReport:
    """What a run did.

    Args:
        strategy: Of the plan.
        calls: Distinct calls queued.
        reused: Of these, answered from the store without sending anything.
        sent_live, sent_batch: Requests sent in this run (retries count).
        interrupted: Live calls an earlier run sent without recording an answer
            (it was interrupted): they may have been billed, and were sent again.
        cached_share: Share of the input tokens that was read from the cache.
        cost_usd: Cost of the answers received in this run, as reported by the
            provider (computed from tokens and profile prices where it was not).
        estimated_usd: The plan's estimate for the calls sent in this run.
        gave_up: (key, reason) of the calls without a usable answer that are
            not pending either.
        pending: Keys of the calls in batches that are still pending: run again
            to collect them.
        problems: Things that need attention (e.g. a batch not known to the
            provider).
    """

    strategy: str | None = None
    calls: int = 0
    reused: int = 0
    sent_live: int = 0
    sent_batch: int = 0
    interrupted: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_share: float = 0.0
    cost_usd: float = 0.0
    estimated_usd: float = 0.0
    gave_up: tuple[tuple[str, str], ...] = ()
    pending: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()

    @property
    def sent(self) -> int:
        return self.sent_live + self.sent_batch

    @property
    def complete(self) -> bool:
        """Whether every queued call has a usable answer."""
        return not self.gave_up and not self.pending


@dataclass(frozen=True)
class LayoutFinding:
    """Text that repeats across calls behind text unique to each call.

    Args:
        calls: Number of calls containing the text.
        offset: Where it starts (characters from the start of the prompt,
            median over the calls).
        tokens: Its estimated length.
        saving_usd: Estimated saving if it were moved before the call-specific
            text (so that it can be cached).
        preview: Its first characters.
    """

    model: str
    calls: int
    offset: int
    tokens: int
    saving_usd: float
    preview: str


@dataclass(frozen=True)
class LayoutReport:
    findings: tuple[LayoutFinding, ...] = ()

    def __str__(self) -> str:
        if not self.findings:
            return "Layout check: no repeated text behind call-specific text."
        lines = [
            "Layout check: text that repeats across calls, but follows text unique "
            "to a call, so that no cache can reach it. Suggestions only: consider "
            "moving it before the call-specific text if the prompt allows it (some "
            "repeats cannot be moved, e.g. text that several responses quote)."
        ]
        for f in self.findings:
            lines.append(
                f"- {f.model}: ~{f.tokens} tokens in {f.calls} calls, around "
                f"character {f.offset}, estimated saving if cached "
                f"${f.saving_usd:.4f}: {f.preview!r}"
            )
        return "\n".join(lines)

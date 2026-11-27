"""Fakes for tests: a provider with a prompt cache and a batch API, and an event
loop with virtual time. No network.

`FakeProvider` implements the transport interface. Its cache works like a
provider's prompt cache, as far as it matters for planning:

- An entry (a prompt prefix up to a breakpoint) is written when the call that
  wrote it finishes, and expires after its lifetime (5 minutes, or 1 hour if
  asked for) unless it is read again.
- A call only reads entries that exist when it starts: calls in flight at the
  same time do not see each other's entries, and each pays to write its own.
- Prefixes shorter than the model's minimum are not cached, and models without
  breakpoints (`ModelProfile.max_breakpoints` 0) cache nothing.

Batches are answered `batch_delay` seconds after their submission. Their
requests read the entries that exist at submission (if `batch_cache`; entries
written by live calls only if `batch_reads_live_cache`), not each other's.

Time is the time of the event loop: run the code under test with `run_virtual`,
which makes waiting (e.g. `asyncio.sleep`) return at once while time advances.
"""

import asyncio
import hashlib
import math

from feedback_forensics.models import openrouter
from feedback_forensics.router.datatypes import TTL_1H, TTL_5M
from feedback_forensics.router.profiles import get_profile
from feedback_forensics.router.transports import (
    BatchNotFound,
    Request,
    TransportError,
)


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """Event loop whose clock jumps to the next timer instead of waiting for it.

    Relies on internals of asyncio's event loop (`_ready`, `_scheduled`,
    `_run_once`). Only for code that does not wait for threads or sockets.
    """

    def __init__(self, start: float = 0.0):
        super().__init__()
        self._virtual_time = start

    def time(self) -> float:
        return self._virtual_time

    def _run_once(self):
        if not self._ready and self._scheduled:
            self._virtual_time = max(self._virtual_time, self._scheduled[0].when())
        super()._run_once()


def run_virtual(coroutine, start: float = 0.0):
    """Run a coroutine in a new event loop with virtual time, starting at
    `start` seconds (e.g. to continue later than an earlier run ended)."""
    loop = VirtualTimeLoop(start)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


class SimulatedCrash(Exception):
    """Raised by `FakeProvider` to interrupt a run (see `crash_after`)."""


class FakeProvider:
    """Fake transport: live calls and batches against a simulated prompt cache.

    Args:
        profiles: Model profiles (prices, cache behaviour).
        latency: Seconds a live call takes.
        batch_delay: Seconds until a batch has ended.
        answer: Function (request) -> answer text.
        batch_cache: Whether batch requests read from and write to the cache.
        batch_reads_live_cache: Whether batch requests read entries written by
            live calls.

    Attributes to set up failures:
        live_errors: key -> exceptions to raise on the next live attempts.
        crash_after: Raise `SimulatedCrash` on every live request after this
            many (None: never).
        failed_in_batch: Keys of requests that fail within their batch.
        batch_status: Status of batches that ended ("completed", or e.g.
            "expired": then only `partial_results` of the results are returned).
        lose_submit_response: True: the batch is created, but the response is
            lost. "not_created": the request is lost.
        reject_submit: The submission is rejected (HTTP 400).
        unknown_batches: Batches are not known (`BatchNotFound`).
        extra_results: Raw results appended to those of every batch.

    Records: `sent` (keys of live requests, in order), `requests` (all live
    requests), `submissions` (keys per submitted batch), `calls` (per answered
    request: key, lane, start, end, usage).
    """

    def __init__(
        self,
        profiles,
        latency: float = 5.0,
        batch_delay: float = 3600.0,
        answer=None,
        batch_cache: bool = True,
        batch_reads_live_cache: bool = True,
    ):
        self.profiles = profiles
        self.latency = latency
        self.batch_delay = batch_delay
        self.answer = answer or (lambda request: f"answer {request.key[:8]}")
        self.batch_cache = batch_cache
        self.batch_reads_live_cache = batch_reads_live_cache
        self.cache: dict[tuple, tuple[float, str]] = {}  # -> (expiry, written by)
        self.live_errors: dict[str, list[Exception]] = {}
        self.crash_after: int | None = None
        self.failed_in_batch: set[str] = set()
        self.batch_status = "completed"
        self.partial_results: int | None = None
        self.lose_submit_response: bool | str = False
        self.reject_submit = False
        self.unknown_batches = False
        self.extra_results: list = []
        self.sent: list[str] = []
        self.requests: list[Request] = []
        self.submissions: list[list[str]] = []
        self.calls: list[dict] = []
        self.batches: dict[str, dict] = {}
        self.polls = 0

    # --- Cache ----------------------------------------------------------------

    def _serve(self, request: Request, lane: str, now: float):
        """(chat completion, cache entries to write once the call finished)."""
        profile = get_profile(self.profiles, request.model)
        caching = profile.max_breakpoints > 0 and (lane == "live" or self.batch_cache)
        digest, chars, breakpoints = hashlib.sha256(), 0, []
        for message in request.body["messages"]:
            digest.update(f"\x1e{message['role']}\x1f".encode())
            content = message["content"]
            blocks = [{"text": content}] if isinstance(content, str) else list(content)
            for block in blocks:
                digest.update(block["text"].encode())
                chars += len(block["text"])
                control = block.get("cache_control")
                tokens = math.ceil(chars / profile.chars_per_token)
                if control and caching and tokens >= profile.min_cacheable_tokens:
                    ttl = TTL_1H if control.get("ttl") == "1h" else TTL_5M
                    entry = (request.model, digest.copy().hexdigest())
                    breakpoints.append((entry, tokens, ttl))
        total = math.ceil(chars / profile.chars_per_token)

        cached = 0
        for entry, tokens, ttl in reversed(breakpoints):
            expiry, written_by = self.cache.get(entry, (0.0, ""))
            visible = lane == "live" or written_by == "batch"
            if expiry > now and (visible or self.batch_reads_live_cache):
                cached = tokens
                # Reading an entry renews its lifetime
                self.cache[entry] = (now + _seconds(profile, ttl), written_by)
                break
        writes = [(e, ttl) for e, tokens, ttl in breakpoints if tokens > cached]
        written = (breakpoints[-1][1] - cached) if writes else 0
        ttl = writes[-1][1] if writes else TTL_5M

        text = self.answer(request)
        output = math.ceil(len(text) / 4)
        price = profile.batch_price_multiplier if lane == "batch" else 1.0
        cost = (
            price
            * (
                profile.input_price
                * (
                    (total - cached - written)
                    + cached * profile.cache_read_multiplier
                    + written * profile.cache_write_multiplier(ttl)
                )
                + profile.output_price * output
            )
            / 1e6
        )
        completion = {
            "id": f"gen-{len(self.calls)}",
            "model": request.model,
            "provider": "Fake",
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": total,
                "completion_tokens": output,
                "prompt_tokens_details": {
                    "cached_tokens": cached,
                    "cache_write_tokens": written,
                },
                "completion_tokens_details": {"reasoning_tokens": 0},
                "cost": cost,
            },
        }
        return completion, writes

    def _write(self, writes, now: float, model: str, lane: str) -> None:
        profile = get_profile(self.profiles, model)
        for entry, ttl in writes:
            self.cache[entry] = (now + _seconds(profile, ttl), lane)

    # --- Live -----------------------------------------------------------------

    async def send(self, request: Request) -> dict:
        self.sent.append(request.key)
        self.requests.append(request)
        if self.crash_after is not None and len(self.sent) > self.crash_after:
            raise SimulatedCrash()
        errors = self.live_errors.get(request.key)
        if errors:
            raise errors.pop(0)
        loop = asyncio.get_running_loop()
        start = loop.time()
        completion, writes = self._serve(request, "live", start)
        await asyncio.sleep(self.latency)
        self._write(writes, loop.time(), request.model, "live")
        self.calls.append(
            {
                "key": request.key,
                "lane": "live",
                "start": start,
                "end": loop.time(),
                "usage": completion["usage"],
            }
        )
        return completion

    # --- Batches --------------------------------------------------------------

    async def submit(self, model, requests, entry, save, discard) -> str:
        entry.update(
            batch_id=None,
            num_requests=len(requests),
            submitted_at=openrouter.now().isoformat(),
        )
        save()
        if self.reject_submit:
            discard()
            raise TransportError("rejected", 400)
        if self.lose_submit_response == "not_created":
            raise TransportError("bad gateway", 502, retryable=True)
        now = asyncio.get_running_loop().time()
        batch_id = f"batch_{len(self.batches) + 1}"
        results, writes = [], []
        for request in requests:
            if request.key in self.failed_in_batch:
                result = {"response": None, "error": {"message": "failed"}}
            else:
                completion, new = self._serve(request, "batch", now)
                writes += new
                self.calls.append(
                    {
                        "key": request.key,
                        "lane": "batch",
                        "start": now,
                        "end": now + self.batch_delay,
                        "usage": completion["usage"],
                    }
                )
                response = {"status_code": 200, "body": completion}
                result = {"response": response, "error": None}
            results.append({"custom_id": request.key, **result})
        self.batches[batch_id] = {
            "model": model,
            "results": results,
            "writes": writes,
            "ready": now + self.batch_delay,
        }
        self.submissions.append([request.key for request in requests])
        if self.lose_submit_response:
            raise TransportError("bad gateway", 502, retryable=True)
        entry["batch_id"] = batch_id
        save()
        return batch_id

    def _ended(self, batch_id: str) -> dict | None:
        """The batch if it has ended (its cache entries are then written)."""
        batch = self.batches[batch_id]
        now = asyncio.get_running_loop().time()
        if now < batch["ready"]:
            return None
        if batch["writes"]:
            self._write(batch["writes"], batch["ready"], batch["model"], "batch")
            batch["writes"] = []
        return batch

    async def poll(self, entry):
        self.polls += 1
        batch_id = entry["batch_id"]
        if self.unknown_batches or batch_id not in self.batches:
            raise BatchNotFound(f"batch {batch_id} not found", 404)
        batch = self._ended(batch_id)
        if batch is None:
            return None, "in_progress"
        results = batch["results"][: self.partial_results] + self.extra_results
        status = self.batch_status
        return {
            "id": batch_id,
            "status": status,
            "results": results,
            "usage": {"cost": None},
            "error": None if status == "completed" else {"message": status},
        }, status

    async def recover(self, model, entry, exclude_ids):
        resolved = True
        for batch_id, batch in self.batches.items():
            if (
                batch["model"] != model
                or len(batch["results"]) != entry["num_requests"]
                or batch_id in exclude_ids
                or batch_id in entry.setdefault("not_ours", [])
            ):
                continue
            if self._ended(batch_id) is None:
                resolved = False  # without results yet
            elif {r["custom_id"] for r in batch["results"]} & set(entry["custom_ids"]):
                return True, batch_id
            else:
                entry["not_ours"].append(batch_id)
        return resolved, None


def _seconds(profile, ttl: str) -> float:
    if ttl == TTL_1H:
        return 3600.0
    return profile.cache_lifetime.total_seconds()

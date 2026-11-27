"""The router: queue calls, run them at the lowest cost, collect the answers.

See the package README. This module resolves calls against the store, asks the
planner for a plan and executes it; it knows nothing about what the calls are
for.

Working state is kept in `<store>/.router/`:

- `state.jsonl`: the state of every call before and after sending (a log).
- `batches.json`: the batches under way. An entry is written before a batch is
  submitted and removed once its results are stored, so a batch is never
  submitted twice, and a batch whose submission response was lost is found
  again (see `feedback_forensics.models.openrouter`).
- `plan.json`: how the calls of an unfinished run are sent (without their
  text), so that running again continues the same plan.
- `unmatched/<batch id>.json`: batch results that match no queued call or
  cannot be read, kept as received: no paid result is lost.
- `profiles.json`: cache hit rates observed in batches, per model.
"""

import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from filelock import FileLock, Timeout
from loguru import logger

from feedback_forensics.models.openrouter import get_result_body, write_json_atomic
from feedback_forensics.router.layout import check_layout as _check_layout
from feedback_forensics.router.profiles import (
    as_profile_map,
    get_profile,
    model_id,
    with_observation,
)
from feedback_forensics.router.store import DEFAULT_TASK, Store
from feedback_forensics.router.strategies import STRATEGIES, make_plan
from feedback_forensics.router.transports import (
    AccountError,
    BatchNotFound,
    Request,
    Transport,
    TransportError,
)
from feedback_forensics.router.datatypes import (
    BATCH,
    LIVE,
    TTL_1H,
    Call,
    LayoutReport,
    Plan,
    Policy,
    Result,
    RunReport,
    Stage,
    Step,
    Usage,
)
from feedback_forensics.router.wire import build_body, parse_completion

# Requests per batch (more are split into several batches), as for annotation
# (`feedback_forensics.annotation.batch`): not a limit verified with OpenRouter.
MAX_REQUESTS_PER_BATCH = 10_000
POLL_INTERVAL = 60  # Seconds between status checks of pending batches
MAX_RETRY_WAIT = 60  # Seconds, upper bound of the wait before a retry


class RouterBusyError(RuntimeError):
    """Another run is using the store (it could send the same calls)."""


class Router:
    """Gets a set of LLM calls answered at the lowest cost.

    Queue all calls with `add`, `run` once, then `collect` the results.

    Args:
        store: Directory of the call store (or a `Store`).
        strategy: Name of a registered strategy (see
            `feedback_forensics.router.strategies`).
        profiles: Model profiles (mapping or list). Default: the profiles in
            `feedback_forensics.router.profiles.DEFAULT_PROFILES`.
        transport: How requests are sent. Default: OpenRouter.
    """

    def __init__(
        self,
        store: str | Path | Store,
        strategy: str = "auto",
        profiles=None,
        transport: Transport | None = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}' (known: {sorted(STRATEGIES)})."
            )
        self.store = store if isinstance(store, Store) else Store(store)
        self.strategy = strategy
        self.transport = transport
        self.state_dir = self.store.path / ".router"
        self.profiles = as_profile_map(profiles)
        for model, observed in _read_json(self.state_dir / "profiles.json").items():
            if model in self.profiles:
                self.profiles[model] = with_observation(
                    self.profiles[model], observed["observed_hit_rate"]
                )
        self._queue: dict[str, Call] = {}
        self._folders: dict[str, tuple[str, str | None]] = {}
        # Of the last run: how answers were obtained, and calls without answer
        self._routes: dict[str, str] = {}
        self._failures: dict[str, str] = {}

    def add(
        self, calls: Iterable[Call], task: str = DEFAULT_TASK, group: str | None = None
    ) -> list[str]:
        """Queue calls (nothing is sent). Returns their keys, in order.

        A call queued twice (same model, params and messages) is answered once:
        both get the same result, with the tag of the call queued first.

        Args:
            task, group: Where the answers are stored:
                `<task>/<model>.jsonl`, or `<task>/<model>/<group>.jsonl` with
                a group (e.g. task "annotation", group: the annotated model).
        """
        keys = []
        for call in calls:
            keys.append(call.key)
            if call.key not in self._queue:
                self._queue[call.key] = call
                self._folders[call.key] = (task, group)
        return keys

    def dry_run(
        self, policy: Policy = Policy(), check_layout: bool = False
    ) -> tuple[Plan, LayoutReport | None]:
        """The plan for the queued calls that have no stored answer (as `run`
        would make it for a new run), without sending anything or writing any
        state. With `check_layout`, also a report of text that repeats across
        the queued calls where no cache can reach it (see
        `feedback_forensics.router.layout`)."""
        todo = [c for c in self._queue.values() if not self._answered(c, policy)]
        plan = make_plan(todo, self.profiles, policy, self.strategy)
        layout = None
        if check_layout:
            layout = _check_layout(self._queue.values(), self.profiles)
        return plan, layout

    def run(self, policy: Policy = Policy()) -> RunReport:
        """Plan the queued calls, send them and wait until they are answered
        (see `arun`, to be used inside a running event loop)."""
        return asyncio.run(self.arun(policy))

    async def arun(self, policy: Policy = Policy()) -> RunReport:
        """Plan the queued calls, send them and wait until they are answered.

        Resumable: every answer is stored as it arrives, and batches are
        tracked. If the run is interrupted, or batches are still pending at the
        deadline (`RunReport.pending`), running again continues: stored answers
        are not requested again and pending batches are not submitted again.
        Only live calls that were in flight when a run was interrupted are sent
        again (`RunReport.interrupted`).

        Raises:
            AccountError: The provider rejected the API key or account. No
                further calls were sent; answers received before are stored.
            RouterBusyError: Another run is using the store.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self.state_dir / "lock")
        try:
            lock.acquire(timeout=0)
        except Timeout as e:
            raise RouterBusyError(
                f"Another run is using the call store {self.store.path}."
            ) from e
        if self.transport is None:
            from feedback_forensics.router.transports import OpenRouterTransport

            self.transport = OpenRouterTransport()
        try:
            return await _Run(self, policy).execute()
        finally:
            lock.release()
            close = getattr(self.transport, "aclose", None)
            if close is not None:
                await close()

    def collect(self, keys: Iterable[str]) -> dict[str, Result]:
        """The results of calls, by key. Every key gets a result: the stored
        answer (`route`: "live" or "batch" if it was obtained by the last run
        of this router, else "store"), or a result with `error` if there is
        none. Tags are those of the queued calls."""
        results = {}
        for key in keys:
            call = self._queue.get(key)
            tag = call.tag if call is not None else None
            result = self.store.result(key, tag, self._routes.get(key, "store"))
            if result is None:
                error = self._failures.get(key, "no answer (not run yet)")
                result = Result(key=key, tag=tag, error=error)
            results[key] = result
        return results

    def _answered(self, call: Call, policy: Policy) -> bool:
        """Whether the store has a usable answer of a call."""
        result = self.store.result(call.key, call.tag)
        if result is None:
            return False
        return policy.accept is None or bool(policy.accept(result))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Run:
    """One run of a router."""

    def __init__(self, router: Router, policy: Policy):
        self.router = router
        self.policy = policy
        self.store = router.store
        self.transport = router.transport
        self.queue = router._queue
        self.state_dir = router.state_dir
        self.batches_file = self.state_dir / "batches.json"
        self.plan_file = self.state_dir / "plan.json"
        self.batches: list[dict] = _read_json(self.batches_file).get("batches", [])
        self.steps: dict[str, Step] = {}
        # Keys of calls that got an answer in this run (their cache is written)
        self.answered: set[str] = set()
        self.gave_up: dict[str, str] = {}
        self.problems: list[str] = []
        self.stopped: str | None = None  # why nothing more is sent
        # A request to the batch API failed: no further ones in this run
        self.batch_trouble = False
        # Batches submitted by this run without response (ids of their entries)
        self.unconfirmed: set[int] = set()
        self.reserved_batches: set[str] = set()  # their cost counts as reserved
        # Whether a call with breakpoints was answered while this run executed
        # its stages: the prefixes later stages read were written just now
        self.executing = self.fresh_prefix = False
        self.account_error: AccountError | None = None
        self.spent = self.reserved = self.estimated = 0.0
        self.sent_live = self.sent_batch = 0
        self.tokens = dict.fromkeys(
            ("input", "cached", "cache_write", "output", "reasoning"), 0
        )
        # model -> [batch requests that read a prefix, that were planned to]
        self.batch_reads: dict[str, list[int]] = {}
        router._routes, router._failures = {}, {}

    # --- State ----------------------------------------------------------------

    def _log(self, key: str, state: str, **fields) -> None:
        """Record the state of a call (before and after sending)."""
        entry = {"time": _now(), "key": key, "state": state, **fields}
        with open(self.state_dir / "state.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _interrupted(self, keys: set[str]) -> int:
        """Number of calls (of `keys`) that an earlier run sent live without
        recording an answer or failure."""
        path = self.state_dir / "state.jsonl"
        if not path.exists():
            return 0
        last = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    last[entry["key"]] = entry["state"]
                except (ValueError, KeyError, TypeError):
                    continue  # e.g. a line cut off by an interruption
        return sum(last.get(key) == "sent" for key in keys)

    def _save_batches(self) -> None:
        if self.batches:
            write_json_atomic(self.batches_file, {"batches": self.batches})
        else:
            self.batches_file.unlink(missing_ok=True)

    def _pending_keys(self) -> set[str]:
        return {key for entry in self.batches for key in entry["custom_ids"]}

    def _give_up(self, key: str, reason: str) -> None:
        self.gave_up[key] = reason
        self.router._failures[key] = reason

    # --- Plan -----------------------------------------------------------------

    def _load_plan(self, todo: set[str], unresolved: set[str]) -> Plan:
        """The plan of the run: that of an unfinished earlier run if it covers
        all calls without answer (same strategy and deadline), else a new one."""
        deadline = self.policy.deadline
        settings = {
            "strategy": self.router.strategy,
            "deadline_s": deadline.total_seconds() if deadline else None,
        }
        saved = _read_json(self.plan_file)
        stages = saved.get("stages", [])
        saved_keys = {step["key"] for stage in stages for step in stage}
        if (
            saved
            and all(saved.get(name) == value for name, value in settings.items())
            and unresolved <= saved_keys
        ):
            logger.info("Continuing the plan of an unfinished run.")
            return Plan(
                strategy=saved["plan_strategy"],
                stages=tuple(
                    Stage(
                        tuple(
                            Step(
                                call=self.queue[step["key"]],
                                lane=step["lane"],
                                breakpoints=tuple(step["breakpoints"]),
                                after=step["after"],
                                cache_ttl=step["cache_ttl"],
                                estimated_usd=step["estimated_usd"],
                            )
                            for step in stage
                            if step["key"] in unresolved
                        )
                    )
                    for stage in stages
                ),
            )
        calls = [call for key, call in self.queue.items() if key in todo]
        plan = make_plan(calls, self.router.profiles, self.policy, self.router.strategy)
        if plan.steps:
            saved = {
                **settings,
                "plan_strategy": plan.strategy,
                "stages": [
                    [
                        {
                            "key": step.call.key,
                            "lane": step.lane,
                            "breakpoints": list(step.breakpoints),
                            "after": step.after,
                            "cache_ttl": step.cache_ttl,
                            "estimated_usd": step.estimated_usd,
                        }
                        for step in stage.steps
                    ]
                    for stage in plan.stages
                ],
            }
            write_json_atomic(self.plan_file, saved)
        return plan

    # --- Execute --------------------------------------------------------------

    async def execute(self) -> RunReport:
        policy, queue = self.policy, self.queue
        loop = asyncio.get_running_loop()
        end = None
        if policy.deadline is not None:
            end = loop.time() + policy.deadline.total_seconds()

        reused = sum(self.router._answered(call, policy) for call in queue.values())
        # Results of batches that ended since the last run
        try:
            retry = await self._poll_batches()
        except TransportError as e:
            retry = []
            self._batch_api_failed(e)
        unresolved = {
            key
            for key, call in queue.items()
            if not self.router._answered(call, policy)
        }
        interrupted = self._interrupted(unresolved)
        plan = self._load_plan(unresolved - self._pending_keys(), unresolved)
        self.steps = {step.call.key: step for step in plan.steps}
        await self._run_live(retry, attempts_used=1)

        self.executing = True
        for index, stage in enumerate(plan.stages):
            if self.stopped:
                break
            pending = self._pending_keys()
            steps = [
                step
                for step in stage.steps
                if step.call.key not in pending
                and step.call.key not in self.answered
                and step.call.key not in self.gave_up
                and not self.router._answered(step.call, policy)
            ]
            if index > 0 and not self.fresh_prefix:
                steps = self._without_stale_breakpoints(steps)
            await self._submit(
                [step for step in steps if step.lane == BATCH], reads_prefix=index > 0
            )
            await self._run_live([step for step in steps if step.lane == LIVE])
            # The next stage reads what this one wrote: wait for its batches
            while self.batches and end is not None and not self.batch_trouble:
                try:
                    retry = await self._poll_batches()
                except TransportError as e:
                    self._batch_api_failed(e)
                    break
                await self._run_live(retry, attempts_used=1)
                if not self.batches or loop.time() + POLL_INTERVAL > end:
                    break
                await asyncio.sleep(POLL_INTERVAL)
            if self.batches:
                break

        pending = self._pending_keys()
        for key in unresolved:
            if key not in self.answered and key not in pending:
                self.gave_up.setdefault(key, self.stopped or "not sent")
                self.router._failures[key] = self.gave_up[key]
        for key in pending & set(queue):
            self.router._failures[key] = "pending (in a batch): run again to collect"
        if not self.batches and not self.gave_up:
            self.plan_file.unlink(missing_ok=True)
        self._record_hit_rates()
        report = RunReport(
            strategy=plan.strategy if plan.steps else None,
            calls=len(queue),
            reused=reused,
            sent_live=self.sent_live,
            sent_batch=self.sent_batch,
            interrupted=interrupted,
            input_tokens=self.tokens["input"],
            cached_tokens=self.tokens["cached"],
            cache_write_tokens=self.tokens["cache_write"],
            output_tokens=self.tokens["output"],
            reasoning_tokens=self.tokens["reasoning"],
            cached_share=self.tokens["cached"] / max(self.tokens["input"], 1),
            cost_usd=self.spent,
            estimated_usd=self.estimated,
            gave_up=tuple(sorted(self.gave_up.items())),
            pending=tuple(sorted(pending & set(queue))),
            problems=tuple(self.problems),
        )
        logger.info(
            f"Router run: {report.calls} calls, {report.reused} answered from the "
            f"store, {report.sent_live} live and {report.sent_batch} batch requests "
            f"sent, {len(report.pending)} pending, {len(report.gave_up)} given up. "
            f"Input tokens: {report.input_tokens:,} ({report.cached_share:.0%} read "
            f"from cache). Cost: ${report.cost_usd:.4f} (estimated: "
            f"${report.estimated_usd:.4f})."
        )
        if self.account_error is not None:
            raise self.account_error
        return report

    def _without_stale_breakpoints(self, steps: list[Step]) -> list[Step]:
        """Batch steps of a later stage read a prefix that an earlier stage
        wrote. If that stage was not answered while this run executed (but by
        an earlier run, or at an unknown time before this one started), the
        cache entry may have expired, and every request would pay to write it
        again: such steps are sent without breakpoints."""
        stale = [s for s in steps if s.lane == BATCH and s.breakpoints]
        if not stale:
            return steps
        logger.warning(
            f"{len(stale)} batch requests were planned to read a prefix written "
            "by an earlier stage. That stage ended before this run: the prefix "
            "may have expired, so they are sent without cache breakpoints."
        )
        steps = [
            Step(s.call, BATCH, estimated_usd=s.estimated_usd) if s in stale else s
            for s in steps
        ]
        self.steps.update({step.call.key: step for step in steps})
        return steps

    def _batch_api_failed(self, error: TransportError) -> None:
        """A request about the batches under way failed: they stay tracked, and
        no further requests are made to the batch API in this run."""
        self.batch_trouble = True
        self.problems.append(
            f"Request to the batch API failed ({error}). Run again to continue: "
            "submitted batches are tracked."
        )

    def _over_budget(self, cost: float) -> bool:
        budget = self.policy.budget_usd
        if budget is None or self.spent + self.reserved + cost <= budget:
            return False
        if not self.stopped:
            self.stopped = f"budget of ${budget} reached"
            logger.warning(f"Router: {self.stopped}, sending nothing more.")
        return True

    def _request(self, step: Step) -> Request:
        ttl = step.cache_ttl if step.cache_ttl == TTL_1H else None
        body = build_body(step.call, step.breakpoints, ttl)
        return Request(step.call.key, step.call.model, body)

    def _record(
        self, step: Step, body: dict, route: str, batch_id=None, reads_prefix=False
    ) -> Result:
        """Store the answer of a call and account for it.

        Raises:
            KeyError, IndexError, TypeError: The answer cannot be read.
        """
        call = step.call
        text, finish_reason, usage, provider = parse_completion(body)
        result = Result(
            key=call.key,
            tag=call.tag,
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            route=route,
            raw=body,
        )
        task, group = self.router._folders[call.key]
        self.store.put(
            call,
            result,
            task=task,
            group=group,
            breakpoints=step.breakpoints,
            batch_id=batch_id,
            provider=provider,
        )
        self._log(call.key, "answered", lane=route, batch_id=batch_id)
        self.answered.add(call.key)
        if self.executing and step.breakpoints:
            self.fresh_prefix = True
        self.router._routes[call.key] = route
        self.router._failures.pop(call.key, None)

        profile = get_profile(self.router.profiles, call.model)
        cost = usage.cost_usd
        if cost is None:
            cost = _cost_from_tokens(usage, profile, route, step.cache_ttl)
        self.spent += cost
        for name, value in (
            ("input", usage.input_tokens),
            ("cached", usage.cached_tokens),
            ("cache_write", usage.cache_write_tokens),
            ("output", usage.output_tokens),
            ("reasoning", usage.reasoning_tokens),
        ):
            self.tokens[name] += value or 0
        if reads_prefix and step.breakpoints:
            reads = self.batch_reads.setdefault(model_id(call.model), [0, 0])
            reads[0] += bool(usage.cached_tokens)
            reads[1] += 1
        return result

    def _record_hit_rates(self) -> None:
        """Record the cache hit rate observed in batches in the model profiles."""
        path = self.state_dir / "profiles.json"
        observed = _read_json(path)
        for model, (hits, planned) in self.batch_reads.items():
            rate = hits / planned
            observed[model] = {"observed_hit_rate": rate, "time": _now()}
            self.router.profiles[model] = with_observation(
                get_profile(self.router.profiles, model), rate
            )
            logger.info(
                f"{model}: {hits} of {planned} batch requests with a cache "
                "breakpoint read from the cache (recorded in the model profile)."
            )
        if self.batch_reads:
            write_json_atomic(path, observed)

    # --- Live lane ------------------------------------------------------------

    async def _run_live(self, steps: list[Step], attempts_used: int = 0) -> None:
        """Send steps on the live lane, in order.

        A step with `after` starts once that call has an answer (the cache
        entry it reads is then written); the steps waiting for the same call
        then run in parallel. If that call is not part of this run or fails,
        the first step waiting for it takes its place. Steps are started close
        to their place in the order (not more than a window ahead of the first
        unfinished one), so that calls sharing a prefix run close together.
        """
        if not steps:
            return
        keys = [step.call.key for step in steps]
        by_key = {step.call.key: step for step in steps}
        after: dict[str, str | None] = {}
        waiting: dict[str, list[str]] = {}  # call -> keys of steps waiting for it

        def wait_for(key: str, other: str | None) -> None:
            after[key] = other
            if other is not None:
                waiting.setdefault(other, []).append(key)

        def replace(writer: str) -> str | None:
            """The first step waiting for `writer` takes its place."""
            first, *others = waiting.pop(writer, [None])
            if first is not None:
                after[first] = None
                for key in others:
                    wait_for(key, first)
            return first

        for step in steps:
            other = step.after if attempts_used == 0 else None
            wait_for(step.call.key, None if other in self.answered else other)
        # Calls whose writer is not part of this run (e.g. answered by an
        # earlier, interrupted run: its cache entry may have expired). They may
        # share a prefix among them, so all wait for the first of them.
        position = {key: i for i, key in enumerate(keys)}
        replacements = sorted(
            (replace(writer) for writer in [w for w in waiting if w not in by_key]),
            key=position.get,
        )
        for key in replacements[1:]:
            wait_for(key, replacements[0])

        window = 2 * self.policy.max_concurrency
        started: set[str] = set()
        finished: set[str] = set()
        running: set[asyncio.Task] = set()
        wake = asyncio.Event()
        first_unstarted = first_unfinished = 0

        async def run(step: Step) -> None:
            key = step.call.key
            try:
                if not await self._send_live(step, attempts_used):
                    replace(key)  # no answer: its cache entry is not written
            finally:
                finished.add(key)
                wake.set()

        try:
            while True:
                while (
                    first_unfinished < len(keys) and keys[first_unfinished] in finished
                ):
                    first_unfinished += 1
                while first_unstarted < len(keys) and keys[first_unstarted] in started:
                    first_unstarted += 1
                limit = min(len(keys), first_unfinished + window)
                for i in range(first_unstarted, limit):
                    if len(running) >= self.policy.max_concurrency or self.stopped:
                        break
                    key = keys[i]
                    if key in started:
                        continue
                    if after[key] is None or after[key] in self.answered:
                        started.add(key)
                        running.add(asyncio.create_task(run(by_key[key])))
                if not running:
                    break  # all done, or stopped
                wake.clear()
                await wake.wait()
                for task in [t for t in running if t.done()]:
                    running.discard(task)
                    task.result()  # raises unexpected errors
        finally:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)

    async def _send_live(self, step: Step, attempts_used: int = 0) -> bool:
        """Send a call on the live lane until it has a usable answer (at most
        `1 + policy.retries` attempts in total). Returns whether it got any
        answer (usable or not)."""
        key, accept = step.call.key, self.policy.accept
        attempts = 1 + self.policy.retries
        answered = False
        reason = self.gave_up.get(key, "no attempts left")
        for attempt in range(attempts_used, attempts):
            if self.stopped or self._over_budget(step.estimated_usd):
                reason = self.stopped
                break
            self._log(key, "sent", lane=LIVE, attempt=attempt)
            self.sent_live += 1
            self.estimated += step.estimated_usd
            self.reserved += step.estimated_usd
            retryable = True
            try:
                body = await self.transport.send(self._request(step))
                result = self._record(step, body, LIVE)
            except AccountError as e:
                self._log(key, "failed", lane=LIVE, error=str(e))
                if self.account_error is None:
                    self.account_error = e
                    self.stopped = f"API key or account rejected: {e}"
                    logger.error(f"Router: {self.stopped} Sending nothing more.")
                reason = self.stopped
                break
            except (TransportError, KeyError, IndexError, TypeError) as e:
                # (unreadable answers are treated like failed calls)
                reason = f"{type(e).__name__}: {e}"
                self._log(key, "failed", lane=LIVE, error=reason)
                retryable = getattr(e, "retryable", True)
            else:
                answered = True
                if accept is None or accept(result):
                    self.gave_up.pop(key, None)
                    return True
                reason = "answer rejected"
            finally:
                self.reserved -= step.estimated_usd
            if not retryable:
                break
            if not answered and attempt + 1 < attempts:
                # Exponential backoff with full jitter
                wait = random.uniform(0, min(2**attempt, MAX_RETRY_WAIT))
                await asyncio.sleep(wait)
        self._give_up(key, reason)
        return answered

    # --- Batch lane -----------------------------------------------------------

    async def _submit(self, steps: list[Step], reads_prefix: bool = False) -> None:
        """Submit steps as batches (one or more per model).

        Args:
            reads_prefix: Whether the breakpoints of the steps are on prefixes
                written by an earlier stage (their requests are then counted
                for the cache hit rate observed in batches).
        """
        by_model: dict[str, list[Step]] = {}
        for step in steps:
            by_model.setdefault(step.call.model, []).append(step)
        for model, model_steps in by_model.items():
            profile = get_profile(self.router.profiles, model)
            if profile.batch_model is None:
                raise ValueError(f"{model} has no batch variant.")
            for i in range(0, len(model_steps), MAX_REQUESTS_PER_BATCH):
                chunk = model_steps[i : i + MAX_REQUESTS_PER_BATCH]
                cost = sum(step.estimated_usd for step in chunk)
                if self.stopped or self.batch_trouble or self._over_budget(cost):
                    return
                await self._submit_batch(model, chunk, cost, reads_prefix)

    async def _submit_batch(
        self, model: str, steps: list[Step], cost: float, reads_prefix: bool
    ) -> None:
        keys = [step.call.key for step in steps]
        # Breakpoints are kept with the batch: its results may be stored by a
        # later run, with another plan
        entry = {
            "model": model,
            "custom_ids": keys,
            "breakpoints": {
                step.call.key: list(step.breakpoints)
                for step in steps
                if step.breakpoints
            },
            "reads_prefix": reads_prefix,
            "estimated_usd": cost,
        }
        self.batches.append(entry)

        def discard():
            self.batches.remove(entry)
            self._save_batches()

        requests = [self._request(step) for step in steps]
        try:
            batch_id = await self.transport.submit(
                model, requests, entry, self._save_batches, discard
            )
        except TransportError as e:
            if entry not in self.batches:
                # Definitely not created
                for key in keys:
                    self._give_up(key, f"batch submission rejected: {e}")
            else:
                self.batch_trouble = True
                self.unconfirmed.add(id(entry))
                self.problems.append(
                    f"Submitting a batch for {model} failed ({e}): not known "
                    "whether it was created. Run again to find out; its calls "
                    "are not sent again until then."
                )
            if isinstance(e, AccountError):
                self.account_error = self.account_error or e
                self.stopped = f"API key or account rejected: {e}"
            return
        except BaseException:
            if "submitted_at" not in entry:
                self.batches.remove(entry)  # nothing was sent
            raise
        logger.info(f"Submitted batch {batch_id} with {len(keys)} requests ({model}).")
        self._submitted(entry)

    def _submitted(self, entry: dict) -> None:
        """Account for a batch that this run submitted."""
        for key in entry["custom_ids"]:
            self._log(key, "submitted", lane=BATCH, batch_id=entry["batch_id"])
        self.sent_batch += len(entry["custom_ids"])
        self.estimated += entry["estimated_usd"]
        self.reserved += entry["estimated_usd"]
        self.reserved_batches.add(entry["batch_id"])

    async def _poll_batches(self) -> list[Step]:
        """Check all batches once and store the results of those that ended.

        Returns:
            Steps to repeat on the live lane: requests that failed within a
            completed batch, and answers that were rejected.
        """
        retry: list[Step] = []
        resubmit: list[str] = []
        for entry in list(self.batches):
            if not entry.get("batch_id"):
                # The submission was interrupted before the batch id was saved
                resolved, batch_id = await self.transport.recover(
                    entry["model"],
                    entry,
                    [e["batch_id"] for e in self.batches if e.get("batch_id")],
                )
                if batch_id:
                    entry["batch_id"] = batch_id
                    logger.info(f"Recovered batch {batch_id} (submission interrupted).")
                    if id(entry) in self.unconfirmed:
                        self._submitted(entry)
                elif resolved:
                    self.batches.remove(entry)
                    resubmit += entry["custom_ids"]
                    logger.info(
                        "A batch submission was interrupted before the batch was "
                        "created: submitting its requests again."
                    )
                self._save_batches()
                if not batch_id:
                    continue
            try:
                batch, status = await self.transport.poll(entry)
            except BatchNotFound as e:
                # Kept: it may be running (and billed) nonetheless
                self.problems.append(
                    f"{e}. Its calls stay pending. To abandon the batch (it may "
                    f"still be billed), remove it from {self.batches_file}."
                )
                continue
            if batch is None:
                continue
            retry += self._save_batch_results(batch, status, entry)
            # Only removed once its results are stored
            self.batches.remove(entry)
            self._save_batches()
            if entry["batch_id"] in self.reserved_batches:
                self.reserved -= entry["estimated_usd"]

        policy = self.policy
        steps = [
            self.steps.get(key) or Step(self.queue[key], BATCH)
            for key in resubmit
            if key in self.queue and not self.router._answered(self.queue[key], policy)
        ]
        await self._submit(steps)
        return retry

    def _save_batch_results(self, batch: dict, status: str, entry: dict) -> list[Step]:
        """Store the results of a batch that ended (also the partial results of
        a batch that did not complete). Returns the steps to repeat live."""
        batch_id = batch.get("id") or entry["batch_id"]
        completed = status == "completed"
        keys = set(entry["custom_ids"])
        breakpoints = entry.get("breakpoints") or {}
        retry, unmatched, seen = [], [], set()
        num_stored = num_failed = 0

        def step_for(key: str) -> Step:
            step = self.steps.get(key)
            if step is None:
                step = Step(self.queue[key], BATCH, tuple(breakpoints.get(key, ())))
            return step

        for result in batch.get("results") or []:
            key = result.get("custom_id") if isinstance(result, dict) else None
            if key not in keys or key not in self.queue:
                # Not a queued call of this batch (e.g. the queue changed)
                unmatched.append(result)
                continue
            if key in seen:
                continue
            seen.add(key)
            record = self.store.record(key)
            if record is not None and record.get("batch_id") == batch_id:
                continue  # stored by a run interrupted while storing the results
            step = step_for(key)
            try:
                body = get_result_body(result)
                if body is not None:
                    answer = self._record(
                        step, body, BATCH, batch_id, entry.get("reads_prefix", False)
                    )
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                logger.warning(f"Could not read a result of batch {batch_id}: {e!r}")
                unmatched.append(result)
                body = None
            if body is None:
                num_failed += 1
                self._log(key, "failed", lane=BATCH, batch_id=batch_id)
                self._give_up(key, f"request failed in batch {batch_id}")
                retry.append(step)
                continue
            num_stored += 1
            if self.policy.accept is not None and not self.policy.accept(answer):
                self._give_up(key, "answer rejected")
                retry.append(step)

        missing = [
            key
            for key in entry["custom_ids"]
            if key not in seen and key in self.queue and key not in self.store
        ]
        for key in missing:
            self._log(key, "failed", lane=BATCH, batch_id=batch_id)
            self._give_up(key, f"no result (batch {batch_id} {status})")
            if completed:
                retry.append(step_for(key))
        if unmatched:
            path = self.state_dir / "unmatched" / f"{batch_id}.json"
            write_json_atomic(path, {"model": entry["model"], "results": unmatched})
            self.problems.append(
                f"{len(unmatched)} results of batch {batch_id} could not be read or "
                f"match no queued call: saved as received to {path}."
            )
        message = (
            f"Batch {batch_id} ({entry['model']}) {status}: stored {num_stored} "
            f"answers, {num_failed} failed requests, {len(missing)} requests "
            f"without result. Cost: {(batch.get('usage') or {}).get('cost')}"
        )
        if completed:
            logger.info(message)
            return retry
        # Not repeated at the full price of the live lane: reported instead
        logger.error(message)
        self.problems.append(
            f"Batch {batch_id} for {entry['model']} did not complete ({status}: "
            f"{batch.get('error')}). Answers received are stored. Run again to "
            "send the remaining calls."
        )
        return [step for step in retry if step.call.key in self.store]


def _cost_from_tokens(usage: Usage, profile, route: str, ttl: str) -> float:
    """Cost of a call from its tokens and the profile prices (used where the
    provider did not report the cost)."""
    cached = usage.cached_tokens or 0
    written = usage.cache_write_tokens or 0
    plain = (usage.input_tokens or 0) - cached - written
    price = profile.batch_price_multiplier if route == BATCH else 1.0
    return (
        price
        * (
            profile.input_price
            * (
                plain
                + cached * profile.cache_read_multiplier
                + written * profile.cache_write_multiplier(ttl)
            )
            + profile.output_price * (usage.output_tokens or 0)
        )
        / 1e6
    )

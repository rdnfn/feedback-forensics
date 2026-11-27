"""Tests of the router on a fake provider (no network): queue, run, collect,
ordering on the live lane, budget, retries, and resuming on both lanes."""

import json
import random
from dataclasses import replace
from datetime import timedelta

import pytest

from feedback_forensics.router.datatypes import Policy, Result, Stage
from feedback_forensics.router.fakes import SimulatedCrash, run_virtual
from feedback_forensics.router.router import Router, RouterBusyError
from feedback_forensics.router.strategies import STRATEGIES, make_plan, strategy
from feedback_forensics.router.testutils import (
    FIXED_END,
    MIDDLE_END,
    MODEL,
    PROFILE,
    PROFILES,
    make_call,
    make_calls,
    make_router,
)
from feedback_forensics.router.transports import AccountError, TransportError

HOUR = 3600
LATER = dict(deadline=timedelta(hours=30))


def run(router, start=0.0, **policy):
    """Run the router with virtual time (starting at `start` seconds)."""
    return run_virtual(router.arun(Policy(**policy)), start=start)


def stored_lines(path) -> list[dict]:
    """All lines of the call files of a store."""
    files = [
        f
        for f in path.rglob("*.jsonl")
        if f.relative_to(path).parts[0] not in ("pieces", ".router")
    ]
    return [json.loads(line) for f in files for line in f.read_text().splitlines()]


# --- Queue, run, collect ------------------------------------------------------


def test_queue_run_collect(tmp_path):
    router, fake = make_router(tmp_path)
    calls = make_calls(groups=3, per_group=4)
    keys = router.add(calls, task="annotation", group="openai/gpt-x")
    assert keys == [call.key for call in calls]
    assert fake.sent == []  # nothing is sent before `run`

    report = run(router)
    assert (report.calls, report.sent, report.sent_live, report.reused) == (
        12,
        12,
        12,
        0,
    )
    assert report.complete and report.strategy == "live_ordered"
    assert report.gave_up == () and report.pending == () and report.problems == ()
    assert report.cost_usd == pytest.approx(
        sum(call["usage"]["cost"] for call in fake.calls)
    )
    assert 0 < report.cost_usd < report.estimated_usd  # (answers are short)
    assert report.input_tokens == sum(len(call.text) for call in calls)

    results = router.collect(keys)
    assert list(results) == keys
    for call in calls:
        result = results[call.key]
        assert result.ok and result.text == f"answer {call.key[:8]}"
        assert result.tag is call.tag  # returned untouched
        assert (result.route, result.finish_reason) == ("live", "stop")
        assert result.usage.input_tokens == len(call.text)
    # Stored in the folder the caller named
    file = tmp_path / "annotation" / MODEL.replace("/", "_") / "openai_gpt-x.jsonl"
    assert len(file.read_text().splitlines()) == 12


def test_duplicate_calls_are_answered_once(tmp_path):
    router, fake = make_router(tmp_path)
    calls = make_calls(groups=2, per_group=2)
    copies = [replace(call, tag="copy") for call in calls]
    keys = router.add(calls + calls[:1])
    assert router.add(copies, task="elsewhere") == keys[:4]
    report = run(router)
    assert (report.calls, report.sent) == (4, 4)
    assert sorted(fake.sent) == sorted(set(keys))
    assert len(stored_lines(tmp_path)) == 4
    # Both callers get the result, with the tag of the call queued first
    results = router.collect(keys)
    assert len(results) == 4
    assert results[calls[0].key].tag is calls[0].tag


def test_stored_results_are_reused_across_runs(tmp_path):
    calls = make_calls(groups=2, per_group=3)
    router, fake = make_router(tmp_path)
    keys = router.add(calls)
    first = run(router)
    assert (first.sent, first.reused) == (6, 0)
    assert run(router).sent == 0  # running again sends nothing

    # A new process: only the new call is sent
    router, fake = make_router(tmp_path)
    new_call = make_call(5, 0)
    router.add(calls + [new_call])
    report = run(router)
    assert (report.calls, report.reused, report.sent) == (7, 6, 1)
    assert fake.sent == [new_call.key]
    results = router.collect(keys + [new_call.key])
    assert {results[key].route for key in keys} == {"store"}
    assert results[new_call.key].route == "live"
    assert results[keys[0]].text == f"answer {keys[0][:8]}"
    assert results[keys[0]].tag is calls[0].tag
    assert len(stored_lines(tmp_path)) == 7


def test_collect_without_answer(tmp_path):
    router, _ = make_router(tmp_path)
    (key,) = router.add(make_calls(1, 1))
    result = router.collect([key, "unknown"])
    assert not result[key].ok and "not run" in result[key].error
    assert result[key].text is None and result[key].route is None
    assert not result["unknown"].ok


def test_dry_run_sends_nothing(tmp_path):
    router, fake = make_router(tmp_path / "calls", strategy="auto")
    router.add(make_calls(groups=2, per_group=3))
    plan, layout = router.dry_run(Policy())
    assert layout is None
    assert plan.strategy == "live_ordered" and len(plan.steps) == 6
    assert plan.estimate.planned_usd < plan.estimate.as_given_usd
    plan, layout = router.dry_run(Policy(**LATER), check_layout=True)
    assert layout is not None and layout.findings == ()
    assert fake.sent == [] and fake.submissions == []
    assert not (tmp_path / "calls" / ".router").exists()
    # Calls with a stored answer are not planned
    run(router)
    router.add([make_call(4, 0)])
    assert len(router.dry_run()[0].steps) == 1


def test_only_one_run_per_store(tmp_path):
    router, _ = make_router(tmp_path)
    router.add(make_calls(1, 2))
    other, fake = make_router(tmp_path)
    other.add(make_calls(1, 2))

    async def run_both():
        from filelock import FileLock

        (tmp_path / ".router").mkdir()
        with FileLock(tmp_path / ".router" / "lock"):
            await other.arun()

    with pytest.raises(RouterBusyError):
        run_virtual(run_both())
    assert fake.sent == []


# --- Ordering on the live lane ------------------------------------------------


def test_no_call_starts_before_its_prefix_is_written(tmp_path):
    router, fake = make_router(tmp_path, latency=7.0)
    calls = make_calls(groups=5, per_group=6)
    random.Random(0).shuffle(calls)
    router.add(calls)
    plan, _ = router.dry_run(Policy(max_concurrency=4))
    report = run(router, max_concurrency=4)
    assert report.sent == 30

    times = {call["key"]: call for call in fake.calls}
    waiting = [step for step in plan.steps if step.after]
    assert len(waiting) == 29
    for step in waiting:
        assert times[step.call.key]["start"] >= times[step.after]["end"]
    # Hence every prefix is written exactly once, by one call, and read by all
    # other calls that share it: the fixed start and five middle parts
    assert report.cache_write_tokens == MIDDLE_END + 4 * (MIDDLE_END - FIXED_END)
    assert report.cached_tokens == 25 * MIDDLE_END + 4 * FIXED_END
    assert report.cached_share == report.cached_tokens / report.input_tokens
    # The concurrency cap holds
    events = sorted(
        [(c["start"], 1) for c in fake.calls] + [(c["end"], -1) for c in fake.calls],
        key=lambda e: (e[0], e[1]),
    )
    in_flight = peak = 0
    for _, change in events:
        in_flight += change
        peak = max(peak, in_flight)
    assert peak == 4
    # Calls that share a prefix run close together: one group after another
    order = sorted(fake.calls, key=lambda c: c["start"])
    groups = [router.collect([c["key"]])[c["key"]].tag["group"] for c in order]
    assert sum(a != b for a, b in zip(groups, groups[1:])) <= 2 * 4


@pytest.fixture
def arbitrary_order():
    """Strategy: the calls and breakpoints of `live_ordered`, sent in arbitrary
    order, without waiting for the calls that write the prefixes."""

    @strategy("arbitrary_order")
    def arbitrary(tree, profiles, policy):
        plan = STRATEGIES["live_ordered"](tree, profiles, policy)
        steps = [replace(step, after=None) for step in plan.steps]
        random.Random(0).shuffle(steps)
        return replace(plan, strategy="arbitrary_order", stages=(Stage(tuple(steps)),))

    yield "arbitrary_order"
    del STRATEGIES["arbitrary_order"]


def test_cached_share_of_ordered_against_arbitrary_order(tmp_path, arbitrary_order):
    """On a provider cache (entries are written when a call finishes, expire
    5 minutes after their last use, and are read only by later calls),
    `live_ordered` reads every shared prefix from the cache but once. The same
    calls with the same breakpoints in arbitrary order read much less, and
    write the same prefixes again and again."""
    # 40 groups of 6 calls. As in the annotation case, the part a group shares
    # (1,200 tokens) is longer than the fixed start (624) and the unique end
    calls = make_calls(groups=40, per_group=6, middle_lines=40)
    middle_end = FIXED_END + 40 * 30
    reports = {}
    for name in ("live_ordered", arbitrary_order):
        router, fake = make_router(tmp_path / name, strategy=name, latency=30.0)
        router.add(calls)
        reports[name] = run(router, max_concurrency=4)
        assert reports[name].sent == 240 and reports[name].complete
    ordered, arbitrary = reports["live_ordered"], reports[arbitrary_order]
    assert ordered.input_tokens == arbitrary.input_tokens

    # Ordered: the most that can be read (all but the first call of each group
    # read the middle part too, the first of each group but one the fixed
    # start), and every prefix is written once
    best = 200 * middle_end + 39 * FIXED_END
    assert ordered.cached_tokens == best
    assert ordered.cache_write_tokens == middle_end + 39 * (middle_end - FIXED_END)
    assert ordered.cached_share == pytest.approx(0.83, abs=0.005)
    # Arbitrary order: calls of a group are minutes apart (their middle part
    # has expired) or in flight together (each writes it)
    assert arbitrary.cached_share == pytest.approx(0.63, abs=0.02)
    assert arbitrary.cache_write_tokens > 2.5 * ordered.cache_write_tokens
    assert ordered.cost_usd < 0.6 * arbitrary.cost_usd
    # The plan's estimate of the cached tokens is what the provider reports
    plan = make_plan(calls, PROFILES, Policy(max_concurrency=4), "live_ordered")
    assert plan.estimate.cached_tokens == best


def test_first_call_is_replaced_if_it_fails(tmp_path):
    """If the call that writes a prefix fails, one of the calls waiting for it
    takes its place (they do not all write the prefix)."""
    router, fake = make_router(tmp_path)
    calls = make_calls(groups=1, per_group=5)
    router.add(calls)
    plan, _ = router.dry_run()
    first = plan.steps[0].call.key
    fake.live_errors[first] = [TransportError("bad request", 400)]
    report = run(router)
    assert report.gave_up == ((first, "TransportError: bad request"),)
    assert report.sent == 5 and len(fake.calls) == 4
    assert report.cache_write_tokens == MIDDLE_END  # written once
    assert report.cached_tokens == 3 * MIDDLE_END
    result = router.collect([first])[first]
    assert not result.ok and "bad request" in result.error


# --- Budget and retries -------------------------------------------------------


def test_budget_stop(tmp_path):
    calls = make_calls(groups=2, per_group=5)
    router, fake = make_router(tmp_path)
    keys = router.add(calls)
    plan, _ = router.dry_run()
    budget = sum(step.estimated_usd for step in plan.steps[:4]) * 1.001
    report = run(router, budget_usd=budget, max_concurrency=2)
    assert 1 <= report.sent < 10 and not report.complete
    assert report.cost_usd <= budget
    assert len(report.gave_up) == 10 - report.sent
    assert {reason for _, reason in report.gave_up} == {f"budget of ${budget} reached"}
    results = router.collect(keys)
    assert sum(r.ok for r in results.values()) == report.sent
    assert all("budget" in r.error for r in results.values() if not r.ok)

    # Running again without the limit sends the remaining calls only
    router, fake = make_router(tmp_path)
    router.add(calls)
    second = run(router)
    assert second.complete and second.sent == 10 - report.sent
    assert second.reused == report.sent


def test_budget_stop_for_batches(tmp_path):
    router, fake = make_router(tmp_path, strategy="batch")
    router.add(make_calls(groups=2, per_group=3))
    report = run(router, budget_usd=1e-9, **LATER)
    assert fake.submissions == [] and report.sent == 0
    assert len(report.gave_up) == 6 and "budget" in report.gave_up[0][1]


def accept(result: Result) -> bool:
    return "bad" not in result.text


def test_rejected_answers_are_retried(tmp_path):
    calls = make_calls(groups=1, per_group=3)
    flaky, hopeless, fine = (call.key for call in calls)
    attempts: dict[str, int] = {}

    def answer(request):
        attempts[request.key] = attempts.get(request.key, 0) + 1
        if request.key == hopeless:
            return f"bad answer {attempts[request.key]}"
        if request.key == flaky and attempts[request.key] == 1:
            return "bad answer"
        return "good answer"

    router, fake = make_router(tmp_path, answer=answer)
    keys = router.add(calls)
    report = run(router, accept=accept, retries=2)
    # The rejected answer is asked again, on the live lane; three attempts each
    assert attempts == {flaky: 2, hopeless: 3, fine: 1}
    assert report.sent_live == 6
    assert report.gave_up == ((hopeless, "answer rejected"),)
    assert not report.complete
    results = router.collect(keys)
    assert results[flaky].text == results[fine].text == "good answer"
    # Still bad: reported, and its last answer is kept (it was paid for)
    assert results[hopeless].text == "bad answer 3"
    assert report.cost_usd == pytest.approx(sum(c["usage"]["cost"] for c in fake.calls))

    # Running again: the stored bad answer is not taken, the call is retried
    router, fake = make_router(tmp_path, answer=lambda request: "good at last")
    router.add(calls)
    report = run(router, accept=accept)
    assert (report.reused, report.sent) == (2, 1) and report.complete
    assert fake.sent == [hopeless]
    assert router.collect([hopeless])[hopeless].text == "good at last"
    # Without the caller's check, any stored answer is taken
    router, fake = make_router(tmp_path / "other", answer=lambda request: "bad")
    router.add(calls)
    assert len(run(router, accept=accept, retries=0).gave_up) == 3
    assert run(router).reused == 3 and len(fake.sent) == 3


def test_failed_calls_are_retried(tmp_path):
    calls = make_calls(groups=1, per_group=3)
    flaky, broken, invalid = (call.key for call in calls)
    router, fake = make_router(tmp_path)
    fake.live_errors = {
        flaky: [TransportError("overloaded", 529, retryable=True)] * 2,
        broken: [TransportError("overloaded", 529, retryable=True)] * 5,
        # Not worth repeating
        invalid: [TransportError("invalid request", 400)] * 5,
    }
    router.add(calls)
    report = run(router, retries=2)
    assert fake.sent.count(flaky) == 3
    assert fake.sent.count(broken) == 3
    assert fake.sent.count(invalid) == 1
    assert dict(report.gave_up) == {
        broken: "TransportError: overloaded",
        invalid: "TransportError: invalid request",
    }
    assert router.collect([flaky])[flaky].ok
    assert report.sent_live == 7
    # The next run tries the failed calls again
    fake.live_errors = {}
    assert run(router).complete and len(fake.sent) == 9


def test_account_error_stops_the_run(tmp_path):
    calls = make_calls(groups=1, per_group=6)
    router, fake = make_router(tmp_path)
    keys = router.add(calls)
    plan, _ = router.dry_run()
    third = plan.steps[2].call.key
    fake.live_errors[third] = [AccountError("no credits left", 402)]
    with pytest.raises(AccountError, match="no credits"):
        run(router, max_concurrency=1)
    # No further calls were sent; answers received before are stored
    assert len(fake.sent) == 3
    results = router.collect(keys)
    assert sum(r.ok for r in results.values()) == 2
    assert all("account" in r.error for r in results.values() if not r.ok)
    assert run(router).complete and len(fake.sent) == 3 + 4


# --- Resuming -----------------------------------------------------------------


def test_resume_after_interruption_on_the_live_lane(tmp_path):
    calls = make_calls(groups=3, per_group=5)
    router, fake = make_router(tmp_path)
    router.add(calls)
    fake.crash_after = 8
    with pytest.raises(SimulatedCrash):
        run(router, max_concurrency=3)
    answered = {line["key"] for line in stored_lines(tmp_path)}
    assert answered == {call["key"] for call in fake.calls} and 1 <= len(answered) <= 8
    in_flight = set(fake.sent) - answered

    # The same script again: answered calls are not sent a second time
    router, fake = make_router(tmp_path)
    keys = router.add(calls)
    report = run(router, max_concurrency=3)
    assert report.complete and report.reused == len(answered)
    assert not set(fake.sent) & answered
    assert len(fake.sent) == len(set(fake.sent)) == 15 - len(answered)
    # Calls in flight at the interruption have no answer: sent again, reported
    assert report.interrupted == len(in_flight) >= 1
    assert all(r.ok for r in router.collect(keys).values())
    assert len(stored_lines(tmp_path)) == 15
    # The prefixes are written again (once): those of the first run have expired
    assert fake.calls[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert report.cache_write_tokens <= MIDDLE_END + 2 * (MIDDLE_END - FIXED_END)
    assert report.cache_write_tokens >= MIDDLE_END


def test_batches(tmp_path):
    router, fake = make_router(tmp_path, strategy="batch")
    calls = make_calls(groups=2, per_group=3)
    keys = router.add(calls, task="generation")
    report = run(router, **LATER)
    assert (report.sent_batch, report.sent_live, report.complete) == (6, 0, True)
    assert report.strategy == "batch"
    assert sorted(fake.submissions[0]) == sorted(keys) and fake.sent == []
    results = router.collect(keys)
    assert {r.route for r in results.values()} == {"batch"}
    assert all(r.text == f"answer {key[:8]}" for key, r in results.items())
    assert {line["batch_id"] for line in stored_lines(tmp_path)} == {"batch_1"}
    # At half the price, without caching
    assert report.cached_tokens == 0
    assert report.cost_usd == pytest.approx(report.estimated_usd, rel=0.2)
    assert not (tmp_path / ".router" / "batches.json").exists()
    assert not (tmp_path / ".router" / "plan.json").exists()


def test_pending_batch_is_collected_by_a_later_run(tmp_path):
    """A batch still pending at the deadline: running again collects it,
    without submitting anything again."""
    calls = make_calls(groups=2, per_group=3)
    router, fake = make_router(tmp_path, strategy="batch", batch_delay=40 * HOUR)
    keys = router.add(calls)
    report = run(router, **LATER)
    assert report.sent_batch == 6 and not report.complete
    assert report.pending == tuple(sorted(keys)) and report.gave_up == ()
    assert "pending" in router.collect(keys)[keys[0]].error
    assert fake.polls > 100  # it waited until the deadline

    # Later, a new process with the same provider
    router = Router(tmp_path, strategy="batch", profiles=PROFILES, transport=fake)
    router.add(calls)
    still = run(router, start=31 * HOUR, deadline=timedelta(hours=1))
    assert still.pending == report.pending and still.sent == 0
    # Results are needed now: the calls of the pending batch are not sent live
    now = run(router, start=32 * HOUR, deadline=None)
    assert now.pending == report.pending and now.sent == 0 and fake.sent == []

    done = run(router, start=41 * HOUR, **LATER)
    assert done.complete and done.sent == 0 and len(fake.submissions) == 1
    assert done.cost_usd > 0
    assert all(r.ok and r.route == "batch" for r in router.collect(keys).values())
    assert len(stored_lines(tmp_path)) == 6
    assert run(router, start=42 * HOUR, **LATER).reused == 6


@pytest.mark.parametrize("created", [True, False])
def test_interrupted_batch_submission(tmp_path, created):
    """The response to a submission is lost: the batch is found again if it was
    created, and submitted again only if it was not."""
    calls = make_calls(groups=2, per_group=3)
    router, fake = make_router(tmp_path, strategy="batch")
    keys = router.add(calls)
    fake.lose_submit_response = True if created else "not_created"
    report = run(router, **LATER)
    assert report.sent == 0 and not report.complete
    assert "not known whether it was created" in report.problems[0]
    assert len(fake.submissions) == (1 if created else 0)
    state = json.loads((tmp_path / ".router" / "batches.json").read_text())
    assert state["batches"][0]["batch_id"] is None
    assert sorted(state["batches"][0]["custom_ids"]) == sorted(keys)
    # Its calls count as pending: they are not sent again in the meantime
    assert report.pending == tuple(sorted(keys))

    fake.lose_submit_response = False
    router, _ = make_router(tmp_path, strategy="batch")
    router.transport = fake
    router.add(calls)
    report = run(router, start=2 * HOUR, **LATER)
    assert report.complete and len(fake.submissions) == 1
    assert report.sent_batch == (0 if created else 6)
    assert len(stored_lines(tmp_path)) == 6
    states = (tmp_path / ".router" / "state.jsonl").read_text()
    assert states.count('"submitted"') == (0 if created else 6)


def test_rejected_batch_submission(tmp_path):
    router, fake = make_router(tmp_path, strategy="batch")
    keys = router.add(make_calls(groups=1, per_group=3))
    fake.reject_submit = True
    report = run(router, **LATER)
    assert len(report.gave_up) == 3 and "rejected" in report.gave_up[0][1]
    assert not (tmp_path / ".router" / "batches.json").exists()
    fake.reject_submit = False
    assert run(router, **LATER).complete and len(fake.submissions) == 1


def test_requests_failed_in_a_batch_are_repeated_live(tmp_path):
    calls = make_calls(groups=1, per_group=4)
    router, fake = make_router(tmp_path, strategy="batch")
    keys = router.add(calls)
    fake.failed_in_batch = {keys[1]}
    # A result that cannot be read, and one of an unknown request
    fake.extra_results = [
        {"custom_id": keys[2], "response": {"status_code": 200, "body": {}}},
        {"custom_id": "unknown", "response": {"status_code": 200, "body": {}}},
    ]
    report = run(router, **LATER)
    assert report.complete and report.sent_batch == 4
    assert sorted(fake.sent) == sorted([keys[1]])
    routes = {key: r.route for key, r in router.collect(keys).items()}
    assert routes == {
        keys[0]: "batch",
        keys[1]: "live",
        keys[2]: "batch",
        keys[3]: "batch",
    }
    # Kept as received
    unmatched = json.loads(
        (tmp_path / ".router" / "unmatched" / "batch_1.json").read_text()
    )
    assert [r["custom_id"] for r in unmatched["results"]] == ["unknown"]
    assert "match no queued call" in report.problems[0]


def test_unreadable_batch_result_is_kept_and_repeated(tmp_path):
    calls = make_calls(groups=1, per_group=2)
    router, fake = make_router(tmp_path, strategy="batch")
    keys = router.add(calls)
    fake.failed_in_batch = {keys[0]}
    fake.partial_results = 0
    fake.extra_results = [
        {
            "custom_id": keys[0],
            "response": {"status_code": 200, "body": {"choices": []}},
        },
        "not even an object",
    ]
    report = run(router, **LATER)
    assert sorted(fake.sent) == sorted(keys) and report.complete
    unmatched = json.loads(
        (tmp_path / ".router" / "unmatched" / "batch_1.json").read_text()
    )
    assert len(unmatched["results"]) == 2


def test_incomplete_batch_keeps_partial_results(tmp_path):
    """A batch that expired: its results are stored, the remaining calls are
    reported, not silently sent at the full price of the live lane."""
    calls = make_calls(groups=1, per_group=5)
    router, fake = make_router(tmp_path, strategy="batch")
    keys = router.add(calls)
    fake.batch_status, fake.partial_results = "expired", 2
    report = run(router, **LATER)
    assert fake.sent == [] and not report.complete
    assert len(report.gave_up) == 3 and report.pending == ()
    assert "no result (batch batch_1 expired)" in report.gave_up[0][1]
    assert "did not complete" in report.problems[0]
    assert sum(r.ok for r in router.collect(keys).values()) == 2
    # Running again submits the remaining calls
    fake.batch_status, fake.partial_results = "completed", None
    report = run(router, start=2 * HOUR, **LATER)
    assert report.complete and report.reused == 2
    assert [len(s) for s in fake.submissions] == [5, 3]


def test_batch_unknown_to_the_provider_is_kept(tmp_path):
    calls = make_calls(groups=1, per_group=3)
    router, fake = make_router(tmp_path, strategy="batch", batch_delay=40 * HOUR)
    keys = router.add(calls)
    run(router, **LATER)
    fake.unknown_batches = True
    report = run(router, start=41 * HOUR, **LATER)
    # It may be running nonetheless: kept, and its calls are not sent again
    assert report.pending == tuple(sorted(keys)) and report.sent == 0
    assert "not found" in report.problems[0] and "abandon" in report.problems[0]
    assert (tmp_path / ".router" / "batches.json").exists()
    fake.unknown_batches = False
    assert run(router, start=42 * HOUR, **LATER).complete
    assert len(fake.submissions) == 1


def test_batch_results_are_stored_once_if_storing_is_interrupted(tmp_path, monkeypatch):
    calls = make_calls(groups=1, per_group=5)
    router, fake = make_router(tmp_path, strategy="batch", batch_delay=40 * HOUR)
    router.add(calls)
    run(router, **LATER)

    put, count = router.store.put, 0

    def interrupted_put(*args, **kwargs):
        nonlocal count
        count += 1
        if count > 2:
            raise SimulatedCrash()
        return put(*args, **kwargs)

    monkeypatch.setattr(router.store, "put", interrupted_put)
    with pytest.raises(SimulatedCrash):
        run(router, start=41 * HOUR, **LATER)
    assert len(stored_lines(tmp_path)) == 2

    router = Router(tmp_path, strategy="batch", profiles=PROFILES, transport=fake)
    router.add(calls)
    report = run(router, start=42 * HOUR, **LATER)
    assert report.complete and len(fake.submissions) == 1 and fake.sent == []
    lines = stored_lines(tmp_path)
    assert len(lines) == 5 and len({line["key"] for line in lines}) == 5


def test_rejected_batch_answers_are_retried_live(tmp_path):
    calls = make_calls(groups=1, per_group=3)
    seen = set()

    def answer(request):
        first = request.key not in seen
        seen.add(request.key)
        return "bad" if first and request.key == calls[0].key else "good"

    router, fake = make_router(tmp_path, strategy="batch", answer=answer)
    keys = router.add(calls)
    report = run(router, accept=accept, **LATER)
    assert report.complete and fake.sent == [keys[0]]
    assert {r.text for r in router.collect(keys).values()} == {"good"}


# --- Strategies with stages ---------------------------------------------------


def cached(fake, lane) -> list[int]:
    return [
        c["usage"]["prompt_tokens_details"]["cached_tokens"]
        for c in fake.calls
        if c["lane"] == lane
    ]


def test_warm_then_batch(tmp_path):
    """Stage 1: one live call writes the fixed prefix (1-hour lifetime).
    Stage 2: batches that read it. The hit rate is recorded in the profile."""
    calls = make_calls(groups=3, per_group=4)
    router, fake = make_router(tmp_path, strategy="warm_then_batch")
    keys = router.add(calls)
    report = run(router, **LATER)
    assert report.complete and (report.sent_live, report.sent_batch) == (1, 11)
    assert report.strategy == "warm_then_batch"
    # The live call came first, with a 1-hour breakpoint on the fixed prefix
    (request,) = fake.requests
    blocks = request.body["messages"][1]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert len(request.body["messages"][0]["content"] + blocks[0]["text"]) == FIXED_END
    assert fake.calls[0]["lane"] == "live"
    assert fake.calls[0]["end"] <= min(c["start"] for c in fake.calls[1:])
    assert cached(fake, "batch") == [FIXED_END] * 11
    assert report.cached_tokens == 11 * FIXED_END
    assert report.cost_usd == pytest.approx(sum(c["usage"]["cost"] for c in fake.calls))

    # All batch requests read the prefix: recorded, also for later processes
    assert router.profiles[PROFILE.model].observed_hit_rate == 1.0
    router, _ = make_router(tmp_path / "other")
    assert router.profiles[PROFILE.model].observed_hit_rate is None
    router, _ = make_router(tmp_path)
    assert router.profiles[PROFILE.model].observed_hit_rate == 1.0
    assert router.profiles[PROFILE.model].cache_in_batch is True


def test_observed_hit_rate_guides_auto(tmp_path):
    """If batches turn out not to read a prefix written live, `auto` stops
    planning for it."""
    calls = make_calls(groups=1, per_group=30)
    measured = {PROFILE.model: replace(PROFILE, observed_hit_rate=1.0)}
    router, fake = make_router(
        tmp_path, strategy="auto", profiles=measured, batch_reads_live_cache=False
    )
    router.add(calls[:15])
    policy = Policy(**LATER, max_concurrency=1)
    # (with these mostly shared prompts, warming beats plain batches)
    plan, _ = router.dry_run(policy)
    assert plan.strategy in ("warm_then_batch", "live_ordered")
    router.strategy = "warm_then_batch"
    report = run(router, **LATER)
    assert report.complete and report.cached_tokens == 0
    assert router.profiles[PROFILE.model].observed_hit_rate == 0.0
    assert router.profiles[PROFILE.model].cache_in_batch is False

    router, fake = make_router(tmp_path, strategy="batch_chain", profiles=measured)
    assert router.profiles[PROFILE.model].observed_hit_rate == 0.0
    router.add(calls)
    plan, _ = router.dry_run(Policy(deadline=timedelta(hours=60)))
    assert (
        plan.estimate.planned_usd
        > make_plan(
            calls[15:], router.profiles, Policy(deadline=timedelta(hours=60)), "batch"
        ).estimate.planned_usd
    )


def test_batch_chain_continues_across_runs(tmp_path):
    """Stage 1 (a small batch writes the prefix) is still pending at the
    deadline. Running again continues the same plan: stage 2 follows, and no
    second batch is sent to write the prefix."""
    calls = make_calls(groups=3, per_group=4)
    router, fake = make_router(tmp_path, strategy="batch_chain", batch_delay=20 * HOUR)
    keys = router.add(calls)
    first = run(router, deadline=timedelta(hours=10))
    assert [len(s) for s in fake.submissions] == [1]
    assert len(first.pending) == 1 and len(first.gave_up) == 11
    assert (tmp_path / ".router" / "plan.json").exists()

    router = Router(tmp_path, strategy="batch_chain", profiles=PROFILES, transport=fake)
    router.add(calls)
    second = run(router, start=20.5 * HOUR, deadline=timedelta(hours=10))
    assert [len(s) for s in fake.submissions] == [1, 11]
    assert second.reused == 0 and second.sent_batch == 11
    assert len(second.pending) == 11 and second.gave_up == ()
    # Stage 1 ended at an unknown time before this run: its prefix may have
    # expired (1-hour lifetime). Stage 2 is sent without breakpoints, rather
    # than have every request pay to write the prefix again.
    assert cached(fake, "batch") == [0] * 12

    third = run(router, start=42 * HOUR, deadline=timedelta(hours=10))
    assert third.complete and third.sent == 0
    assert [len(s) for s in fake.submissions] == [1, 11]
    assert all(r.ok for r in router.collect(keys).values())
    assert third.cache_write_tokens == 0
    # No request was sent to read the prefix: no hit rate is recorded
    assert router.profiles[PROFILE.model].observed_hit_rate is None
    assert not (tmp_path / ".router" / "plan.json").exists()
    state = (tmp_path / ".router" / "state.jsonl").read_text().splitlines()
    states = [json.loads(line)["state"] for line in state]
    assert states.count("submitted") == states.count("answered") == 12


def test_batch_chain_in_one_run(tmp_path):
    router, fake = make_router(tmp_path, strategy="batch_chain", batch_delay=HOUR)
    router.add(make_calls(groups=3, per_group=4))
    report = run(router, deadline=timedelta(hours=60))
    assert report.complete and [len(s) for s in fake.submissions] == [1, 11]
    # Stage 2 was submitted once the prefix was written: its requests read it
    assert cached(fake, "batch") == [0] + [FIXED_END] * 11
    assert report.cache_write_tokens == FIXED_END
    assert router.profiles[PROFILE.model].observed_hit_rate == 1.0


def test_calls_of_several_models(tmp_path):
    other = replace(PROFILE, model="x/other", batch_model=None, max_breakpoints=0)
    profiles = {**PROFILES, other.model: other}
    router, fake = make_router(tmp_path, strategy="auto", profiles=profiles)
    calls = make_calls(groups=2, per_group=3) + make_calls(
        groups=2, per_group=3, model="openrouter/x/other"
    )
    keys = router.add(calls, task="generation")
    report = run(router)
    assert report.complete and report.sent_live == 12
    # The model without breakpoints gets none, and caches nothing
    for request in fake.requests:
        blocks = isinstance(request.body["messages"][1]["content"], list)
        assert blocks == (request.model == MODEL)
    files = sorted(f.name for f in (tmp_path / "generation").iterdir())
    assert files == [
        "openrouter_anthropic_claude-haiku-5.5.jsonl",
        "openrouter_x_other.jsonl",
    ]


def test_failed_request_to_the_batch_api_keeps_the_batches(tmp_path, monkeypatch):
    calls = make_calls(groups=1, per_group=3)
    router, fake = make_router(tmp_path, strategy="batch")
    keys = router.add(calls)
    poll = fake.poll

    async def failing_poll(entry):
        raise TransportError("service unavailable", 503, retryable=True)

    monkeypatch.setattr(fake, "poll", failing_poll)
    for start in (0, 2 * HOUR):  # while waiting, and when a run starts
        report = run(router, start=start, **LATER)
        assert report.pending == tuple(sorted(keys)) and fake.sent == []
        assert "Request to the batch API failed" in report.problems[0]
        assert len(fake.submissions) == 1
    monkeypatch.setattr(fake, "poll", poll)
    assert run(router, start=3 * HOUR, **LATER).complete
    assert len(fake.submissions) == 1


def test_no_retries(tmp_path):
    calls = make_calls(groups=1, per_group=2)
    router, fake = make_router(tmp_path, strategy="batch", answer=lambda r: "bad")
    keys = router.add(calls)
    fake.failed_in_batch = {keys[0]}
    report = run(router, accept=accept, retries=0, **LATER)
    # Nothing is repeated on the live lane; the reasons are kept
    assert fake.sent == []
    assert dict(report.gave_up) == {
        keys[0]: "request failed in batch batch_1",
        keys[1]: "answer rejected",
    }

"""Tests of the strategies and the planner (pure)."""

import random
from dataclasses import replace
from datetime import timedelta

import pytest

from feedback_forensics.router.datatypes import (
    BATCH,
    LIVE,
    TTL_1H,
    TTL_5M,
    Call,
    Message,
    Policy,
    Stage,
)
from feedback_forensics.router.strategies import (
    ASSUMED_BATCH_TURNAROUND,
    STRATEGIES,
    batch_hit_rate,
    make_plan,
    strategy,
)
from feedback_forensics.router.testutils import (
    FIXED_END,
    MIDDLE_END,
    MODEL,
    PROFILE,
    PROFILES,
    make_call,
    make_calls,
)

DAY = timedelta(hours=24)
LATER = Policy(deadline=timedelta(hours=72))  # time for two batches in a row


def plan_for(strategy_name, calls=None, policy=LATER, profile=PROFILE):
    calls = make_calls(groups=3, per_group=4) if calls is None else calls
    return make_plan(calls, {profile.model: profile}, policy, strategy_name)


def test_planner_is_pure():
    """The same calls give the same plan, whatever their order, and the planner
    touches neither network nor disk (it has no access to a store or transport)."""
    calls = make_calls(groups=4, per_group=3)
    shuffled = list(calls)
    random.Random(1).shuffle(shuffled)
    for name in STRATEGIES:
        assert plan_for(name, calls) == plan_for(name, shuffled + calls[:2])


def test_live_ordered():
    """Live lane, tree order, all cache layers."""
    plan = plan_for("live_ordered", policy=Policy())
    assert plan.strategy == "live_ordered"
    (stage,) = plan.stages
    steps = stage.steps
    assert {step.lane for step in steps} == {LIVE}
    assert {step.cache_ttl for step in steps} == {TTL_5M}
    assert {step.breakpoints for step in steps} == {(FIXED_END, MIDDLE_END)}
    groups = [step.call.tag["group"] for step in steps]
    assert groups == sorted(groups)

    # One call goes first and writes the fixed start. The first call of every
    # other group waits for it, the others of a group for the group's first.
    first = steps[0]
    assert first.after is None
    for step in steps[1:]:
        group_first = next(
            s for s in steps if s.call.tag["group"] == step.call.tag["group"]
        )
        expected = first if step is group_first else group_first
        assert step.after == expected.call.key
    # `after` always names a call that comes earlier
    position = {step.call.key: i for i, step in enumerate(steps)}
    assert all(position[s.after] < position[s.call.key] for s in steps if s.after)


def test_live_ordered_estimate():
    calls = make_calls(groups=1, per_group=2)
    plan = plan_for("live_ordered", calls, Policy())
    writer, reader = plan.stages[0].steps
    assert writer.breakpoints == reader.breakpoints == (MIDDLE_END,)
    tokens = len(calls[0].text)  # one character per token
    output = 50 * PROFILE.output_price  # `max_tokens` of the calls
    rest = tokens - MIDDLE_END
    assert writer.estimated_usd == pytest.approx(
        ((MIDDLE_END * 1.25 + rest) * PROFILE.input_price + output) / 1e6
    )
    assert reader.estimated_usd == pytest.approx(
        ((MIDDLE_END * 0.1 + rest) * PROFILE.input_price + output) / 1e6
    )
    estimate = plan.estimate
    assert estimate.planned_usd == pytest.approx(
        writer.estimated_usd + reader.estimated_usd
    )
    assert estimate.as_given_usd == pytest.approx(
        2 * (tokens * PROFILE.input_price + output) / 1e6
    )
    assert estimate.cached_tokens == MIDDLE_END
    assert estimate.input_tokens == 2 * tokens
    assert estimate.planned_usd < estimate.as_given_usd


def test_batch():
    """Everything in batches, assuming no cache hits."""
    plan = plan_for("batch")
    (stage,) = plan.stages
    assert {(s.lane, s.breakpoints, s.after) for s in stage.steps} == {
        (BATCH, (), None)
    }
    assert plan.estimate.planned_usd == pytest.approx(plan.estimate.as_given_usd / 2)
    assert plan.estimate.duration == ASSUMED_BATCH_TURNAROUND


@pytest.mark.parametrize(
    "name, writer_lane", [("warm_then_batch", LIVE), ("batch_chain", BATCH)]
)
def test_fixed_prefix_then_batches(name, writer_lane):
    """Stage 1: one call writes the fixed prefix with a 1-hour lifetime (live,
    or as a small batch). Stage 2: batches with a breakpoint on that prefix."""
    plan = plan_for(name)
    first, second = plan.stages
    (writer,) = first.steps
    assert (writer.lane, writer.breakpoints, writer.cache_ttl, writer.after) == (
        writer_lane,
        (FIXED_END,),
        TTL_1H,
        None,
    )
    assert len(second.steps) == 11
    assert {(s.lane, s.breakpoints, s.cache_ttl, s.after) for s in second.steps} == {
        (BATCH, (FIXED_END,), TTL_1H, None)
    }
    assert writer.call.key not in {s.call.key for s in second.steps}
    batches = 1 if writer_lane == LIVE else 2
    assert DAY * batches <= plan.estimate.duration < DAY * batches + timedelta(hours=1)


def test_fixed_prefix_strategies_without_shared_prefix():
    calls = [Call(MODEL, (Message("user", f"{i} alone"),)) for i in range(3)]
    for name in ("warm_then_batch", "batch_chain"):
        (stage,) = plan_for(name, calls).stages
        assert {(s.lane, s.breakpoints) for s in stage.steps} == {(BATCH, ())}


def test_batch_strategies_need_a_deadline_and_a_batch_variant():
    for name in ("batch", "warm_then_batch", "batch_chain"):
        with pytest.raises(ValueError, match="deadline"):
            plan_for(name, policy=Policy(deadline=None))
        with pytest.raises(ValueError, match="batch variant"):
            plan_for(name, profile=replace(PROFILE, batch_model=None))
    with pytest.raises(ValueError, match="Unknown strategy"):
        plan_for("nonsense")
    with pytest.raises(KeyError, match="No model profile"):
        make_plan([make_call(0, 0, model="openrouter/x/y")], PROFILES, Policy())


def test_batch_estimate_depends_on_hit_rate():
    """In batches, a request that misses the prefix pays to write it again."""
    calls = mostly_unique_calls()
    always = replace(PROFILE, observed_hit_rate=1.0)
    never = replace(PROFILE, observed_hit_rate=0.0)
    cost = {
        (name, profile.observed_hit_rate): plan_for(
            name, calls, profile=profile
        ).estimate.planned_usd
        for name in ("batch", "warm_then_batch", "batch_chain")
        for profile in (always, never, PROFILE)
    }
    assert cost["batch_chain", 1.0] < cost["warm_then_batch", 1.0] < cost["batch", 1.0]
    assert cost["batch", 0.0] < cost["batch_chain", 0.0]
    assert cost["batch", 0.0] < cost["warm_then_batch", 0.0]
    # Not measured: assumed not to pay off
    assert cost["batch", None] < cost["batch_chain", None]
    assert batch_hit_rate(PROFILE) == 0.5
    assert batch_hit_rate(replace(always, cache_in_batch=False)) == 0.0
    assert batch_hit_rate(replace(always, max_breakpoints=0)) == 0.0


def mostly_unique_calls(n: int = 30) -> list[Call]:
    """Calls whose prompts are mostly unique: caching alone saves less than
    batches, but enough calls share the start for writing it to pay off."""
    calls = []
    for i in range(n):
        call = make_call(0, i)
        user = Message("user", call.messages[1].text + f"{i} " + "x" * 3000)
        calls.append(Call(call.model, (call.messages[0], user), call.params))
    return calls


def auto(profile=PROFILE, deadline=None, calls=None) -> str:
    """The strategy `auto` picks."""
    calls = mostly_unique_calls() if calls is None else calls
    return plan_for("auto", calls, Policy(deadline=deadline), profile).strategy


def test_auto_picks_the_cheapest_strategy_that_meets_the_deadline():
    measured = replace(PROFILE, observed_hit_rate=0.95)
    # Results needed now: no batches
    assert auto(deadline=None) == "live_ordered"
    assert auto(measured, deadline=None) == "live_ordered"
    # Too little time for a batch
    assert auto(measured, deadline=timedelta(hours=2)) == "live_ordered"
    # Time for one batch. Hit rate in batches not measured: plain batches
    assert auto(deadline=timedelta(hours=30)) == "batch"
    # ... measured high: writing the prefix first pays off
    assert auto(measured, deadline=timedelta(hours=30)) == "warm_then_batch"
    # Time for two batches in a row: writing it with a small batch is cheapest
    assert auto(measured, deadline=timedelta(hours=60)) == "batch_chain"
    assert auto(deadline=timedelta(hours=60)) == "batch"
    # ... measured low
    low = replace(PROFILE, observed_hit_rate=0.2)
    assert auto(low, deadline=timedelta(hours=60)) == "batch"
    # Caching does not work in batches, or not at all
    no_batch_cache = replace(measured, cache_in_batch=False)
    assert auto(no_batch_cache, deadline=timedelta(hours=60)) == "batch"
    no_cache = replace(PROFILE, max_breakpoints=0)
    assert auto(no_cache, deadline=timedelta(hours=60)) == "batch"
    # No batch variant
    no_batch = replace(measured, batch_model=None)
    assert auto(no_batch, deadline=timedelta(hours=60)) == "live_ordered"
    # Batches at the live price: caching on the live lane is cheapest
    full_price = replace(PROFILE, batch_price_multiplier=1.0)
    assert auto(full_price, deadline=timedelta(hours=60)) == "live_ordered"


def test_auto_prefers_live_caching_where_most_of_the_prompt_is_shared():
    calls = make_calls(groups=2, per_group=20)
    assert auto(calls=calls, deadline=timedelta(hours=60)) == "live_ordered"


def test_auto_is_the_cheapest_plan():
    measured = replace(PROFILE, observed_hit_rate=0.95)
    plans = {
        name: plan_for(name, profile=measured)
        for name in ("live_ordered", "batch", "warm_then_batch", "batch_chain")
    }
    chosen = plan_for("auto", profile=measured)
    assert chosen == min(plans.values(), key=lambda p: p.estimate.planned_usd)


def test_plans_of_several_models_are_merged():
    other = replace(PROFILE, model="x/other", batch_model=None)
    profiles = {**PROFILES, other.model: other}
    calls = make_calls(groups=2, per_group=3) + make_calls(
        groups=2, per_group=3, model="openrouter/x/other"
    )
    plan = make_plan(calls, profiles, LATER, "live_ordered")
    assert plan.strategy == "live_ordered"
    assert plan.choices == (
        (MODEL, "live_ordered"),
        ("openrouter/x/other", "live_ordered"),
    )
    assert len(plan.steps) == 12
    # No waiting for calls of another model
    by_key = {step.call.key: step for step in plan.steps}
    assert all(
        by_key[s.after].call.model == s.call.model for s in plan.steps if s.after
    )

    mixed = make_plan(
        calls,
        {**profiles, PROFILE.model: replace(PROFILE, max_breakpoints=0)},
        LATER,
        "auto",
    )
    assert mixed.strategy == "mixed"
    assert dict(mixed.choices)[MODEL] == "batch"
    assert {s.lane for s in mixed.steps} == {LIVE, BATCH}
    assert make_plan([], profiles, LATER, "auto").stages == ()


def test_adding_a_strategy():
    """A strategy is a function from tree, profiles and policy to a plan."""

    @strategy("live_reversed")
    def live_reversed(tree, profiles, policy):
        plan = STRATEGIES["live_ordered"](tree, profiles, policy)
        steps = tuple(replace(s, after=None) for s in reversed(plan.steps))
        return replace(plan, strategy="live_reversed", stages=(Stage(steps),))

    try:
        plan = plan_for("live_reversed", policy=Policy())
        assert plan.strategy == "live_reversed"
        groups = [step.call.tag["group"] for step in plan.steps]
        assert groups == sorted(groups, reverse=True)
        # `auto` considers it too
        assert "live_reversed" in STRATEGIES
    finally:
        del STRATEGIES["live_reversed"]

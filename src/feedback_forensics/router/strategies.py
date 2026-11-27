"""Strategies: each turns the prefix tree of one model's calls into a plan.

A strategy is a function `(tree, profiles, policy) -> Plan`, registered under a
name with `@strategy(name)`. It decides the lane, the cache breakpoints and the
stage of every call; `finish_plan` then derives the `after` ordering and the
cost estimate from that. Strategies are pure (no network, no disk). A strategy
that cannot be used (e.g. batches without a deadline) raises `ValueError`.

The estimates rest on the assumptions below (constants) and on the model
profile. On the live lane, a call reads a prefix that an earlier call has
finished writing (the executor orders them so). In batches, a request reads a
prefix written in an earlier stage with probability `batch_hit_rate`, and a
request that misses pays to write it again.
"""

import math
from dataclasses import dataclass
from datetime import timedelta
from itertools import zip_longest
from typing import Callable, Iterable, Mapping

from feedback_forensics.router.prefix import (
    build_prefix_tree,
    estimate_tokens,
    node_id,
    node_paths,
)
from feedback_forensics.router.profiles import get_profile
from feedback_forensics.router.datatypes import (
    BATCH,
    LIVE,
    TTL_1H,
    TTL_5M,
    Call,
    Estimate,
    ModelProfile,
    Plan,
    Policy,
    PrefixNode,
    PrefixTree,
    Stage,
    Step,
)

# --- Assumptions of the estimates (not measured) --------------------------------
# Output tokens per call (capped by the call's `max_tokens`)
ASSUMED_OUTPUT_TOKENS = 300
# Duration of a live call
ASSUMED_LIVE_LATENCY = timedelta(seconds=20)
# Time until the results of a batch are available (OpenRouter: within ~24 h)
ASSUMED_BATCH_TURNAROUND = timedelta(hours=24)
# Share of batch requests reading a prefix written in an earlier stage, while
# not measured (`ModelProfile.observed_hit_rate`). At this value writing the
# prefix does not pay off for typical prices, so `auto` only chooses
# `warm_then_batch` or `batch_chain` once a run with one of them has measured
# a higher rate.
UNMEASURED_BATCH_HIT_RATE = 0.5

Strategy = Callable[[PrefixTree, Mapping[str, ModelProfile], Policy], Plan]
STRATEGIES: dict[str, Strategy] = {}


def strategy(name: str):
    """Register a strategy under a name (decorator)."""

    def register(fn: Strategy) -> Strategy:
        STRATEGIES[name] = fn
        return fn

    return register


def batch_hit_rate(profile: ModelProfile) -> float:
    """Expected share of batch requests that read a prefix written before."""
    if profile.cache_in_batch is False or profile.max_breakpoints < 1:
        return 0.0
    if profile.observed_hit_rate is not None:
        return profile.observed_hit_rate
    return UNMEASURED_BATCH_HIT_RATE


@dataclass
class Draft:
    """A strategy's decision for one call: lane, and the shared prefixes
    (`nodes`, shortest first) that get a cache breakpoint."""

    call: Call
    lane: str = LIVE
    nodes: tuple[PrefixNode, ...] = ()
    ttl: str = TTL_5M


def _output_tokens(call: Call) -> int:
    if call.params.max_tokens is None:
        return ASSUMED_OUTPUT_TOKENS
    return min(ASSUMED_OUTPUT_TOKENS, call.params.max_tokens)


def finish_plan(
    name: str,
    tree: PrefixTree,
    profile: ModelProfile,
    policy: Policy,
    stages: list[list[Draft]],
) -> Plan:
    """The plan for the drafts of a strategy (one list per stage, in sending
    order), with `after` and the cost estimate.

    The first call with a breakpoint at a prefix writes it. Live calls after it
    in the same stage wait for it (`Step.after`: the writer of the longest of
    their prefixes that they do not write themselves). Later stages read it.
    """
    written: dict[tuple, tuple[int, str, str]] = {}  # (stage, lane, writer key)
    plan_stages = []
    as_given = planned = cached_total = 0.0
    input_total = output_total = 0
    duration = timedelta(0)
    for stage_index, drafts in enumerate(stages):
        new: dict[tuple, tuple[int, str, str]] = {}
        steps = []
        depth = num_live = num_batch = 0
        for draft in drafts:
            call, nodes = draft.call, draft.nodes
            hit_tokens, rate, after = 0, 0.0, None
            for node in reversed(nodes):
                writer = written.get(node_id(tree, node))
                if writer is None and draft.lane == LIVE:
                    writer = new.get(node_id(tree, node))
                    if writer is not None and writer[1] != LIVE:
                        writer = None
                if writer is None:
                    continue
                if draft.lane == LIVE and writer[1] == LIVE:
                    rate = 1.0
                    after = writer[2] if writer[0] == stage_index else None
                else:
                    rate = batch_hit_rate(profile)
                hit_tokens = node.tokens
                break
            for node in nodes:
                if node_id(tree, node) not in written:
                    new.setdefault(
                        node_id(tree, node), (stage_index, draft.lane, call.key)
                    )

            total = estimate_tokens(len(call.text), profile)
            deepest = nodes[-1].tokens if nodes else 0
            cached = rate * hit_tokens
            write = (deepest - hit_tokens) + (1 - rate) * hit_tokens
            output = _output_tokens(call)
            lane_price = profile.batch_price_multiplier if draft.lane == BATCH else 1.0
            usd = (
                lane_price
                * (
                    profile.input_price
                    * (
                        cached * profile.cache_read_multiplier
                        + write * profile.cache_write_multiplier(draft.ttl)
                        + (total - deepest)
                    )
                    + profile.output_price * output
                )
                / 1e6
            )
            steps.append(
                Step(
                    call=call,
                    lane=draft.lane,
                    breakpoints=tuple(node.end for node in nodes),
                    after=after,
                    cache_ttl=draft.ttl,
                    estimated_usd=usd,
                )
            )
            planned += usd
            as_given += (
                profile.input_price * total + profile.output_price * output
            ) / 1e6
            input_total += total
            output_total += output
            cached_total += cached
            if draft.lane == LIVE:
                num_live += 1
                depth = max(depth, len(nodes))
            else:
                num_batch += 1
        written.update(new)
        if steps:
            plan_stages.append(Stage(tuple(steps)))
        live_time = (
            (math.ceil(num_live / policy.max_concurrency) + depth)
            * ASSUMED_LIVE_LATENCY
            if num_live
            else timedelta(0)
        )
        duration += max(
            live_time, ASSUMED_BATCH_TURNAROUND if num_batch else timedelta(0)
        )
    return Plan(
        strategy=name,
        stages=tuple(plan_stages),
        estimate=Estimate(
            as_given_usd=as_given,
            planned_usd=planned,
            input_tokens=input_total,
            cached_tokens=cached_total,
            output_tokens=output_total,
            duration=duration,
        ),
        choices=((tree.model, name),) if tree.calls else (),
    )


def choose_breakpoints(
    tree: PrefixTree, profile: ModelProfile, ttl: str = TTL_5M
) -> dict[str, tuple[PrefixNode, ...]]:
    """For each call (key), the shared prefixes that get a breakpoint: at most
    `profile.max_breakpoints`, those with the largest saving. The saving of a
    prefix is that of the tokens it adds to the next shorter one: read by all
    calls but one at the read price, minus the surcharge for writing them once.
    """
    paths = node_paths(tree)
    if profile.max_breakpoints < 1:
        return {key: () for key in paths}
    read_saving = 1 - profile.cache_read_multiplier
    write_surcharge = profile.cache_write_multiplier(ttl) - 1
    chosen = {}
    for key, path in paths.items():
        savings, shorter = [], 0
        for depth, node in enumerate(path):
            added = node.tokens - shorter
            shorter = node.tokens
            saving = added * ((len(node.keys) - 1) * read_saving - write_surcharge)
            if saving > 0:
                savings.append((-saving, depth, node))
        best = sorted(savings, key=lambda s: s[:2])[: profile.max_breakpoints]
        chosen[key] = tuple(node for _, _, node in sorted(best, key=lambda s: s[1]))
    return chosen


def _check_batch(name: str, profile: ModelProfile, policy: Policy) -> None:
    if policy.deadline is None:
        raise ValueError(
            f"Strategy '{name}' sends batches, which needs a deadline "
            "(`Policy.deadline` is None: results are needed now)."
        )
    if profile.batch_model is None:
        raise ValueError(
            f"Strategy '{name}' sends batches, but {profile.model} has no batch "
            "variant (`ModelProfile.batch_model`)."
        )


@strategy("live_ordered")
def live_ordered(tree, profiles, policy) -> Plan:
    """Live lane, tree order, all cache layers."""
    profile = get_profile(profiles, tree.model)
    nodes = choose_breakpoints(tree, profile)
    drafts = [Draft(call, LIVE, nodes[call.key]) for call in tree.calls]
    return finish_plan("live_ordered", tree, profile, policy, [drafts])


@strategy("batch")
def batch(tree, profiles, policy) -> Plan:
    """Everything in batches, assuming no cache hits (no breakpoints)."""
    profile = get_profile(profiles, tree.model)
    _check_batch("batch", profile, policy)
    drafts = [Draft(call, BATCH) for call in tree.calls]
    return finish_plan("batch", tree, profile, policy, [drafts])


def _fixed_prefix_stages(tree, profile, writer_lane: str) -> list[list[Draft]]:
    """Stage 1: for each fixed prefix (the longest prefixes shared most widely:
    the children of the root), its first call writes it with a 1-hour lifetime.
    Stage 2: all other calls in batches, with a breakpoint on that prefix
    (without breakpoint for calls that share no prefix)."""
    calls = {call.key: call for call in tree.calls}
    first, second = [], []
    under_prefix = set()
    for node in tree.root.children if profile.max_breakpoints >= 1 else ():
        first.append(Draft(calls[node.keys[0]], writer_lane, (node,), TTL_1H))
        second += [Draft(calls[k], BATCH, (node,), TTL_1H) for k in node.keys[1:]]
        under_prefix.update(node.keys)
    second += [Draft(c, BATCH) for c in tree.calls if c.key not in under_prefix]
    return [first, second]


@strategy("warm_then_batch")
def warm_then_batch(tree, profiles, policy) -> Plan:
    """Stage 1: one live call writes the fixed prefix with a 1-hour lifetime.
    Stage 2: batches with a breakpoint on that prefix."""
    profile = get_profile(profiles, tree.model)
    _check_batch("warm_then_batch", profile, policy)
    stages = _fixed_prefix_stages(tree, profile, LIVE)
    return finish_plan("warm_then_batch", tree, profile, policy, stages)


@strategy("batch_chain")
def batch_chain(tree, profiles, policy) -> Plan:
    """Stage 1: a small batch writes the fixed prefix. Stage 2: the main batches,
    with a breakpoint on that prefix."""
    profile = get_profile(profiles, tree.model)
    _check_batch("batch_chain", profile, policy)
    stages = _fixed_prefix_stages(tree, profile, BATCH)
    return finish_plan("batch_chain", tree, profile, policy, stages)


@strategy("auto")
def auto(tree, profiles, policy) -> Plan:
    """The cheapest of the registered strategies that the policy allows and
    that meets the deadline (the fastest if none does)."""
    plans = []
    for name, fn in STRATEGIES.items():
        if name == "auto":
            continue
        try:
            plans.append(fn(tree, profiles, policy))
        except ValueError:
            continue  # not allowed for this policy or model
    in_time = [
        plan
        for plan in plans
        if policy.deadline is None or plan.estimate.duration <= policy.deadline
    ]
    if not in_time:
        in_time = [min(plans, key=lambda plan: plan.estimate.duration)]
    # (the first registered wins ties: the simpler strategies come first)
    return min(in_time, key=lambda plan: plan.estimate.planned_usd)


def make_plan(
    calls: Iterable[Call],
    profiles: Mapping[str, ModelProfile],
    policy: Policy,
    strategy: str = "auto",
) -> Plan:
    """Plan a set of calls (pure: the same input gives the same plan; no
    network, no disk).

    Builds the prefix tree of each model's calls and hands it to the strategy.
    The plans of the models are merged stage by stage.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}' (known: {sorted(STRATEGIES)})."
        )
    by_model: dict[str, dict[str, Call]] = {}
    for call in calls:
        by_model.setdefault(call.model, {}).setdefault(call.key, call)
    plans = []
    for model in sorted(by_model):
        profile = get_profile(profiles, model)
        tree = build_prefix_tree(by_model[model].values(), profile)
        plans.append(STRATEGIES[strategy](tree, profiles, policy))
    if not plans:
        return Plan(strategy=strategy)
    names = {plan.strategy for plan in plans}
    estimates = [plan.estimate for plan in plans]
    return Plan(
        strategy=names.pop() if len(names) == 1 else "mixed",
        stages=tuple(
            Stage(tuple(step for stage in stages if stage for step in stage.steps))
            for stages in zip_longest(*(plan.stages for plan in plans))
        ),
        estimate=Estimate(
            as_given_usd=sum(e.as_given_usd for e in estimates),
            planned_usd=sum(e.planned_usd for e in estimates),
            input_tokens=sum(e.input_tokens for e in estimates),
            cached_tokens=sum(e.cached_tokens for e in estimates),
            output_tokens=sum(e.output_tokens for e in estimates),
            # (the models' plans run side by side)
            duration=max(e.duration for e in estimates),
        ),
        choices=tuple(choice for plan in plans for choice in plan.choices),
    )

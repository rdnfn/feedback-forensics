"""Helpers for router tests: synthetic calls shaped like the annotation case,
a model profile, and a router on a fake provider."""

from dataclasses import replace

from feedback_forensics.router.datatypes import Call, Message, Params
from feedback_forensics.router.fakes import FakeProvider
from feedback_forensics.router.profiles import HAIKU_5_5
from feedback_forensics.router.router import Router

MODEL = "openrouter/anthropic/claude-haiku-5.5"
# One character per token, and short prefixes can be cached: lengths in the
# tests are easy to follow
PROFILE = replace(HAIKU_5_5, chars_per_token=1.0, min_cacheable_tokens=100)
PROFILES = {PROFILE.model: PROFILE}
SYSTEM = "You compare two samples."


def lines(label: str, n: int) -> str:
    """`n` lines of 30 characters each."""
    return "".join(f"{label:<20.20} line {i:03d}\n" for i in range(n))


FIXED = lines("fixed instructions", 20)  # 600 characters


def middle(group: int, n: int = 10) -> str:
    return lines(f"prompt {group}", n)


def make_call(
    group: int, i: int, model: str = MODEL, fixed: str = FIXED, middle_lines: int = 10
) -> Call:
    """A call with a fixed start (shared by all), a middle part (shared by its
    group) and a unique end. The unique ends start alike within a line."""
    user = (
        fixed
        + middle(group, middle_lines)
        + f"response {group}-{i}\n"
        + lines(f"r{group}-{i}", 4)
    )
    return Call(
        model=model,
        messages=(Message("system", SYSTEM), Message("user", user)),
        params=Params(max_tokens=50),
        tag={"group": group, "i": i},
    )


def make_calls(groups: int = 3, per_group: int = 4, **kwargs) -> list[Call]:
    return [make_call(g, i, **kwargs) for g in range(groups) for i in range(per_group)]


FIXED_END = len(SYSTEM) + len(FIXED)  # end of the fixed start in `Call.text`
MIDDLE_END = FIXED_END + len(middle(0))


def make_router(path, strategy="live_ordered", profiles=None, **fake_kwargs):
    """(router, its fake provider), on the store in `path`."""
    profiles = PROFILES if profiles is None else profiles
    fake = FakeProvider(profiles, **fake_kwargs)
    router = Router(path, strategy=strategy, profiles=profiles, transport=fake)
    return router, fake

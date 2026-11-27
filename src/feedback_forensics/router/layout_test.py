"""Tests of the layout check, on a synthetic case with a known answer."""

import pytest

from feedback_forensics.router.datatypes import Call, Message
from feedback_forensics.router.layout import check_layout
from feedback_forensics.router.testutils import (
    FIXED,
    MODEL,
    PROFILE,
    PROFILES,
    lines,
    make_calls,
)

RULES = "Rules:\n" + lines("rule", 10) + "\nAnswer in JSON.\n" + lines("format", 3)
QUOTED = lines("quoted code", 5)


def make_call(i: int, quoted: bool = False) -> Call:
    """Fixed start, then text unique to the call, then text that all calls
    share (the rules), which no cache can reach."""
    unique = f"Sample {i}:\n" + lines(f"sample {i}", 4)
    if quoted:
        unique += "\n\n" + QUOTED + "\n\n" + f"More of sample {i}."
    return Call(MODEL, (Message("user", FIXED + "\n\n" + unique + "\n\n" + RULES),))


def test_repeated_text_behind_unique_text_is_reported():
    calls = [make_call(i, quoted=i < 3) for i in range(6)]
    report = check_layout(calls, PROFILES)
    rules, quoted = report.findings

    # The rules: two paragraphs, in all six calls
    assert rules.model == MODEL and rules.calls == 6
    assert rules.tokens == len(RULES.rstrip())  # one character per token
    assert rules.preview.startswith("Rules:\nrule")
    # If cached: read by five calls at 10% of the price, written once for 25% more
    assert rules.saving_usd == pytest.approx(
        rules.tokens * PROFILE.input_price * (5 * 0.9 - 0.25) / 1e6
    )
    offsets = sorted(call.text.index("Rules:") for call in calls)
    assert offsets[0] <= rules.offset <= offsets[-1]

    # Text that three calls quote (cannot be moved: reported as a suggestion)
    assert quoted.calls == 3 and quoted.preview.startswith("quoted code")
    assert quoted.saving_usd == pytest.approx(
        quoted.tokens * PROFILE.input_price * (2 * 0.9 - 0.25) / 1e6
    )
    text = str(report)
    assert "Suggestions only" in text and "6 calls" in text and "3 calls" in text


def test_nothing_to_report():
    # Shared text only at the start (reachable by the cache)
    report = check_layout(make_calls(groups=3, per_group=3), PROFILES)
    assert report.findings == ()
    assert "no repeated text" in str(report)
    # A model that does not cache has nothing to gain
    from dataclasses import replace

    no_cache = {PROFILE.model: replace(PROFILE, max_breakpoints=0)}
    assert check_layout([make_call(i) for i in range(6)], no_cache).findings == ()
    assert check_layout([], PROFILES).findings == ()


def test_prompts_are_not_changed():
    calls = [make_call(i) for i in range(4)]
    texts = [call.text for call in calls]
    check_layout(calls, PROFILES, min_calls=3, max_findings=1)
    assert [call.text for call in calls] == texts

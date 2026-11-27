"""Tests of the adapter from annotation requests to router calls."""

from feedback_forensics.annotation.batch import _request_body
from feedback_forensics.annotation.voting import VotingConfig, _VoteRequest
from feedback_forensics.router.adapters import call_from_vote_request
from feedback_forensics.router.datatypes import Policy
from feedback_forensics.router.layout import check_layout
from feedback_forensics.router.profiles import HAIKU_5_5
from feedback_forensics.router.strategies import make_plan
from feedback_forensics.router.wire import build_body

MODEL = "openrouter/anthropic/claude-haiku-5.5"
PRINCIPLES = [f"Select the response that is more {x}" for x in ("concise", "formal")]
PROFILES = {HAIKU_5_5.model: HAIKU_5_5}


def vote_request(prompt="What is a cache?", response="b", flipped=False):
    hashes = {
        principle: f"hash-{i}-{flipped}" for i, principle in enumerate(PRINCIPLES)
    }
    return _VoteRequest(
        preferred="Reference response " * 20,
        rejected=f"Response of model {response} " * 20,
        flipped=flipped,
        hashes=hashes,
        prompt=prompt,
    )


def test_call_matches_the_request_annotation_sends_today():
    for reasoning_effort in (None, "low"):
        config = VotingConfig(
            model=MODEL, max_output_tokens=2000, reasoning_effort=reasoning_effort
        )
        request = vote_request()
        call = call_from_vote_request(request, config)
        assert call.model == MODEL
        assert [(m.role, m.text) for m in call.messages] == [
            (m["role"], m["content"]) for m in request.messages
        ]
        # Without breakpoints, the router sends what annotation sends in batches
        assert build_body(call) == _request_body(request, config)
        assert call.tag == request.request_id
        assert call_from_vote_request(request, config, tag="mine").tag == "mine"
        assert call_from_vote_request(request, config).key == call.key
    other = call_from_vote_request(vote_request(flipped=True), config)
    assert other.key != call.key
    assert call_from_vote_request(request, VotingConfig(model=MODEL)).key != call.key


def test_annotation_calls_fit_the_planner():
    """Calls for the same user prompt share their start (the system message,
    and the user prompt): the planner finds it without being told."""
    config = VotingConfig(model=MODEL)
    prompt = "Explain prompt caching. " * 150
    calls = [
        call_from_vote_request(vote_request(prompt, response, flipped=True), config)
        for response in "bcdef"
    ]
    plan = make_plan(calls, PROFILES, Policy(), "live_ordered")
    # (up to and including the line "Sample A:")
    shared = calls[0].text.index("Sample A:") + len("Sample A:\n")
    assert {step.breakpoints for step in plan.steps} == {(shared,)}
    assert sum(step.after is None for step in plan.steps) == 1
    # In today's prompt, the reference response (here sample B) and the rules
    # follow the response that differs between the calls, where no cache can
    # reach them: the layout check points at them
    report = check_layout(calls, PROFILES)
    finding = report.findings[0]
    assert finding.calls == 5 and finding.preview.startswith("Sample B:\nReference")
    rules = calls[0].text[calls[0].text.index("Given the samples data above") :]
    assert finding.tokens > len(rules) / HAIKU_5_5.chars_per_token

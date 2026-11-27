"""Tests of the router's data types: call keys."""

import hashlib
import json

import pytest

from feedback_forensics.router.datatypes import (
    Call,
    Message,
    Params,
    Result,
    call_key,
)
from feedback_forensics.router.store import Store
from feedback_forensics.router.wire import build_body, split_pieces

MESSAGES = (Message("system", "Be brief."), Message("user", "Line one\nLine two\n"))


def make(**kwargs) -> Call:
    kwargs = {"model": "openrouter/a/b", "messages": MESSAGES, **kwargs}
    return Call(**kwargs)


def test_key_is_stable():
    # Pinned: keys identify stored answers, so they must never change
    call = make(params=Params(max_tokens=100, reasoning_effort="low"))
    assert call.key == (
        "2ebb62fce6a9be3d86a803d3294520d7407a62aad217dccfca27a435875668f8"
    )
    assert call.key == make(params=Params(100, "low", 0.0)).key


def test_key_follows_documented_recipe():
    """The key can be computed without this code (as the store README says)."""
    call = make(params=Params(max_tokens=100))
    content = {
        "messages": [["system", "Be brief."], ["user", "Line one\nLine two\n"]],
        "model": "openrouter/a/b",
        "params": {"max_tokens": 100, "reasoning_effort": None, "temperature": 0.0},
    }
    text = json.dumps(
        content, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    assert call.key == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_key_depends_on_model_params_and_messages_only():
    base = make()
    assert make(tag={"any": "thing"}).key == base.key
    assert make(messages=list(MESSAGES)).key == base.key
    assert make(params=Params(temperature=0)).key == base.key  # 0 == 0.0
    different = [
        make(model="openrouter/a/c"),
        make(params=Params(max_tokens=5)),
        make(params=Params(reasoning_effort="low")),
        make(params=Params(temperature=None)),
        make(messages=MESSAGES[1:]),
        make(messages=(Message("user", "Be brief."), MESSAGES[1])),
        # Same text, cut into messages elsewhere
        make(
            messages=(
                Message("system", "Be brief.Line one\n"),
                Message("user", "Line two\n"),
            )
        ),
    ]
    keys = {call.key for call in different} | {base.key}
    assert len(keys) == len(different) + 1


def test_key_does_not_depend_on_cut_points(tmp_path):
    """Cutting the prompt for caching changes the request and the stored
    pieces, but not the key or the prompt."""
    call = make()
    cut = len("Be brief.") + len("Line one\n")
    assert build_body(call, (cut,)) != build_body(call)
    for i, breakpoints in enumerate([(), (cut,), (len("Be brief."), cut)]):
        pieces = split_pieces(call, breakpoints)
        assert ["".join(text for text, _ in m) for m in pieces] == [
            m.text for m in call.messages
        ]
        store = Store(tmp_path / str(i))
        store.put(call, Result(call.key, text="ok"), breakpoints=breakpoints)
        assert store.record(call.key)["key"] == call.key
        assert store.call(call.key).key == call.key
    assert call_key(call.model, call.params, call.messages) == call.key


def test_invalid_calls_are_rejected():
    with pytest.raises(ValueError):
        make(messages=())
    with pytest.raises(ValueError):
        Message("user\n", "text")

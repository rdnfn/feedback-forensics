"""Tests of the call store: round trip, file layout, interrupted writes."""

import hashlib
import json

import pytest

from feedback_forensics.router.datatypes import Call, Message, Params, Result, Usage
from feedback_forensics.router.store import Store
from feedback_forensics.router.testutils import FIXED_END, MIDDLE_END, make_calls

FIELDS = {
    "key",
    "task",
    "group",
    "model",
    "params",
    "messages",
    "tag",
    "text",
    "finish_reason",
    "usage",
    "route",
    "batch_id",
    "provider",
    "created",
    "origin",
}


def answer(call: Call, text="An answer", **kwargs) -> Result:
    usage = Usage(
        input_tokens=120,
        cached_tokens=100,
        cache_write_tokens=0,
        output_tokens=7,
        reasoning_tokens=2,
        cost_usd=0.0012,
    )
    kwargs = {"finish_reason": "stop", "usage": usage, "route": "live", **kwargs}
    return Result(call.key, call.tag, text, **kwargs)


def read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_round_trip(tmp_path):
    calls = make_calls(groups=2, per_group=2)
    store = Store(tmp_path / "calls")
    assert calls[0].key not in store and store.result(calls[0].key) is None
    for i, call in enumerate(calls):
        store.put(
            call,
            answer(call, f"Answer {i} ✓"),
            task="annotation",
            group="openai/gpt-x",
            breakpoints=(FIXED_END, MIDDLE_END),
            provider="Anthropic",
        )

    for store in (store, Store(tmp_path / "calls")):  # as written, and reopened
        assert len(store) == 4 and set(store.keys()) == {c.key for c in calls}
        for i, call in enumerate(calls):
            assert call.key in store
            result = store.result(call.key, tag=call.tag)
            assert result == Result(
                call.key,
                call.tag,
                f"Answer {i} ✓",
                finish_reason="stop",
                usage=answer(call).usage,
                route="store",
            )
            assert result.tag is call.tag and result.ok
            restored = store.call(call.key)
            assert restored == call and restored.key == call.key
            assert restored.tag == call.tag
    assert store.call("missing") is None


def test_file_layout(tmp_path):
    """The documented layout, readable without this code."""
    path = tmp_path / "calls"
    store = Store(path)
    annotation = make_calls(groups=2, per_group=2)
    generation = Call(
        "openrouter/openai/gpt-x", (Message("user", "Write a poem."),), Params(100)
    )
    for call in annotation:
        store.put(
            call,
            answer(call, route="batch"),
            task="annotation",
            group="openrouter/openai/gpt-x",
            breakpoints=(FIXED_END, MIDDLE_END),
            batch_id="batch_1",
        )
    store.put(generation, answer(generation), task="generation")

    files = sorted(str(f.relative_to(path)) for f in path.rglob("*") if f.is_file())
    assert files == [
        "README.md",
        "annotation/openrouter_anthropic_claude-haiku-5.5/openrouter_openai_gpt-x.jsonl",
        "generation/openrouter_openai_gpt-x.jsonl",
        "pieces/part-000.jsonl",
    ]
    readme = (path / "README.md").read_text()
    assert all(f"`{field}`" in readme for field in FIELDS)

    # Each distinct piece of prompt text is stored once: the system message,
    # the fixed start, two middle parts, four unique ends, the generation prompt
    pieces = read_lines(path / "pieces" / "part-000.jsonl")
    assert len(pieces) == 1 + 1 + 2 + 4 + 1
    assert len({p["hash"] for p in pieces}) == len(pieces)
    texts = {p["hash"]: p["text"] for p in pieces}
    assert all(
        hashlib.sha256(text.encode()).hexdigest() == h for h, text in texts.items()
    )

    records = read_lines(path / files[1])
    assert len(records) == 4
    for record, call in zip(records, annotation):
        assert set(record) == FIELDS
        assert record["key"] == call.key
        assert (record["task"], record["group"]) == (
            "annotation",
            "openrouter/openai/gpt-x",
        )
        assert record["model"] == call.model
        assert record["params"] == {
            "max_tokens": 50,
            "reasoning_effort": None,
            "temperature": 0.0,
        }
        assert record["tag"] == call.tag
        assert (record["text"], record["finish_reason"]) == ("An answer", "stop")
        assert record["usage"] == {
            "input_tokens": 120,
            "cached_tokens": 100,
            "cache_write_tokens": 0,
            "output_tokens": 7,
            "reasoning_tokens": 2,
            "cost_usd": 0.0012,
        }
        assert (record["route"], record["batch_id"]) == ("batch", "batch_1")
        assert record["origin"] == "router" and record["created"]
        # The prompt, from the pieces: cut where the planner cut it
        assert [m["role"] for m in record["messages"]] == ["system", "user"]
        assert [len(m["pieces"]) for m in record["messages"]] == [1, 3]
        prompt = ["".join(texts[h] for h in m["pieces"]) for m in record["messages"]]
        assert prompt == [m.text for m in call.messages]
        # The key, as documented
        content = {
            "messages": [
                [m["role"], text] for m, text in zip(record["messages"], prompt)
            ],
            "model": record["model"],
            "params": record["params"],
        }
        content = json.dumps(
            content, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        assert hashlib.sha256(content.encode()).hexdigest() == record["key"]
    (record,) = read_lines(path / files[2])
    assert (record["task"], record["group"], record["route"]) == (
        "generation",
        None,
        "live",
    )


def test_no_secrets_are_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-key")
    store = Store(tmp_path)
    (call,) = make_calls(1, 1)
    store.put(
        call, answer(call, raw={"headers": {"Authorization": "sk-or-secret-key"}})
    )
    content = "".join(f.read_text() for f in tmp_path.rglob("*") if f.is_file())
    assert "sk-or-secret-key" not in content and "Authorization" not in content


def test_later_answer_replaces_earlier(tmp_path):
    store = Store(tmp_path)
    (call,) = make_calls(1, 1)
    store.put(call, answer(call, "first"))
    store.put(call, answer(call, "second"))
    assert Store(tmp_path).result(call.key).text == "second"
    assert len(read_lines(store.file_for(call))) == 2
    with pytest.raises(ValueError):
        store.put(call, Result(call.key, error="failed"))


def test_interrupted_writes(tmp_path):
    """A line cut off by an interruption is dropped; the rest stays valid and
    later writes are not corrupted."""
    store = Store(tmp_path)
    a, b, c = make_calls(1, 3)
    store.put(a, answer(a, "a"))
    store.put(b, answer(b, "b"))
    calls_file = store.file_for(a)
    pieces_file = tmp_path / "pieces" / "part-000.jsonl"
    # Cut the last line of both files (the answer of b and its unique end)
    for file in (calls_file, pieces_file):
        content = file.read_bytes()
        file.write_bytes(content[: len(content) - 40])

    store = Store(tmp_path)
    assert a.key in store and b.key not in store
    assert store.call(a.key) == a
    store.put(b, answer(b, "b again"))
    store.put(c, answer(c, "c"))
    store = Store(tmp_path)
    assert [store.result(k.key).text for k in (a, b, c)] == ["a", "b again", "c"]
    assert [store.call(k.key) for k in (a, b, c)] == [a, b, c]
    assert len(read_lines(calls_file)) == 3
    hashes = [p["hash"] for p in read_lines(pieces_file)]
    assert len(hashes) == len(set(hashes))


def test_pieces_files_are_split(tmp_path, monkeypatch):
    monkeypatch.setattr("feedback_forensics.router.store.MAX_PIECES_FILE_BYTES", 2000)
    store = Store(tmp_path)
    calls = make_calls(groups=3, per_group=2)
    for call in calls:
        store.put(call, answer(call))
    parts = sorted(f.name for f in (tmp_path / "pieces").iterdir())
    assert parts[:2] == ["part-000.jsonl", "part-001.jsonl"]
    store = Store(tmp_path)
    assert [store.call(c.key) for c in calls] == calls
    extra = make_calls(groups=4, per_group=1)[-1]
    store.put(extra, answer(extra))
    assert Store(tmp_path).call(extra.key) == extra

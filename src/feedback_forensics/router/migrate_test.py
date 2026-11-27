"""Tests of the migration of generation files into the call store, on a small
fixture in the format written by `ff_modelgen`."""

import json

import pytest

from feedback_forensics.data.ap_format import hash_string
from feedback_forensics.router.datatypes import Call, Message, Params, Policy
from feedback_forensics.router.fakes import run_virtual
from feedback_forensics.router.migrate import main, migrate_generations
from feedback_forensics.router.store import Store
from feedback_forensics.router.testutils import make_router
from feedback_forensics.tools.ff_modelgen import get_failure, get_usage

MODEL = "openrouter/anthropic/claude-haiku-5.5"
OLD_MODEL = "openrouter/openai/gpt-old"


def live_record(prompt: str, response: str, finish_reason="stop") -> dict:
    """As written by `run_model_on_prompts_async` (a LangChain message)."""
    full_response = {
        "content": response,
        "response_metadata": {
            "token_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 300,
                "completion_tokens_details": {"reasoning_tokens": 100},
                "prompt_tokens_details": {"cached_tokens": 8},
                "cost": 0.0002,
            },
            "finish_reason": finish_reason,
            "id": "gen-1",
        },
        "usage_metadata": {"input_tokens": 20, "output_tokens": 300},
    }
    return {
        "prompt_id": hash_string(prompt),
        "prompt": prompt,
        "response": response,
        "full_response": full_response,
        "model": MODEL,
        "timestamp": "2026-11-20T10:00:00.123456",
        "failure": get_failure(response, full_response),
        "usage": get_usage(full_response),
        "generation_params": {"api": "live", "max_tokens": 10000, "temperature": 0.0},
        "provider": "Anthropic",
        "provider_model": "claude-haiku-5.5-20261001",
    }


def batch_record(prompt: str, response: str) -> dict:
    """As written by `collect_batch` (a chat completion)."""
    body = {
        "model": "anthropic/claude-haiku-5.5-20261001",
        "provider": "Anthropic",
        "choices": [
            {
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 40, "cost": 0.00001},
    }
    return {
        "prompt_id": hash_string(prompt),
        "prompt": prompt,
        "response": response,
        "full_response": body,
        "model": MODEL,
        "timestamp": "2026-11-21T08:30:00+00:00",
        "batch_id": "batch_abc",
        "failure": get_failure(response, body),
        "usage": get_usage(body),
        "generation_params": {"api": "batch", "max_tokens": 10000, "temperature": 0.0},
        "provider": "Anthropic",
        "provider_model": "anthropic/claude-haiku-5.5-20261001",
    }


def old_record(prompt: str, response: str) -> dict:
    """As written before usage and generation settings were saved."""
    return {
        "prompt_id": hash_string(prompt),
        "prompt": prompt,
        "response": response,
        "full_response": {"response_metadata": {"finish_reason": "stop"}},
        "model": OLD_MODEL,
    }


RECORDS = [
    live_record("Write a haiku.", "Leaves fall."),
    live_record("Write a novel.", "", finish_reason="length"),
    batch_record("Tell a joke.", "Why did the cache miss?"),
]


@pytest.fixture
def generations(tmp_path):
    directory = tmp_path / "data" / "generations"
    directory.mkdir(parents=True)
    lines = [json.dumps(record) for record in RECORDS]
    lines.insert(2, '{"prompt_id": "torn line')
    (directory / (MODEL.replace("/", "_") + ".jsonl")).write_text(
        "\n".join(lines) + "\n"
    )
    (directory / (OLD_MODEL.replace("/", "_") + ".jsonl")).write_text(
        json.dumps(old_record("Old prompt.", "Old response.")) + "\n\n"
    )
    return tmp_path / "data"


def test_migration(generations, tmp_path):
    before = {f: f.read_bytes() for f in generations.rglob("*.jsonl")}
    counts = migrate_generations(generations, tmp_path / "calls")
    assert counts == {"migrated": 4, "skipped": 0, "malformed": 1}
    assert {f: f.read_bytes() for f in generations.rglob("*.jsonl")} == before

    store_files = sorted(
        str(f.relative_to(tmp_path / "calls"))
        for f in (tmp_path / "calls").rglob("*")
        if f.is_file()
    )
    assert store_files == [
        "README.md",
        "generation/openrouter_anthropic_claude-haiku-5.5.jsonl",
        "generation/openrouter_openai_gpt-old.jsonl",
        "pieces/part-000.jsonl",
    ]
    lines = (tmp_path / "calls" / store_files[1]).read_text().splitlines()
    haiku, novel, joke = (json.loads(line) for line in lines)

    assert {r["origin"] for r in (haiku, novel, joke)} == {"migrated"}
    assert {r["task"] for r in (haiku, novel, joke)} == {"generation"}
    assert haiku["model"] == MODEL and haiku["group"] is None
    assert haiku["params"] == {
        "max_tokens": 10000,
        "reasoning_effort": None,
        "temperature": 0.0,
    }
    assert (haiku["text"], haiku["finish_reason"]) == ("Leaves fall.", "stop")
    assert haiku["usage"] == {
        "input_tokens": 20,
        "cached_tokens": 8,
        "cache_write_tokens": None,
        "output_tokens": 300,
        "reasoning_tokens": 100,
        "cost_usd": 0.0002,
    }
    assert (haiku["route"], haiku["batch_id"], haiku["provider"]) == (
        "live",
        None,
        "Anthropic",
    )
    assert haiku["created"] == "2026-11-20T10:00:00.123456"
    assert haiku["tag"] == {"prompt_id": hash_string("Write a haiku."), "failure": None}
    # A failed generation is an answer too (it was paid for, and is not retried)
    assert (novel["text"], novel["finish_reason"]) == ("", "length")
    assert novel["tag"]["failure"] == "length"
    assert (joke["route"], joke["batch_id"]) == ("batch", "batch_abc")
    assert (
        joke["usage"]["cost_usd"] == 0.00001 and joke["usage"]["cached_tokens"] is None
    )

    store = Store(tmp_path / "calls")
    # The key is that of the call that produces the generation
    call = Call(MODEL, (Message("user", "Write a haiku."),), Params(10000))
    assert haiku["key"] == call.key and store.call(call.key) == call
    # Older records do not say how they were generated
    (old,) = [
        store.record(k) for k in store.keys() if store.record(k)["model"] == OLD_MODEL
    ]
    assert old["params"] == {
        "max_tokens": None,
        "reasoning_effort": None,
        "temperature": None,
    }
    assert old["usage"]["input_tokens"] is None and old["route"] is None
    assert (old["text"], old["finish_reason"]) == ("Old response.", "stop")

    # Migrating again adds nothing
    assert migrate_generations(generations / "generations", tmp_path / "calls") == {
        "migrated": 0,
        "skipped": 4,
        "malformed": 1,
    }
    assert len((tmp_path / "calls" / store_files[1]).read_text().splitlines()) == 3


def test_router_reuses_migrated_generations(generations, tmp_path):
    main([str(generations), str(tmp_path / "calls")])
    router, fake = make_router(tmp_path / "calls")
    prompts = ["Write a haiku.", "Tell a joke.", "A new prompt."]
    calls = [Call(MODEL, (Message("user", p),), Params(10000)) for p in prompts]
    keys = router.add(calls, task="generation")
    report = run_virtual(router.arun(Policy()))
    assert (report.reused, report.sent) == (2, 1) and fake.sent == [keys[2]]
    results = router.collect(keys)
    assert results[keys[0]].text == "Leaves fall." and results[keys[0]].route == "store"
    assert results[keys[1]].text == "Why did the cache miss?"
    lines = (
        (
            tmp_path
            / "calls"
            / "generation"
            / "openrouter_anthropic_claude-haiku-5.5.jsonl"
        )
        .read_text()
        .splitlines()
    )
    assert [json.loads(line)["origin"] for line in lines] == ["migrated"] * 3 + [
        "router"
    ]


def test_no_generation_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        migrate_generations(tmp_path, tmp_path / "calls")

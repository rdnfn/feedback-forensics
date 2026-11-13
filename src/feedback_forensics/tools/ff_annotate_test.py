"""Tests for the ff-annotate CLI (argument handling, without annotating)."""

import json

import pytest

from feedback_forensics.tools import ff_annotate

PRINCIPLES = [
    "Select the response that is more concise",
    "Select the response that sounds more like a pirate",
]


@pytest.fixture
def annotate_calls(monkeypatch):
    """Record the arguments ff-annotate passes to annotate_dataset (no annotation)."""
    calls = []
    monkeypatch.setattr(
        ff_annotate, "annotate_dataset", lambda **kwargs: calls.append(kwargs)
    )
    return calls


def test_principles_list(annotate_calls):
    exit_code = ff_annotate.run(
        ["-d", "data.csv", "-v", "", "-p", json.dumps(PRINCIPLES)]
    )
    assert exit_code == 0
    assert annotate_calls[0]["principles"] == PRINCIPLES


def test_principles_from_icai_principles_file(tmp_path, annotate_calls):
    """The documented hand-off of principles discovered with ICAI: the content of its
    030_distilled_principles_per_cluster.json, a JSON object (id -> principle)."""
    principles_file = tmp_path / "030_distilled_principles_per_cluster.json"
    with open(principles_file, "w", encoding="utf-8") as f:
        json.dump(dict(enumerate(PRINCIPLES)), f, indent=4)  # as written by ICAI

    exit_code = ff_annotate.run(
        ["-d", "data.csv", "-v", "", "-p", principles_file.read_text()]
    )
    assert exit_code == 0
    assert annotate_calls[0]["principles"] == PRINCIPLES

    # Added to the default principles unless --principles-version is ""
    assert ff_annotate.run(["-d", "data.csv", "-p", principles_file.read_text()]) == 0
    assert annotate_calls[1]["principles"][-2:] == PRINCIPLES
    assert len(annotate_calls[1]["principles"]) == 40 + len(PRINCIPLES)


@pytest.mark.parametrize(
    "value",
    [
        '"Select the response that is more concise"',
        "42",
        "null",
        '[1, "Select the response that is more concise"]',
        '{"0": ["Select the response that is more concise"]}',
        "['Select the response that is more concise']",  # not JSON
    ],
)
def test_invalid_principles_rejected(value, annotate_calls, capsys):
    with pytest.raises(SystemExit) as exc_info:
        ff_annotate.run(["-d", "data.csv", "-p", value])
    assert exc_info.value.code == 2
    assert "argument -p/--principles" in capsys.readouterr().err
    assert not annotate_calls


@pytest.mark.parametrize("flag", ["--max-concurrent", "--max-output-tokens"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_limits_rejected(flag, value, annotate_calls, capsys):
    """E.g. --max-concurrent 0 made every API call wait forever."""
    with pytest.raises(SystemExit) as exc_info:
        ff_annotate.run(["-d", "data.csv", flag, value])
    assert exc_info.value.code == 2
    assert f"argument {flag}: must be a positive integer" in capsys.readouterr().err
    assert not annotate_calls


def test_limits_passed_on(annotate_calls):
    exit_code = ff_annotate.run(
        ["-d", "data.csv", "--max-concurrent", "1", "--max-output-tokens", "50"]
    )
    assert exit_code == 0
    assert annotate_calls[0]["max_concurrent_tasks"] == 1
    assert annotate_calls[0]["max_output_tokens"] == 50


@pytest.mark.parametrize(
    "args, error",
    [
        (
            ["-m", "anthropic/claude-x", "--reasoning-effort", "high"],
            "reasoning_effort is not supported for 'anthropic/' models",
        ),
        (["-m", "gemini-2.5-flash"], "Unsupported model name 'gemini-2.5-flash'"),
    ],
)
def test_unsupported_model_settings_rejected(args, error, annotate_calls, capsys):
    """Usage error before any output is written (not a traceback)."""
    with pytest.raises(SystemExit) as exc_info:
        ff_annotate.run(["-d", "data.csv", *args])
    assert exc_info.value.code == 2
    assert error in capsys.readouterr().err
    assert not annotate_calls


def test_reasoning_effort_passed_on(annotate_calls):
    model = "openrouter/anthropic/claude-x"
    exit_code = ff_annotate.run(
        ["-d", "data.csv", "-m", model, "--reasoning-effort", "high"]
    )
    assert exit_code == 0
    assert annotate_calls[0]["model"] == model
    assert annotate_calls[0]["reasoning_effort"] == "high"

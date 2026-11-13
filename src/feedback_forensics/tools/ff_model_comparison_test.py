"""Tests for ff-model-comparison (stubbed generation, mock annotator API)."""

import contextlib
import shutil
import sys

import pandas as pd
import pytest
from loguru import logger

import feedback_forensics.tools.ff_modelgen as modelgen
from feedback_forensics.annotation.testutils import mock_llm_server  # noqa: F401
from feedback_forensics.data.operations import load_ap, save_ap
from feedback_forensics.tools import ff_model_comparison

PROMPTS = [f"Prompt {i}: please explain topic {i}." for i in range(4)]
REFERENCE_MODEL = "openai/reference"
MODEL_X = "openai/model-x"
MODEL_Y = "openai/model-y"
MODEL_Z = "openai/model-z"


@pytest.fixture
def generations(monkeypatch):
    """Stub the generation stage (no API calls), records the generating models."""
    generating_models = []

    async def run_model_on_prompts_async(prompts, model_name, output_path, **kwargs):
        generating_models.append(model_name)

    def load_generations(output_path, model_name):
        responses = [
            f"{model_name}: {p}" + " Detail." * i for i, p in enumerate(PROMPTS)
        ]
        return pd.DataFrame({"prompt": PROMPTS, "response": responses})

    monkeypatch.setattr(
        modelgen, "run_model_on_prompts_async", run_model_on_prompts_async
    )
    monkeypatch.setattr(modelgen, "load_generations", load_generations)
    return generating_models


@contextlib.contextmanager
def log_warnings():
    """Warnings logged in the context."""
    warnings = []
    handler_id = logger.add(
        lambda m: warnings.append(m.record["message"]), level="WARNING"
    )
    try:
        yield warnings
    finally:
        logger.remove(handler_id)


def compare(output_path, models, **kwargs):
    ff_model_comparison.compare_models(
        prompts=PROMPTS,
        model_names=[REFERENCE_MODEL, *models],
        reference_models=[REFERENCE_MODEL],
        output_path=str(output_path),
        **kwargs,
    )


def test_rerun_with_other_annotation_settings_fails_before_api_calls(
    tmp_path, mock_llm_server, generations, monkeypatch
):
    server = mock_llm_server()
    compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-a")
    num_requests, num_generations = len(server.requests), len(generations)

    with pytest.raises(ValueError, match="other annotation settings") as exc_info:
        compare(
            tmp_path,
            [MODEL_X, MODEL_Y],
            annotator_model="openai/annotator-b",
            position_debias=True,
        )
    assert (
        "openai_model-x_ap.json: annotator_model='openai/annotator-a' (requested: "
        "'openai/annotator-b'), position_debias=False (requested: True)"
    ) in str(exc_info.value)
    assert len(server.requests) == num_requests
    assert len(generations) == num_generations
    assert not (tmp_path / "annotations" / "openai_model-y_ap.json").exists()

    # CLI: error message instead of a traceback
    monkeypatch.setattr(
        sys,
        "argv",
        ["ff-model-comparison", "-p", *PROMPTS, "-o", str(tmp_path)]
        + ["-m", REFERENCE_MODEL, MODEL_X, "-r", REFERENCE_MODEL]
        + ["--annotator-model", "openai/annotator-a", "--position-debias"],
    )
    assert ff_model_comparison.run() == 1
    assert len(server.requests) == num_requests


def test_all_existing_annotations_are_checked(tmp_path, mock_llm_server, generations):
    """Also annotations of models not compared in this run (all annotation files are
    merged), and an earlier combined dataset."""
    mock_llm_server()
    compare(tmp_path, [MODEL_X, MODEL_Y], annotator_model="openai/annotator-a")
    annotations_dir = tmp_path / "annotations"

    with pytest.raises(ValueError, match="other annotation settings") as exc_info:
        compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-b")
    for name in (
        "combined_ap.json",
        "openai_model-x_ap.json",
        "openai_model-y_ap.json",
    ):
        assert f"{annotations_dir / name}: annotator_model=" in str(exc_info.value)

    # Only the combined dataset is left over
    (annotations_dir / "openai_model-x_ap.json").unlink()
    (annotations_dir / "openai_model-y_ap.json").unlink()
    with pytest.raises(ValueError, match="other annotation settings") as exc_info:
        compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-b")
    assert f"{annotations_dir / 'combined_ap.json'}: " in str(exc_info.value)


def test_rerun_with_same_settings_reuses_annotations(
    tmp_path, mock_llm_server, generations
):
    server = mock_llm_server()
    settings = {"annotator_model": "openai/annotator-a", "position_debias": True}
    compare(tmp_path, [MODEL_X], **settings)
    server.reset()

    compare(tmp_path, [MODEL_X, MODEL_Y], **settings)
    # Only the new model's comparisons were annotated (in both response orders)
    assert len(server.requests) == 2 * len(PROMPTS)
    combined = load_ap(tmp_path / "annotations" / "combined_ap.json")
    assert {c["response_a"]["model"] for c in combined["comparisons"]} == {
        MODEL_X,
        MODEL_Y,
    }
    assert combined["metadata"]["annotation"] == {
        **settings,
        "max_output_tokens": 3000,
        "reasoning_effort": None,
    }

    # API usage of each annotation file (both orders per comparison) and in total
    usage_per_file = combined["metadata"]["annotation_usage_per_file"]
    assert list(usage_per_file) == ["openai_model-x_ap.json", "openai_model-y_ap.json"]
    for name, usage in usage_per_file.items():
        assert usage["num_calls"] == 2 * len(PROMPTS)
        file_ap = load_ap(tmp_path / "annotations" / name)
        assert file_ap["metadata"]["annotation_usage"] == usage
    usage = combined["metadata"]["annotation_usage"]
    assert usage["num_annotation_runs"] == 2
    assert usage["num_calls"] == 4 * len(PROMPTS)
    assert usage["cost_complete"] is True


def test_earlier_combined_dataset_is_not_merged(tmp_path, mock_llm_server, generations):
    """Its votes could be outdated (e.g. of a deleted annotation file) and would take
    precedence over the annotation files' votes when merged first."""
    mock_llm_server()
    compare(tmp_path, [MODEL_X])
    combined_path = tmp_path / "annotations" / "combined_ap.json"
    outdated = load_ap(combined_path)
    outdated["comparisons"].append({**outdated["comparisons"][0], "id": "outdated"})
    save_ap(outdated, combined_path)

    compare(tmp_path, [MODEL_X])
    combined = load_ap(combined_path)
    assert "outdated" not in {c["id"] for c in combined["comparisons"]}
    assert len(combined["comparisons"]) == len(PROMPTS)


def unparsable_response(handle, body):
    status, payload = handle(body)
    payload["choices"][0]["message"]["content"] = "No vote."
    return status, payload


@pytest.mark.parametrize(
    "make_response, reported",
    [
        (
            lambda handle, body: (
                402,
                {"error": {"message": "Insufficient credits", "code": 402}},
            ),
            f"{len(PROMPTS)} API calls failed",
        ),
        (unparsable_response, f"{len(PROMPTS)} had an unparsable response"),
    ],
    ids=["out_of_credits", "unparsable"],
)
def test_annotation_calls_without_votes_reported(
    tmp_path, mock_llm_server, generations, monkeypatch, make_response, reported
):
    """Annotations with failed API calls (e.g. out of credits) or unparsable
    responses are saved, and reused by later runs: the warning tells how to repeat
    these calls (not by re-running, as annotate_dataset's warning suggests)."""
    server = mock_llm_server()
    handle = server._handle
    monkeypatch.setattr(server, "_handle", lambda body: make_response(handle, body))
    annotation_path = tmp_path / "annotations" / "openai_model-x_ap.json"

    with log_warnings() as warnings:
        compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-a")
    assert any(
        reported in w and f"delete {annotation_path} and re-run" in w for w in warnings
    )

    monkeypatch.setattr(server, "_handle", handle)  # provider works again
    server.reset()
    compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-a")
    assert server.requests == []  # reused
    annotation_path.unlink()
    compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-a")
    assert len(server.requests) == len(PROMPTS)  # failed calls repeated


def make_icai_era_annotations(output_path, server, monkeypatch) -> None:
    """Annotation files (of MODEL_X and MODEL_Y) as created by the former ICAI-based
    pipeline: without `metadata.annotation`. The default annotator (an openrouter/
    model) uses the mock API too."""
    compare(output_path, [MODEL_X, MODEL_Y], annotator_model="openai/annotator-a")
    for path in (output_path / "annotations").glob("*.json"):
        ap = load_ap(path)
        del ap["metadata"]["annotation"]
        if path.name == "openai_model-y_ap.json":
            ap["metadata"]["annotation"] = None  # also no information
        save_ap(ap, path)
    shutil.rmtree(output_path / "tmp")  # as in the workflow
    monkeypatch.setattr(
        "feedback_forensics.models.OPENROUTER_API_BASE", server.base_url
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
    monkeypatch.setattr(
        "feedback_forensics.annotation.usage.get_openrouter_pricing",
        lambda model_id: None,  # no list prices (network)
    )
    server.reset()


def test_icai_era_annotations_reused_with_default_settings(
    tmp_path, mock_llm_server, generations, monkeypatch
):
    """Annotation files without `metadata.annotation` were created by the former
    ICAI-based pipeline with its default settings (e.g. the model personality
    dataset that .github/workflows/update-model-comparison.yml updates)."""
    server = mock_llm_server()
    make_icai_era_annotations(tmp_path, server, monkeypatch)

    # As in the workflow: default settings, a model added
    with log_warnings() as warnings:
        compare(tmp_path, [MODEL_X, MODEL_Y, MODEL_Z])
    # Only the new model's comparisons were annotated
    assert len(server.requests) == len(PROMPTS)
    assert {r["model"] for r in server.requests} == {"google/gemini-2.5-flash"}
    assert not [w for w in warnings if "annotation settings" in w]  # when merging
    combined = load_ap(tmp_path / "annotations" / "combined_ap.json")
    assert {c["response_a"]["model"] for c in combined["comparisons"]} == {
        MODEL_X,
        MODEL_Y,
        MODEL_Z,
    }
    assert combined["metadata"]["annotation"] == {
        "annotator_model": "openrouter/google/gemini-2.5-flash",
        "position_debias": False,
        "max_output_tokens": 3000,
        "reasoning_effort": None,
    }

    with pytest.raises(ValueError, match="other annotation settings"):
        compare(tmp_path, [MODEL_X, MODEL_Y], position_debias=True)


@pytest.mark.parametrize(
    "settings",
    [{"position_debias": True}, {"annotator_model": "openai/annotator-b"}],
    ids=["position_debias", "annotator_model"],
)
def test_icai_era_annotations_with_other_settings_fail_before_api_calls(
    tmp_path, mock_llm_server, generations, monkeypatch, settings
):
    """ICAI-era annotation files only (no new-format files): they count as
    annotated with the default settings."""
    server = mock_llm_server()
    make_icai_era_annotations(tmp_path, server, monkeypatch)
    num_generations = len(generations)

    with pytest.raises(ValueError, match="other annotation settings") as exc_info:
        compare(tmp_path, [MODEL_X, MODEL_Y, MODEL_Z], **settings)
    for name in (
        "combined_ap.json",
        "openai_model-x_ap.json",
        "openai_model-y_ap.json",
    ):
        assert f"{tmp_path / 'annotations' / name}: " in str(exc_info.value)
    assert server.requests == []
    assert len(generations) == num_generations
    assert not (tmp_path / "annotations" / "openai_model-z_ap.json").exists()

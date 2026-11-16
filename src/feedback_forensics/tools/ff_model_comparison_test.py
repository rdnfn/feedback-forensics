"""Tests for ff-model-comparison (stubbed generation, mock annotator API)."""

import contextlib
import shutil
import sys

import pandas as pd
import pytest
from loguru import logger

import feedback_forensics.tools.ff_modelgen as modelgen
from feedback_forensics.annotation import annotate_dataset
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
    compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-a")
    combined_path = tmp_path / "annotations" / "combined_ap.json"
    outdated = load_ap(combined_path)
    outdated["comparisons"].append({**outdated["comparisons"][0], "id": "outdated"})
    save_ap(outdated, combined_path)

    compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-a")
    combined = load_ap(combined_path)
    assert "outdated" not in {c["id"] for c in combined["comparisons"]}
    assert len(combined["comparisons"]) == len(PROMPTS)


def unparsable_response(handle, body):
    status, payload = handle(body)
    payload["choices"][0]["message"]["content"] = "No vote."
    return status, payload


def test_annotations_with_unparsable_responses_saved_and_reported(
    tmp_path, mock_llm_server, generations, monkeypatch
):
    """Unparsable (or truncated) responses are the annotator's behaviour: the
    annotations are saved, and reused by later runs. The warning tells how to repeat
    these calls (not by re-running, as annotate_dataset's warning suggests)."""
    server = mock_llm_server()
    handle = server._handle
    monkeypatch.setattr(
        server, "_handle", lambda body: unparsable_response(handle, body)
    )
    annotation_path = tmp_path / "annotations" / "openai_model-x_ap.json"

    with log_warnings() as warnings:
        compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-a")
    assert any(
        f"{len(PROMPTS)} had an unparsable response" in w
        and f"delete {annotation_path} and re-run" in w
        for w in warnings
    )

    monkeypatch.setattr(server, "_handle", handle)  # parsable responses again
    server.reset()
    compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-a")
    assert server.requests == []  # reused
    annotation_path.unlink()
    compare(tmp_path, [MODEL_X], annotator_model="openai/annotator-a")
    assert len(server.requests) == len(PROMPTS)  # unparsable calls repeated


def annotated_model(body) -> str:
    """Model (X, Y or Z) whose comparison a request to the mock annotator is for."""
    content = str(body["messages"])
    return next(m for m in (MODEL_X, MODEL_Y, MODEL_Z) if f"{m}: Prompt" in content)


def test_out_of_credits_stops_annotation_without_saving(
    tmp_path, mock_llm_server, generations, monkeypatch
):
    """Regression: when the credits ran out, all remaining annotation calls failed,
    and the annotation files (only invalid votes) were saved as a successful run,
    and reused (never annotated again) by later runs."""
    server = mock_llm_server()
    handle = server._handle
    num_ok = len(PROMPTS) + 1  # credits run out while annotating the second file
    num_calls = []

    def out_of_credits(body):
        num_calls.append(annotated_model(body))
        if len(num_calls) <= num_ok:
            return handle(body)
        return 402, {"error": {"message": "Insufficient credits", "code": 402}}

    monkeypatch.setattr(server, "_handle", out_of_credits)
    monkeypatch.setattr(  # one call at a time: exactly one is rejected
        ff_model_comparison,
        "annotate_dataset",
        lambda **kwargs: annotate_dataset(**kwargs, max_concurrent_tasks=1),
    )
    annotations_dir = tmp_path / "annotations"
    models = [MODEL_X, MODEL_Y, MODEL_Z]
    args = {"annotator_model": "openai/annotator-a"}

    with pytest.raises(
        ff_model_comparison.AnnotationFailedError, match="were not saved"
    ) as exc_info:
        compare(tmp_path, models, **args)

    # Stopped at the first rejected call: no calls for the third file
    assert len(num_calls) == num_ok + 1
    first, second = list(dict.fromkeys(num_calls))
    (third,) = set(models) - {first, second}
    # Only the complete file was saved, the others are listed in the error
    saved = f"{first.replace('/', '_')}_ap.json"
    assert [p.name for p in annotations_dir.glob("*.json")] == [saved]
    message = str(exc_info.value)
    assert saved not in message
    assert (
        f"{annotations_dir / (second.replace('/', '_') + '_ap.json')}: stopped by "
        "API key or account error (HTTP 402)"
    ) in message
    assert (
        f"{annotations_dir / (third.replace('/', '_') + '_ap.json')}: not annotated"
    ) in message

    # Next run (credits added): only the missing calls are made
    monkeypatch.setattr(server, "_handle", handle)
    server.reset()
    compare(tmp_path, models, **args)
    assert len(server.requests) == 3 * len(PROMPTS) - num_ok
    combined = load_ap(annotations_dir / "combined_ap.json")
    assert {c["response_a"]["model"] for c in combined["comparisons"]} == set(models)
    # All calls behind the saved annotations succeeded, except the rejected one
    usage = combined["metadata"]["annotation_usage"]
    assert usage["num_calls"] == 3 * len(PROMPTS) + 1
    assert usage["num_failed_calls"] == 1


def test_out_of_credits_cli_fails(tmp_path, mock_llm_server, generations, monkeypatch):
    """Exit code 1 (the workflow then saves partial results to the 'wip' branch
    instead of reporting success), error message instead of a traceback."""
    server = mock_llm_server()
    monkeypatch.setattr(
        server,
        "_handle",
        lambda body: (402, {"error": {"message": "Insufficient credits", "code": 402}}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["ff-model-comparison", "-p", *PROMPTS, "-o", str(tmp_path)]
        + ["-m", REFERENCE_MODEL, MODEL_X, "-r", REFERENCE_MODEL]
        + ["--annotator-model", "openai/annotator-a", "--position-debias"],
    )
    assert ff_model_comparison.run() == 1
    assert not list((tmp_path / "annotations").glob("*.json"))


def test_annotations_with_failed_calls_not_saved(
    tmp_path, mock_llm_server, generations, monkeypatch
):
    """Failed API calls other than account errors (here: requests rejected by the
    provider) do not stop the run: the other files are annotated and saved, the
    file with failed calls is not, and the run fails."""
    server = mock_llm_server()
    handle = server._handle

    def fail_one_call_of_model_y(body):
        if annotated_model(body) == MODEL_Y and "Prompt 2" in str(body["messages"]):
            return 400, {"error": {"message": "Provider returned error", "code": 400}}
        return handle(body)

    monkeypatch.setattr(server, "_handle", fail_one_call_of_model_y)
    annotations_dir = tmp_path / "annotations"
    models = [MODEL_X, MODEL_Y, MODEL_Z]
    args = {"annotator_model": "openai/annotator-a"}

    with pytest.raises(ff_model_comparison.AnnotationFailedError) as exc_info:
        compare(tmp_path, models, **args)
    assert (
        f"{annotations_dir / 'openai_model-y_ap.json'}: 1 of {len(PROMPTS)} API calls "
        "failed"
    ) in str(exc_info.value)
    assert "model-x" not in str(exc_info.value)
    assert sorted(p.name for p in annotations_dir.glob("*.json")) == [
        "openai_model-x_ap.json",
        "openai_model-z_ap.json",
    ]

    # Next run: only calls of this file are repeated, and only those without a
    # valid vote (the failed call and, depending on the mock, unparsable ones)
    monkeypatch.setattr(server, "_handle", handle)
    server.reset()
    compare(tmp_path, models, **args)
    assert {annotated_model(body) for body in server.requests} == {MODEL_Y}
    assert any("Prompt 2" in str(body["messages"]) for body in server.requests)
    assert len(server.requests) < len(PROMPTS)
    assert (annotations_dir / "openai_model-y_ap.json").exists()
    assert (annotations_dir / "combined_ap.json").exists()


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

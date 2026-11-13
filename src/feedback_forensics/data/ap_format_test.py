"""Tests for the AnnotatedPairs format module.

The golden values pin the ids of comparisons and annotators: published datasets and
ICAI-era annotations depend on them. They were computed with (and are identical to)
`inverse_cai.data.annotated_pairs_format` (inverse-cai v0.4.8), but these tests do not
need inverse-cai. Never update a golden value to make a test pass.
"""

import pandas as pd
import pytest

from feedback_forensics.data.ap_format import (
    DEFAULT_ANNOTATOR_DESCRIPTION,
    create_annotated_pairs,
    hash_comparison,
    hash_string,
    load_preference_data,
)


@pytest.mark.parametrize(
    "string, expected",
    [
        ("test", "098f6bcd"),
        ("", "d41d8cd9"),
        ("Select the response that is more concise", "bf731c7f"),
        (DEFAULT_ANNOTATOR_DESCRIPTION, "ba751e7b"),
        ("column_human_label", "5a47042f"),
        ("ünïcödé 🎭", "7c1244a4"),
        # Hashed as is: no stripping or Unicode normalisation
        (" Select the response that is more concise ", "a417f244"),
        ("line\nbreak\ttab", "247f2cf2"),
        ("caf\u00e9", "07117fe4"),
        ("cafe\u0301", "10a85865"),  # decomposed "é"
    ],
)
def test_hash_string_golden_values(string, expected):
    assert hash_string(string) == expected


@pytest.mark.parametrize(
    "response_a, response_b, prompt, expected",
    [
        ({"text": "response A"}, {"text": "response B"}, None, "8d8949e1"),
        (
            {"text": "response A"},
            {"text": "response B"},
            "What is the capital of France?",
            "49897fb3",
        ),
        # An empty prompt is part of the hash, unlike a missing prompt
        ({"text": "a"}, {"text": "b"}, "", "2502b911"),
        (
            {"text": "response A", "model": "Model X"},
            {"text": "response B", "model": "Model Y"},
            None,
            "a72933af",
        ),
        (
            {"text": "response B", "model": "Model Y"},
            {"text": "response A", "model": "Model X"},
            None,
            "1ba991f6",
        ),
        # All fields are hashed, independent of key order
        (
            {
                "model": "Model X",
                "text": "response A",
                "metadata": {"timestamp": "2023-01-01", "a": 1},
            },
            {"text": "response B"},
            "Prompt",
            "801199fc",
        ),
        ({"text": "Grüße 🎭"}, {"text": "Ciao"}, "Prompt ünïcödé", "c1ab200c"),
    ],
)
def test_hash_comparison_golden_values(response_a, response_b, prompt, expected):
    assert hash_comparison(response_a, response_b, prompt) == expected


def test_create_annotated_pairs_ids_golden_values():
    df = pd.DataFrame(
        {
            "input": ["What is 2+2?", "Say hi"],
            "text_a": ["4", "Hi!"],
            "text_b": ["Four", "Hello 🎭"],
            "model_a": ["model-x", "model-x"],
            "model_b": ["model-y", "model-z"],
            "preferred_text": ["text_a", "text_b"],
            "human_label": ["text_b", "text_b"],
        }
    )
    ap = create_annotated_pairs(
        df=df,
        dataset_name="test",
        principle_index_to_text={
            0: "Select the response that is more concise",
            1: "Select the response that is more polite",
        },
        comparison_votes={0: {0: True, 1: None}, 1: {0: False, 1: "invalid"}},
    )

    assert ap["metadata"]["default_annotator"] == "ba751e7b"
    assert {
        annotator_id: annotator.get("name") or annotator["description"]
        for annotator_id, annotator in ap["annotators"].items()
    } == {
        "ba751e7b": "preferred_text",
        "bf731c7f": "Select the response that is more concise",
        "2e2d4304": "Select the response that is more polite",
        "5a47042f": "human_label",
    }
    assert [comparison["id"] for comparison in ap["comparisons"]] == [
        "6a90b9a0",
        "ff47a8b9",
    ]
    assert ap["comparisons"][1]["annotations"] == {
        "ba751e7b": {"pref": "b"},
        "bf731c7f": {"pref": "a"},
        "2e2d4304": {"pref": None, "no_pref_reason": "invalid"},
        "5a47042f": {"pref": "text_b"},
    }


def test_create_annotated_pairs_ids_of_loaded_data_golden_values(tmp_path):
    """Ids of data as loaded for annotation: prompts are merged into the texts sent to
    the annotator, but the ids use the original texts (and no prompt, as the column
    is "prompt", not "input"). A missing model counts as model "nan"."""
    pd.DataFrame(
        {
            "prompt": ["What is 2+2?", "Say hi"],
            "text_a": ["4", "Hi!"],
            "text_b": ["Four", "Hello 🎭"],
            "model_a": ["model-x", "model-x"],
            "model_b": ["model-y", None],
            "preferred_text": ["text_a", "text_b"],
        }
    ).to_csv(tmp_path / "data.csv", index=False)
    df = load_preference_data(tmp_path / "data.csv")  # as by annotate_dataset
    assert df.loc[0, "text_a"] != "4"  # prompt merged into the texts

    ap = create_annotated_pairs(df=df, dataset_name="test")

    assert [comparison["id"] for comparison in ap["comparisons"]] == [
        "e6508ec7",
        "c9774be1",
    ]
    assert ap["comparisons"][1]["response_b"] == {"text": "Hello 🎭", "model": "nan"}

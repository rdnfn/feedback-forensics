"""Tests for dataset_utils.py"""

import pandas as pd

from feedback_forensics.app.dataset_utils import (
    add_annotators_to_votes_dict,
    get_annotators_by_type,
    get_available_models,
)


def test_get_available_models_missing_columns():
    """Test get_available_models with a dataframe missing required columns."""
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    models = get_available_models(df)
    assert models == []


def test_get_available_models_with_data():
    """Test get_available_models with valid model data."""
    df = pd.DataFrame(
        {
            "model_a": ["gpt4", "claude", "llama", "gpt4", None],
            "model_b": ["claude", "gpt4", None, "mistral", "mistral"],
            "other_col": [1, 2, 3, 4, 5],
        }
    )
    models = get_available_models(df)
    assert sorted(models) == ["claude", "gpt4", "llama", "mistral"]


def test_add_annotators_to_votes_dict():
    """Test add_annotators_to_votes_dict functionality."""
    df = pd.DataFrame(
        {
            "comparison_id": [1, 2, 3],
            "text_a": ["a1", "a2", "a3"],
            "text_b": ["b1", "b2", "b3"],
            "existing_annotator": ["text_a", "text_b", "text_a"],
        }
    )

    votes_dict = {
        "df": df,
        "annotator_metadata": {
            "existing_annotator": {
                "variant": "test",
                "annotator_visible_name": "Existing Annotator",
            }
        },
        "reference_annotator_col": "existing_annotator",
        "shown_annotator_rows": [],
    }

    annotations_df = pd.DataFrame(
        {
            "comparison_id": [1, 2, 3],
            "new_annotator1": ["text_b", "text_a", "text_b"],
            "new_annotator2": ["text_a", "text_a", "text_a"],
        }
    )

    new_annotator_metadata = {
        "new_annotator1": {
            "variant": "test_new",
            "annotator_visible_name": "New Annotator 1",
        },
        "new_annotator2": {
            "variant": "test_new",
            "annotator_visible_name": "New Annotator 2",
        },
    }

    result = add_annotators_to_votes_dict(
        votes_dict, new_annotator_metadata, annotations_df
    )

    assert "new_annotator1" in result["df"].columns
    assert "new_annotator2" in result["df"].columns
    assert "new_annotator1" in result["annotator_metadata"]
    assert "new_annotator2" in result["annotator_metadata"]
    assert "existing_annotator" in result["annotator_metadata"]
    assert result["annotator_metadata"]["new_annotator1"]["variant"] == "test_new"
    assert result["df"]["new_annotator1"].iloc[0] == "text_b"

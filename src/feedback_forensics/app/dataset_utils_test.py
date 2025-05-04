"""Tests for dataset_utils.py"""

import pandas as pd

from feedback_forensics.app.dataset_utils import (
    add_annotators_to_votes_dict,
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

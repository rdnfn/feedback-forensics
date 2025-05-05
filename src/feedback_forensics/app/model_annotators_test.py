"""Tests for model_annotators.py"""

import pandas as pd
import pytest
from inverse_cai.data.annotated_pairs_format import hash_string

from feedback_forensics.app.model_annotators import generate_model_identity_annotators
from feedback_forensics.app.constants import (
    PREFIX_MODEL_IDENTITY_ANNOTATORS,
    MODEL_IDENTITY_ANNOTATOR_TYPE,
)


def create_test_dataframe():
    """Create a test dataframe with model comparison data."""
    return pd.DataFrame(
        {
            "comparison_id": [1, 2, 3, 4, 5],
            "model_a": ["gpt4", "claude", "llama", "gpt4", "mistral"],
            "model_b": ["claude", "gpt4", "mistral", "llama", "gpt4"],
            "text_a": ["text1", "text2", "text3", "text4", "text5"],
            "text_b": ["textA", "textB", "textC", "textD", "textE"],
            "preferred_text": ["text_a", "text_b", "text_a", "text_b", "text_a"],
        }
    )


def create_test_votes_dict():
    """Create a test votes_dict with dataframe and metadata."""
    df = create_test_dataframe()
    return {
        "df": df,
        "shown_annotator_rows": ["annotator1", "annotator2"],
        "annotator_metadata": {
            "annotator1": {
                "variant": "test_annotator",
                "annotator_visible_name": "Test Annotator 1",
            },
            "annotator2": {
                "variant": "test_annotator",
                "annotator_visible_name": "Test Annotator 2",
            },
        },
        "reference_annotator_col": "preferred_text",
    }


def test_generate_model_identity_annotators_default_one_vs_all():
    """Test generate_model_identity_annotators with default one-vs-all behavior."""
    df = create_test_dataframe()

    # Call with no reference_models should default to one-vs-all
    metadata, df_result = generate_model_identity_annotators(df, [], [])

    cols = list(metadata.keys())

    assert len(cols) == 4  # gpt4, claude, llama, mistral

    for col in cols:
        assert metadata[col]["variant"] == MODEL_IDENTITY_ANNOTATOR_TYPE

    gpt4_annotator_id = hash_string("model_identity_gpt4_over_references")
    if gpt4_annotator_id in cols:
        # In row 0, gpt4 is model_a, so should be preferred (text_a)
        assert df_result.iloc[0][gpt4_annotator_id] == "text_a"
        # In row 4, gpt4 is model_b, so should be preferred (text_b)
        assert df_result.iloc[4][gpt4_annotator_id] == "text_b"


def test_generate_model_identity_annotators_with_specific_reference_models():
    """Test generate_model_identity_annotators with specific reference models."""
    df = create_test_dataframe()

    # Use only 'claude' and 'llama' as reference models
    reference_models = ["claude", "llama"]
    metadata, df_result = generate_model_identity_annotators(df, [], reference_models)

    cols = list(metadata.keys())

    assert len(cols) == 4  # gpt4, claude, llama, mistral

    # Verify the reference models in the metadata
    for col in cols:
        assert metadata[col]["variant"] == MODEL_IDENTITY_ANNOTATOR_TYPE
        assert set(metadata[col]["reference_models"]) == set(reference_models)

    # Check that gpt4's annotator works correctly
    gpt4_annotator_id = hash_string("model_identity_gpt4_over_references")
    if gpt4_annotator_id in cols:
        # In row 0, gpt4 is model_a and claude is model_b (a reference), so should be preferred (text_a)
        assert df_result.iloc[0][gpt4_annotator_id] == "text_a"
        # In row 3, gpt4 is model_a and llama is model_b (a reference), so should be preferred (text_a)
        assert df_result.iloc[3][gpt4_annotator_id] == "text_a"
        # Rows where the comparison doesn't involve reference models should be "Not applicable"
        assert (
            df_result.iloc[4][gpt4_annotator_id] == "Not applicable"
        )  # gpt4 vs mistral

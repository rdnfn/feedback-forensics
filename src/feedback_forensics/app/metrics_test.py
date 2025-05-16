"""
Tests for the metrics module.
"""

import pandas as pd
import numpy as np
from feedback_forensics.app.metrics import (
    get_agreement,
    get_acc,
    get_relevance,
    get_principle_strength,
    get_cohens_kappa,
    get_cohens_kappa_randomized,
    get_num_votes,
    get_agreed,
    get_disagreed,
    get_not_applicable,
    compute_annotator_metrics,
)


def test_get_agreement():
    """Test agreement calculation for different vote distributions."""
    # Test with all vote types
    value_counts = pd.Series({"Agree": 3, "Disagree": 2, "Not applicable": 1})
    assert get_agreement(value_counts) == 0.5  # 3 / (3 + 2 + 1)

    # Test with no votes
    value_counts = pd.Series({"Not applicable": 5})
    assert get_agreement(value_counts) == 0

    # Test with only agrees
    value_counts = pd.Series({"Agree": 5})
    assert get_agreement(value_counts) == 1.0


def test_get_acc():
    """Test accuracy calculation for different vote distributions."""
    # Test with agree and disagree votes
    value_counts = pd.Series({"Agree": 3, "Disagree": 1})
    assert get_acc(value_counts) == 0.75  # 3 / (3 + 1)

    # Test with only agrees
    value_counts = pd.Series({"Agree": 5})
    assert get_acc(value_counts) == 1.0

    # Test with only disagrees
    value_counts = pd.Series({"Disagree": 5})
    assert get_acc(value_counts) == 0.0

    # Test with no relevant votes
    value_counts = pd.Series({"Not applicable": 5})
    assert (
        get_acc(value_counts) == 0.5
    )  # When no relevant votes, returns 0.5 (chance level)


def test_get_relevance():
    """Test relevance calculation for different vote distributions."""
    # Test with all vote types
    value_counts = pd.Series({"Agree": 3, "Disagree": 2, "Not applicable": 1})
    assert np.isclose(
        get_relevance(value_counts), 0.833, rtol=1e-3
    )  # (3 + 2) / (3 + 2 + 1), rounded to 3 decimals

    # Test with no not applicable votes
    value_counts = pd.Series({"Agree": 3, "Disagree": 2})
    assert get_relevance(value_counts) == 1.0

    # Test with only not applicable votes
    value_counts = pd.Series({"Not applicable": 5})
    assert get_relevance(value_counts) == 0.0


def test_get_principle_strength():
    """Test principle strength calculation for different vote distributions."""
    # Test with perfect performance
    value_counts = pd.Series({"Agree": 4, "Disagree": 0, "Not applicable": 1})
    expected = (1.0 - 0.5) * (4 / 5) * 2  # (acc - 0.5) * relevance * 2
    assert get_principle_strength(value_counts) == expected

    # Test with worst performance
    value_counts = pd.Series({"Agree": 0, "Disagree": 4, "Not applicable": 1})
    expected = (0.0 - 0.5) * (4 / 5) * 2
    assert get_principle_strength(value_counts) == expected

    # Test with neutral performance
    value_counts = pd.Series({"Agree": 2, "Disagree": 2, "Not applicable": 1})
    assert get_principle_strength(value_counts) == 0.0


def test_get_cohens_kappa_randomized():
    """Test Cohen's kappa calculation for different vote distributions."""
    # Test with perfect agreement
    value_counts = pd.Series({"Agree": 5, "Disagree": 0, "Not applicable": 0})
    assert get_cohens_kappa_randomized(value_counts) == 1.0  # 2 * (1.0 - 0.5)

    # Test with perfect disagreement
    value_counts = pd.Series({"Agree": 0, "Disagree": 5, "Not applicable": 0})
    assert get_cohens_kappa_randomized(value_counts) == -1.0  # 2 * (0.0 - 0.5)

    # Test with random performance (equal agree/disagree)
    value_counts = pd.Series({"Agree": 3, "Disagree": 3, "Not applicable": 2})
    assert get_cohens_kappa_randomized(value_counts) == 0.0  # 2 * (0.5 - 0.5)

    # Test with 75% agreement
    value_counts = pd.Series({"Agree": 3, "Disagree": 1, "Not applicable": 0})
    assert get_cohens_kappa_randomized(value_counts) == 0.5  # 2 * (0.75 - 0.5)

    # Test with no relevant votes
    value_counts = pd.Series({"Not applicable": 5})
    assert (
        get_cohens_kappa_randomized(value_counts) == 0.0
    )  # Returns 0 for no relevant votes


def test_get_cohens_kappa():
    """Test Cohen's kappa calculation for different vote distributions."""
    # Test with perfect agreement
    annotations_a = ["text_b", "text_a", "text_a", "text_a", "text_a"]
    annotations_b = ["text_b", "text_a", "text_a", "text_a", "text_a"]
    assert (
        get_cohens_kappa(None, annotation_a=annotations_a, annotation_b=annotations_b)
        == 1.0
    )

    # Test with perfect disagreement
    annotations_a = ["text_b", "text_a", "text_b", "text_a", "text_a", "text_b"]
    annotations_b = ["text_a", "text_b", "text_a", "text_b", "text_b", "text_a"]
    assert (
        get_cohens_kappa(None, annotation_a=annotations_a, annotation_b=annotations_b)
        == -1.0
    )

    # Test with random performance
    annotations_a = ["text_a", "text_a", "text_a", "text_a", "text_a"]
    annotations_b = ["text_b", "text_b", "text_b", "text_b", "text_b"]
    assert (
        get_cohens_kappa(
            None,
            annotation_a=annotations_a,
            annotation_b=annotations_b,
        )
        == 0.0
    )

    # Test with no relevant votes
    annotations_a = ["text_a", "text_b", "text_b", "text_a", "text_b"]
    annotations_b = ["text_a", "text_a", "text_b", "text_a", "text_a"]
    assert (
        get_cohens_kappa(None, annotation_a=annotations_a, annotation_b=annotations_b)
        > 0.0
    )


def test_vote_count_functions():
    """Test basic vote counting functions."""
    value_counts = pd.Series({"Agree": 3, "Disagree": 2, "Not applicable": 1})

    assert get_num_votes(value_counts) == 6
    assert get_agreed(value_counts) == 3
    assert get_disagreed(value_counts) == 2
    assert get_not_applicable(value_counts) == 1


def test_compute_metrics():
    """Test metric computation with sample vote data."""
    # Create a test dataframe with the proper structure
    votes_df = pd.DataFrame(
        {
            "comparison_id": [1, 2],
            "text_a": ["text A1", "text A2"],
            "text_b": ["text B1", "text B2"],
            "preferred_text": ["text_a", "text_b"],  # Reference annotator column
            "p1": ["text_a", "text_a"],  # First principle/annotator
            "p2": ["text_a", "text_b"],  # Second principle/annotator
        }
    )

    for col in ["preferred_text", "p1", "p2"]:
        votes_df[col] = votes_df[col].astype("category")

    votes_dict = {
        "df": votes_df,
        "shown_annotator_rows": ["p1", "p2"],
        "annotator_metadata": {
            "p1": {
                "annotator_in_row_name": "p1",
                "annotator_visible_name": "Principle 1",
            },
            "p2": {
                "annotator_in_row_name": "p2",
                "annotator_visible_name": "Principle 2",
            },
        },
        "reference_annotator_col": "preferred_text",
    }

    metrics = compute_annotator_metrics(
        votes_df=votes_dict["df"],
        annotator_metadata=votes_dict["annotator_metadata"],
        annotator_cols=votes_dict["shown_annotator_rows"],
        ref_annotator_col=votes_dict["reference_annotator_col"],
    )

    # Check structure
    assert "annotator_names" in metrics
    assert "num_pairs" in metrics
    assert "metrics" in metrics

    # Check annotator names
    assert set(metrics["annotator_names"]) == {"p1", "p2"}
    assert metrics["num_pairs"] == 2

    # Check metrics for p1
    p1_metrics = {
        metric: metrics["metrics"][metric]["p1"]
        for metric in ["agreement", "acc", "relevance", "strength"]
    }

    # p1 agrees with preferred_text on example 1, disagrees on example 2
    assert p1_metrics["agreement"] == 0.5  # 1 agree out of 2 total
    assert p1_metrics["acc"] == 0.5  # 1 agree out of 2 relevant votes
    assert p1_metrics["relevance"] == 1.0  # 2 relevant out of 2 total


def test_compute_metrics_empty_data():
    """Test metric computation with empty input data."""
    empty_df = pd.DataFrame(columns=["comparison_id", "principle", "vote"])

    metrics = compute_annotator_metrics(
        votes_df=empty_df,
        annotator_metadata={},
        annotator_cols=[],
        ref_annotator_col="vote",
    )

    assert len(metrics["annotator_names"]) == 0
    assert metrics["num_pairs"] == 0

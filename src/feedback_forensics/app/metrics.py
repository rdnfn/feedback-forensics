"""Compute metrics"""

import pandas as pd
import gradio as gr
import numpy as np
import sklearn.metrics

from loguru import logger


DEFAULT_METRIC_NAME = "strength"

METRIC_COL_OPTIONS = {
    "agreement": {
        "name": "Agreement",
        "short": "Agr.",
        "descr": "Agreement: proportion of all votes that agree with original preferences",
    },
    "acc": {
        "name": "Accuracy",
        "short": "Acc.",
        "descr": "Accuracy: proportion of non-irrelevant votes ('agree' or 'disagree')<br>that agree with original preferences",
    },
    "relevance": {
        "name": "Relevance",
        "short": "Rel.",
        "descr": "Relevance: proportion of all votes that are not 'not applicable'",
    },
    "strength": {
        "name": "Principle strength (Relevance-weighted Cohen's kappa)",
        "short": "strength",
        "descr": "Principle strength: relevance * Cohen's kappa, or relevance * 2 * (accuracy - 0.5)",
    },
    "cohens_kappa": {
        "name": "Cohen's kappa",
        "short": "kappa",
        "descr": "Cohen's kappa: measures agreement beyond chance, 2 * (accuracy - 0.5).",
    },
}


def get_agreement(
    value_counts: pd.Series, *, annotation_a=None, annotation_b=None
) -> float:
    return value_counts.get("Agree", 0) / value_counts.sum()


def get_acc(value_counts: pd.Series, *, annotation_a=None, annotation_b=None) -> float:
    """
    Accuracy: proportion of non-irrelevant votes ('agree' or 'disagree')
    that agree with original preferences.

    If there are no non-irrelevant votes, return 0.5.
    """
    num_agreed = value_counts.get("Agree", 0)
    num_disagreed = value_counts.get("Disagree", 0)

    denominator = num_agreed + num_disagreed
    if denominator == 0:
        return 0.5
    else:
        return num_agreed / denominator


def get_cohens_kappa(
    value_counts: pd.Series, *, annotation_a=None, annotation_b=None
) -> float:
    """
    Cohen's kappa: measures agreement beyond chance.

    This takes into account agreement across categories (including non-text_a/text_b votes).
    """

    # TODO: replace with sklearn.metrics.cohen_kappa_score

    accuracy = get_acc(value_counts)
    return 2 * (accuracy - 0.5)


def get_relevance(
    value_counts: pd.Series, *, annotation_a=None, annotation_b=None
) -> float:
    return (
        value_counts.get("Agree", 0) + value_counts.get("Disagree", 0)
    ) / value_counts.sum()


def get_principle_strength(
    value_counts: pd.Series, *, annotation_a=None, annotation_b=None
) -> float:
    """
    Relevance-weighted Cohen's kappa: combines Cohen's kappa with relevance.

    This is computed as: (Cohen's kappa) * relevance
    which simplifies to: 2 * (accuracy - 0.5) * relevance
    """
    cohens_kappa = get_cohens_kappa(value_counts)
    relevance = get_relevance(value_counts)
    return cohens_kappa * relevance


def get_num_votes(
    value_counts: pd.Series, *, annotation_a=None, annotation_b=None
) -> int:
    return value_counts.sum()


def get_agreed(value_counts: pd.Series, *, annotation_a=None, annotation_b=None) -> int:
    return value_counts.get("Agree", 0)


def get_disagreed(
    value_counts: pd.Series, *, annotation_a=None, annotation_b=None
) -> int:
    return value_counts.get("Disagree", 0)


def get_not_applicable(
    value_counts: pd.Series, *, annotation_a=None, annotation_b=None
) -> int:
    return value_counts.get("Not applicable", 0)


def compute_annotator_metrics(
    votes_df: pd.DataFrame,
    annotator_metadata: dict,
    annotator_cols: list[str],
    ref_annotator_col: str,
) -> dict:

    # votes_df is a pd.DataFrame with one row
    # per vote, and columns "comparison_id", "principle", "vote"

    metric_fns = {
        "agreement": get_agreement,
        "acc": get_acc,
        "relevance": get_relevance,
        "strength": get_principle_strength,
        "cohens_kappa": get_cohens_kappa,
        "num_votes": get_num_votes,
        "agreed": get_agreed,
        "disagreed": get_disagreed,
        "not_applicable": get_not_applicable,
    }

    # check that ref annotator col only contains "text_a" or "text_b"
    if not all(votes_df[ref_annotator_col].isin(["text_a", "text_b"])):
        values = ", ".join([str(v) for v in list(votes_df[ref_annotator_col].unique())])
        logger.warning(
            f"Reference annotator column '{ref_annotator_col}' contains values other than 'text_a' or 'text_b'(Values: {values}). Metrics will be computed on the subset of votes where the reference annotator is 'text_a' or 'text_b'."
        )
        votes_df = votes_df[
            votes_df[ref_annotator_col].isin(["text_a", "text_b"])
        ].copy()

    annotator_names = [
        annotator_metadata[col]["annotator_in_row_name"] for col in annotator_cols
    ]
    num_pairs = len(votes_df)

    metrics = {}

    for annotator_col in annotator_cols:

        annotator_name = annotator_metadata[annotator_col]["annotator_in_row_name"]

        # Ensure both columns have the same set of categories (annotations)
        joint_categories = set(votes_df[annotator_col].cat.categories).union(
            set(votes_df[ref_annotator_col].cat.categories)
        )
        for col in [annotator_col, ref_annotator_col]:
            votes_df[col] = votes_df[col].cat.set_categories(
                list(joint_categories), rename=False
            )

        valid_votes_mask = votes_df[annotator_col].isin(["text_a", "text_b"])
        agree_mask = (
            votes_df[annotator_col] == votes_df[ref_annotator_col]
        ) & valid_votes_mask
        disagree_mask = valid_votes_mask & ~agree_mask

        # Initialize with "Not applicable" values
        agreement = pd.Series("Not applicable", index=votes_df.index)
        # Set values based on masks
        agreement[agree_mask] = "Agree"
        agreement[disagree_mask] = "Disagree"

        value_counts = agreement.value_counts(sort=False, dropna=False)
        value_counts = value_counts.fillna(0)

        for metric_name, metric_fn in metric_fns.items():
            if metric_name not in metrics:
                metrics[metric_name] = {}
            metrics[metric_name][annotator_name] = metric_fn(value_counts)

    return {
        "annotator_names": annotator_names,
        "num_pairs": num_pairs,
        "metrics": metrics,
    }


def get_overall_metrics(votes_df: pd.DataFrame, ref_annotator_col: str) -> dict:
    """Compute overall metrics

    Includes
      - overall number of votes,
      - percentage of votes that are text_a,
      - percentage of votes that are text_b,
      - average length winning text
      - average length losing text

    Args:
        votes_df: pd.DataFrame
        ref_annotator_col: name of the column that contains the reference
            annotator's preference, e.g. "preferred_text" usually

    Returns:
        dict: overall metrics
    """

    # assert that comparison_id is unique
    comparison_id_counts = votes_df["comparison_id"].value_counts()
    # TODO: ensure that all comparison_ids are unique
    # even if two models produce the same output for the same prompt
    # this should not happen
    non_unique_comparison_ids = comparison_id_counts[comparison_id_counts > 1]
    if len(non_unique_comparison_ids) > 0:
        logger.warning(
            f"Comparison_id is not unique. non-unique values:\n{non_unique_comparison_ids}"
        )
        # limiting to unique comparison_ids, always only leaving in the first occurrence
        votes_df = votes_df.drop_duplicates(subset=["comparison_id"])
        logger.warning(
            f"Limiting to unique comparison_ids, always only leaving in the first occurrence. Num of comparisons removed: {non_unique_comparison_ids.sum() - len(non_unique_comparison_ids)}"
        )

    # limit to votes where ref_annotator_col is "text_a" or "text_b"
    votes_df = votes_df[votes_df[ref_annotator_col].isin(["text_a", "text_b"])].copy()

    # Vectorized implementation: use numpy where instead of apply
    is_text_a = votes_df[ref_annotator_col] == "text_a"
    votes_df["preferred_text_str"] = np.where(
        is_text_a, votes_df["text_a"], votes_df["text_b"]
    )
    votes_df["rejected_text_str"] = np.where(
        is_text_a, votes_df["text_b"], votes_df["text_a"]
    )

    num_votes = len(votes_df)
    num_text_a_preferred = votes_df[ref_annotator_col].value_counts().get("text_a", 0)

    average_length_text_a = votes_df["text_a"].str.len().mean()
    average_length_text_b = votes_df["text_b"].str.len().mean()
    average_length_preferred_text = votes_df["preferred_text_str"].str.len().mean()
    average_length_rejected_text = votes_df["rejected_text_str"].str.len().mean()

    votes_df["longer_text"] = (
        votes_df["text_a"].str.len() > votes_df["text_b"].str.len()
    )

    # make longer_text a str variable with values "text_a" and "text_b"
    votes_df["longer_text"] = votes_df["longer_text"].map(
        {True: "text_a", False: "text_b"}
    )
    votes_df["longer_text_preferred"] = (
        votes_df[ref_annotator_col] == votes_df["longer_text"]
    )

    num_longer_text_preferred = votes_df["longer_text_preferred"].sum()
    proportion_longer_text_preferred = num_longer_text_preferred / num_votes

    return {
        "Number of preference pairs": int(num_votes),
        "Prop selecting text_a": num_text_a_preferred / num_votes,
        "Avg len text_a (chars)": average_length_text_a,
        "Avg len text_b (chars)": average_length_text_b,
        "Avg len selected text (chars)": average_length_preferred_text,
        "Avg len rejected text (chars)": average_length_rejected_text,
        "Prop selecting longer text": proportion_longer_text_preferred,
    }

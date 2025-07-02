"""Compute metrics"""

import pandas as pd
import gradio as gr
import numpy as np
import sklearn.metrics

from loguru import logger

from feedback_forensics.app.constants import (
    DISABLE_SKLEARN_WARNINGS,
    DEFAULT_AVAIL_METRICS,
)

if DISABLE_SKLEARN_WARNINGS:
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)


DEFAULT_METRIC_NAME = "strength"


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
    try:
        kappa = sklearn.metrics.cohen_kappa_score(
            annotation_a,
            annotation_b,
            labels=["text_a", "text_b"],
        )
    except ValueError:
        logger.debug(
            f"Cohen's kappa could not be computed because the annotators have no agreement. Returning 0."
        )
        kappa = 0
    return kappa


def get_cohens_kappa_randomized(
    value_counts: pd.Series, *, annotation_a=None, annotation_b=None
) -> float:
    """
    Cohen's kappa: measures agreement beyond chance.

    This version assumes that at least one of the annotators
    had no access to order information, making the probability
    of chance agreement at 0.5.

    In the regular version, the value will always be 0 if one
    annotator is imbalanced (e.g., by construction).
    """

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
    cohens_kappa = get_cohens_kappa_randomized(
        value_counts, annotation_a=annotation_a, annotation_b=annotation_b
    )
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


def get_metrics():
    return {
        "agreement": {
            "name": "Agreement",
            "short": "Agr",
            "descr": "Agreement: proportion of all votes that agree with original preferences",
            "fn": get_agreement,
        },
        "acc": {
            "name": "Accuracy",
            "short": "Acc",
            "descr": "Accuracy: proportion of non-irrelevant votes ('agree' or 'disagree')<br>that agree with original preferences",
            "fn": get_acc,
        },
        "relevance": {
            "name": "Relevance",
            "short": "Relevance",
            "descr": "Relevance: proportion of all votes that are not 'not applicable'",
            "fn": get_relevance,
        },
        "strength": {
            "name": "Principle strength (Relevance-weighted Cohen's kappa)",
            "short": "Strength",
            "descr": "Principle strength: relevance * Cohen's kappa, or relevance * 2 * (accuracy - 0.5)",
            "fn": get_principle_strength,
        },
        "cohens_kappa_og": {
            "name": "Cohen's kappa (non-adjusted)",
            "short": "kappa",
            "descr": "Cohen's kappa: measures agreement beyond chance. Does not account for randomization in response order during annotation.",
            "fn": get_cohens_kappa,
        },
        "cohens_kappa": {
            "name": "Cohen's kappa (adjusted)",
            "short": "Cohen's kappa",
            "descr": "Cohen's kappa: measures agreement beyond chance. Adjusted version where we assume that at least one of the annotators had no access to order information, meaning random chance agreement has a probability of 0.5.",
            "fn": get_cohens_kappa_randomized,
        },
        "num_votes": {
            "name": "Number of votes",
            "short": "num votes",
            "descr": "Overall number of votes.",
            "fn": get_num_votes,
        },
        "agreed": {
            "name": "Number of agreed votes",
            "short": "num agreed",
            "fn": get_agreed,
        },
        "disagreed": {
            "name": "Number of disagreed votes",
            "short": "num disagreed",
            "fn": get_disagreed,
        },
        "not_applicable": {
            "name": "Number of non-applicable votes",
            "short": "num non-applicable",
            "fn": get_not_applicable,
        },
    }


def compute_annotator_metrics(
    votes_df: pd.DataFrame,
    annotator_metadata: dict,
    annotator_cols: list[str],
    ref_annotator_col: str,
) -> dict:

    # votes_df is a pd.DataFrame with one row
    # per vote, and columns "comparison_id", "principle", "vote"

    metric_dicts = get_metrics()

    # check that ref annotator col only contains "text_a" or "text_b"
    if not all(votes_df[ref_annotator_col].isin(["text_a", "text_b"])):
        values = ", ".join([str(v) for v in list(votes_df[ref_annotator_col].unique())])
        logger.warning(
            f"Reference annotator column '{ref_annotator_col}' contains values other than 'text_a' or 'text_b' (Values: {values}). Metrics will be computed on the subset of votes where the reference annotator is 'text_a' or 'text_b'."
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

        votes_df = ensure_categories_identical(
            df=votes_df, col_a=annotator_col, col_b=ref_annotator_col
        )

        valid_votes_mask = votes_df[annotator_col].isin(["text_a", "text_b"])
        agree_mask = (
            votes_df[annotator_col] == votes_df[ref_annotator_col]
        ) & valid_votes_mask
        disagree_mask = valid_votes_mask & ~agree_mask

        annotation_a = votes_df[annotator_col].copy()
        annotation_b = votes_df[ref_annotator_col].copy()

        # make sure all annotations are strings
        annotation_a = annotation_a.astype(str)
        annotation_b = annotation_b.astype(str)

        # Initialize with "Not applicable" values
        agreement = pd.Series("Not applicable", index=votes_df.index)
        # Set values based on masks
        agreement[agree_mask] = "Agree"
        agreement[disagree_mask] = "Disagree"

        value_counts = agreement.value_counts(sort=False, dropna=False)
        value_counts = value_counts.fillna(0)

        for metric_name, metric_dict in metric_dicts.items():
            metric_fn = metric_dict["fn"]
            if metric_name not in metrics:
                metrics[metric_name] = {}
            metrics[metric_name][annotator_name] = metric_fn(
                value_counts, annotation_a=annotation_a, annotation_b=annotation_b
            )

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
            f"Comparison_id is not unique. non-unique values:{list(non_unique_comparison_ids.index)}"
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

    if num_votes > 0:
        num_longer_text_preferred = votes_df["longer_text_preferred"].sum()
        proportion_longer_text_preferred = num_longer_text_preferred / num_votes
        prop_selecting_text_a = num_text_a_preferred / num_votes
    else:
        proportion_longer_text_preferred = None
        prop_selecting_text_a = None

    return {
        "Number of preference pairs": int(num_votes),
        "Prop selecting text_a": prop_selecting_text_a,
        "Avg len text_a (chars)": average_length_text_a,
        "Avg len text_b (chars)": average_length_text_b,
        "Avg len selected text (chars)": average_length_preferred_text,
        "Avg len rejected text (chars)": average_length_rejected_text,
        "Prop selecting longer text": proportion_longer_text_preferred,
    }


def ensure_categories_identical(
    df: pd.DataFrame, col_a: str, col_b: str
) -> pd.DataFrame:
    # Ensure both columns have the same set of categories (annotations)
    joint_categories = set(df[col_a].cat.categories).union(
        set(df[col_b].cat.categories)
    )
    for col in [col_a, col_b]:
        df[col] = df[col].cat.set_categories(list(joint_categories), rename=False)

    return df


def get_default_avail_metrics():
    all_metrics = get_metrics()

    # sanity check that metric config is valid
    assert isinstance(
        DEFAULT_AVAIL_METRICS, list
    ), f"FF_AVAIL_METRICS setting is not json list ({DEFAULT_AVAIL_METRICS})"
    for metric_name in DEFAULT_AVAIL_METRICS:
        assert (
            metric_name in all_metrics
        ), f"metric '{metric_name}' set in FF_AVAIL_METRICS not in available metric names {list(all_metrics.keys())}"

    dropdown_values = []

    for metric_name in DEFAULT_AVAIL_METRICS:
        dropdown_values.append((all_metrics[metric_name]["short"], metric_name))

    return dropdown_values

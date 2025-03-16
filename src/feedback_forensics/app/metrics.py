"""Compute metrics"""

import pandas as pd


def get_agreement(value_counts: pd.Series) -> float:
    return value_counts.get("Agree", 0) / value_counts.sum()


def get_acc(value_counts: pd.Series) -> float:
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


def get_cohens_kappa(value_counts: pd.Series) -> float:
    """
    Cohen's kappa: measures agreement beyond chance.

    Since the ICAI principle annotator randomizes the order of the alternatives
    (see the function `get_preference_vote_for_single_text` in ICAI), it has no
    position bias and the probability of chance agreement is 50% (even if the
    dataset has position bias). Therefore, the calculation simplifies to

    Cohen's kappa = (accuracy - 0.5) / 0.5 = 2 * (accuracy - 0.5)

    This ranges from -1 (perfect disagreement) through 0 (random agreement)
    to 1 (perfect agreement).

    Returns 0 if there are no relevant votes.
    """
    num_agreed = value_counts.get("Agree", 0)
    num_disagreed = value_counts.get("Disagree", 0)

    accuracy = get_acc(value_counts)
    return 2 * (accuracy - 0.5)


def get_relevance(value_counts: pd.Series) -> float:
    return (
        value_counts.get("Agree", 0) + value_counts.get("Disagree", 0)
    ) / value_counts.sum()


def get_principle_strength(value_counts: pd.Series) -> float:
    """
    Relevance-weighted Cohen's kappa: combines Cohen's kappa with relevance.

    This is computed as: (Cohen's kappa) * relevance
    which simplifies to: 2 * (accuracy - 0.5) * relevance
    """
    cohens_kappa = get_cohens_kappa(value_counts)
    relevance = get_relevance(value_counts)
    return cohens_kappa * relevance


def get_num_votes(value_counts: pd.Series) -> int:
    return value_counts.sum()


def get_agreed(value_counts: pd.Series) -> int:
    return value_counts.get("Agree", 0)


def get_disagreed(value_counts: pd.Series) -> int:
    return value_counts.get("Disagree", 0)


def get_not_applicable(value_counts: pd.Series) -> int:
    return value_counts.get("Not applicable", 0)


def compute_metrics(votes_df: pd.DataFrame, baseline_metrics: dict = None) -> dict:

    # votes_df is a pd.DataFrame with one row
    # per vote, and columns "comparison_id", "principle", "vote"

    metric_fn = {
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

    principles = votes_df["principle"].unique()
    num_pairs = len(votes_df["comparison_id"].unique())

    metrics = {}

    # slightly faster to make data types categorical
    votes_df = votes_df.assign(
        principle=votes_df["principle"].astype("category"),
        vote=votes_df["vote"].astype("category"),
    )

    # more efficient than doing operation for each principle group separately
    value_counts_all = (
        votes_df.groupby(["principle", "vote"], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    # this is equivalent to:
    # grouped = votes_df.groupby("principle", observed=False)
    # for principle in principles:
    #     value_counts = grouped.get_group(principle)["vote"].value_counts(
    #         sort=False, dropna=False
    #     )

    for principle in principles:
        value_counts: pd.Series = value_counts_all.loc[principle]
        value_counts = value_counts.fillna(0)

        for metric in metric_fn.keys():
            if metric not in metrics:
                metrics[metric] = {}
            if "by_principle" not in metrics[metric]:
                metrics[metric]["by_principle"] = {}
            metrics[metric]["by_principle"][principle] = metric_fn[metric](value_counts)

        if baseline_metrics is not None:
            for metric in metric_fn.keys():
                if metric + "_diff" not in metrics:
                    metrics[metric + "_diff"] = {"by_principle": {}}
                if metric + "_base" not in metrics:
                    metrics[metric + "_base"] = {"by_principle": {}}

                metrics[metric + "_diff"]["by_principle"][principle] = (
                    metrics[metric]["by_principle"][principle]
                    - baseline_metrics["metrics"][metric]["by_principle"][principle]
                )

                metrics[metric + "_base"]["by_principle"][principle] = baseline_metrics[
                    "metrics"
                ][metric]["by_principle"][principle]

    for metric in metrics.keys():
        metrics[metric]["principle_order"] = sorted(
            principles,
            key=lambda x: (
                metrics[metric]["by_principle"][x],
                metrics["relevance"]["by_principle"][x],
            ),
        )

    return {
        "principles": principles,
        "num_pairs": num_pairs,
        "metrics": metrics,
    }


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
    "strength_base": {
        "name": "Principle strength on full dataset",
        "short": "(all)",
        "descr": "Principle strength on all datapoints (not just selected subset)",
    },
    "strength_diff": {
        "name": "Principle strength difference (full vs subset)",
        "short": "(diff)",
        "descr": "Absolute principle strength difference to votes on entire dataset",
    },
}


def get_metric_cols_by_principle(
    principle: str,
    metrics: dict,
    metric_names: str,
    metrics_cols_start_y: float,
    metrics_cols_width: float,
) -> dict:
    num_cols = len(metric_names)
    metric_col_width = metrics_cols_width / num_cols

    return [
        [
            metrics_cols_start_y + (i + 1) * metric_col_width,
            metrics["metrics"][metric_name]["by_principle"][principle],
            METRIC_COL_OPTIONS[metric_name]["short"],
            METRIC_COL_OPTIONS[metric_name]["descr"],
        ]
        for i, metric_name in enumerate(metric_names)
    ]


def get_ordering_options(
    metrics,
    shown_metric_names: list,
    initial: str,
) -> list:
    order_options = {
        "agreement": [
            "Agreement ↓",
            metrics["metrics"]["agreement"]["principle_order"],
        ],
        "acc": [
            "Accuracy ↓",
            metrics["metrics"]["acc"]["principle_order"],
        ],
        "relevance": [
            "Relevance ↓",
            metrics["metrics"]["relevance"]["principle_order"],
        ],
        "principle_strength": [
            "Principle strength ↓",
            metrics["metrics"]["principle_strength"]["principle_order"],
        ],
        "cohens_kappa": [
            "Cohen's kappa ↓",
            metrics["metrics"]["cohens_kappa"]["principle_order"],
        ],
        "principle_strength_base": [
            "Principle strength on full dataset ↓",
            metrics["metrics"]["principle_strength_base"]["principle_order"],
        ],
        "principle_strength_diff": [
            "Principle strength difference ↓",
            metrics["metrics"]["principle_strength_diff"]["principle_order"],
        ],
    }

    if initial not in order_options.keys():
        raise ValueError(f"Initial ordering metric '{initial}' not found.")

    ordering = [
        value for key, value in order_options.items() if key in shown_metric_names
    ]

    if initial in shown_metric_names:
        # make sure initial is first
        ordering.insert(0, ordering.pop(ordering.index(order_options[initial])))

    return ordering


def get_overall_metrics(votes_df: pd.DataFrame) -> dict:
    """Compute overall metrics

    Includes
      - overall number of votes,
      - percentage of votes that are text_a,
      - percentage of votes that are text_b,
      - average length winning text
      - average length losing text
    """

    # get a single row per comparison_id
    votes_df = votes_df.groupby("comparison_id").first()

    # if preferred_text is text_a, preferred_text_str is value of column "text_a" (otherwise of "text_b")
    votes_df["preferred_text_str"] = votes_df.apply(
        lambda x: x["text_a"] if x["preferred_text"] == "text_a" else x["text_b"],
        axis=1,
    )
    votes_df["rejected_text_str"] = votes_df.apply(
        lambda x: x["text_b"] if x["preferred_text"] == "text_a" else x["text_a"],
        axis=1,
    )

    num_votes = len(votes_df)
    num_text_a_preferred = votes_df["preferred_text"].value_counts().get("text_a", 0)
    num_text_b_preferred = votes_df["preferred_text"].value_counts().get("text_b", 0)

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
        votes_df["preferred_text"] == votes_df["longer_text"]
    )

    num_longer_text_preferred = votes_df["longer_text_preferred"].sum()
    proportion_longer_text_preferred = num_longer_text_preferred / num_votes

    return {
        "Number of preference pairs": num_votes,
        "Prop preferring text_a": num_text_a_preferred / num_votes,
        "Avg len text_a (chars)": average_length_text_a,
        "Avg len text_b (chars)": average_length_text_b,
        "Avg len pref. text (chars)": average_length_preferred_text,
        "Avg len rej. text (chars)": average_length_rejected_text,
        "Prop preferring longer text": proportion_longer_text_preferred,
    }

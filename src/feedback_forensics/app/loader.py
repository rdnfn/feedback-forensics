import pathlib
import json
import ast
import pandas as pd
from loguru import logger


def load_json_file(path: str):
    with open(path, "r") as f:
        content = json.load(f)

    return content


def convert_vote_to_string(vote: bool | None) -> str:
    if vote is True:
        return "Agree"
    elif vote is False:
        return "Disagree"
    elif vote is None:
        return "Not applicable"
    elif vote == "invalid":
        return "Invalid"
    else:
        raise ValueError(f"Completely invalid vote value: {vote}")


def get_votes_df(results_dir: pathlib.Path, cache: dict) -> pd.DataFrame:
    """
    Get the votes dataframe for a given results directory.
    If the dataframe is already in the cache, return it.
    Otherwise, create it, add it to the cache, and return it.
    """

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found in path '{results_dir}'")

    # check if the results dir is empty
    if not any(results_dir.iterdir()):
        raise FileNotFoundError(f"Results directory is empty in path '{results_dir}'")

    if "votes_df" in cache and results_dir in cache["votes_df"]:
        return cache["votes_df"][results_dir]
    else:
        votes_df = create_votes_df(results_dir)

        if "votes_df" not in cache:
            cache["votes_df"] = {}
        cache["votes_df"][results_dir] = votes_df

        return votes_df


def create_votes_df(results_dir: pathlib.Path) -> list[dict]:
    """Create the votes dataframe from log files.

    Args:
        results_dir (pathlib.Path): Path to the results directory.

    Returns:
        pd.DataFrame: The votes dataframe.
    """

    # load relevant data from experiment logs
    votes_per_comparison = pd.read_csv(
        results_dir / "040_votes_per_comparison.csv", index_col="index"
    )
    principles_by_id: dict = load_json_file(
        results_dir / "030_distilled_principles_per_cluster.json",
    )
    comparison_df = pd.read_csv(results_dir / "000_train_data.csv", index_col="index")

    # merge original comparison data with votes per comparison
    full_df = comparison_df.merge(
        votes_per_comparison, left_index=True, right_index=True
    )
    full_df["comparison_id"] = full_df.index

    # add vote data column
    full_df["votes_dicts"] = full_df["votes"].apply(ast.literal_eval)

    # Instead of exploding into rows, create columns for each principle
    for principle_id, principle_text in principles_by_id.items():
        column_name = f"annotation_principle_{principle_id}"

        # Extract vote for this principle and convert to string
        full_df[column_name] = full_df["votes_dicts"].apply(
            lambda x: convert_vote_to_string(x.get(principle_id, None))
        )

    # add a weight column
    full_df["weight"] = 1

    # Clean up temporary columns if no longer needed
    full_df = full_df.drop(columns=["votes_dicts"])

    return full_df

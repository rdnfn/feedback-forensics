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


def get_votes_dict(results_dir: pathlib.Path, cache: dict) -> dict:
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

    if "votes_dict" in cache and results_dir in cache["votes_dict"]:
        return cache["votes_dict"][results_dir]
    else:
        votes_dict = create_votes_dict(results_dir)

        if "votes_dict" not in cache:
            cache["votes_dict"] = {}
        cache["votes_dict"][results_dir] = votes_dict
        return votes_dict


def create_votes_dict(results_dir: pathlib.Path) -> list[dict]:
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

    annotator_metadata = {}

    # Instead of exploding into rows, create columns for each principle
    for principle_id, principle_text in principles_by_id.items():
        column_name = f"annotation_principle_{principle_id}"
        annotator_metadata[column_name] = {
            "variant": "icai_principle",
            "principle_id": principle_id,
            "principle_text": principle_text,
            "annotator_visible_name": principle_text,
        }

        # Extract vote for this principle and convert to string
        full_df[column_name] = full_df["votes_dicts"].apply(
            lambda x: convert_vote_to_string(x.get(int(principle_id), None))
        )

        # Vectorized implementation instead of row-by-row apply
        # First check that all preferred_text values are either text_a or text_b
        assert (
            full_df["preferred_text"].isin(["text_a", "text_b"]).all()
        ), "Tie or other votes currently not supported."

        # Create a Series for the rejected text (opposite of preferred_text)
        rejected_text = pd.Series(
            [
                "text_b" if pt == "text_a" else "text_a"
                for pt in full_df["preferred_text"]
            ],
            index=full_df.index,
        )

        # Create masks based on the current column values
        agree_mask = full_df[column_name] == "Agree"
        disagree_mask = full_df[column_name] == "Disagree"

        # Create a copy of the column to store results
        result = pd.Series("Not applicable", index=full_df.index)

        # Set values based on conditions
        result[agree_mask] = full_df.loc[agree_mask, "preferred_text"].values
        result[disagree_mask] = rejected_text.loc[disagree_mask].values

        # Update the column
        full_df[column_name] = result

        # ensure column is categorical
        full_df[column_name] = full_df[column_name].astype("category")

    # add a weight column
    full_df["weight"] = 1

    # Clean up temporary columns if no longer needed
    full_df = full_df.drop(columns=["votes_dicts"])

    return {
        "df": full_df,
        "annotator_metadata": annotator_metadata,
        "reference_annotator_col": "preferred_text",
    }

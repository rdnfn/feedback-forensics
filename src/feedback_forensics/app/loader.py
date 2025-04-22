import pathlib
import json
import ast
import pandas as pd
from loguru import logger
from inverse_cai.data.annotated_pairs_format import hash_string
from feedback_forensics.app.constants import (
    DEFAULT_ANNOTATOR_NAME,
    DEFAULT_ANNOTATOR_HASH,
    PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS,
)


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


def get_votes_dict(results_path: pathlib.Path, cache: dict) -> dict:
    """
    Get the votes dataframe for a given results directory.
    If the dataframe is already in the cache, return it.
    Otherwise, create it, add it to the cache, and return it.
    """

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found in path '{results_path}'")

    if "votes_dict" in cache and results_path in cache["votes_dict"]:
        return cache["votes_dict"][results_path]
    else:
        # check if results_path is a directory and non empty
        if results_path.is_dir():
            if not any(results_path.iterdir()):
                raise FileNotFoundError(f"Empty directory: {results_path}")
            # use class method to create votes dict
            votes_dict = create_votes_dict_from_icai_log_files(results_path / "results")
        elif results_path.is_file() and results_path.suffix == ".json":
            votes_dict = get_votes_dict_from_annotated_pairs_json(results_path)
        else:
            raise FileNotFoundError(f"Unsupported results directory: {results_path}")

        if "votes_dict" not in cache:
            cache["votes_dict"] = {}
        cache["votes_dict"][results_path] = votes_dict
        return votes_dict


def get_votes_dict_from_annotated_pairs_json(results_path: pathlib.Path) -> dict:
    """
    Get the votes dataframe for a given json path
    """

    # load json file
    json_data = load_json_file(results_path)

    # check format version
    format_version = json_data.get("metadata", {}).get("version", "1.0")
    major_version = format_version.split(".")[0]
    is_format_v2 = major_version == "2"
    logger.info(f"Annotated pairs format version: {format_version}")

    # create dataframe from comparisons
    comparisons_data = []
    for comparison in json_data["comparisons"]:
        # Handle responses based on format version
        if is_format_v2:
            # Format 2.0: responses are dictionaries with multiple fields
            response_a = comparison["response_a"]
            response_b = comparison["response_b"]

            row_data = {
                "comparison_id": comparison["id"],
            }

            # Add all response fields as key_a and key_b in the dataframe
            for key, value in response_a.items():
                row_data[f"{key}_a"] = value

            for key, value in response_b.items():
                row_data[f"{key}_b"] = value
        else:
            # Format 1.0: responses are strings
            row_data = {
                "comparison_id": comparison["id"],
                "text_a": comparison["text_a"],
                "text_b": comparison["text_b"],
            }

        # Add prompt if it exists
        if comparison.get("prompt"):
            row_data["prompt"] = comparison["prompt"]

        # Add annotations
        for annotator_id, annotation in comparison.get("annotations", {}).items():
            if "pref" in annotation:
                # Convert "a"/"b" format to "text_a"/"text_b" format
                pref_value = annotation["pref"]
                if is_format_v2 and pref_value in ["a", "b"]:
                    row_data[annotator_id] = f"text_{pref_value}"
                else:
                    row_data[annotator_id] = pref_value
            else:
                logger.warning(
                    f"No preference found for annotator {annotator_id} in comparison {comparison['id']}, (annotation: '{annotation}')"
                )

        # add metadata columns per comparison
        for key, value in comparison.get("metadata", {}).items():
            row_data[key] = value

        comparisons_data.append(row_data)

    # create dataframe
    full_df = pd.DataFrame(comparisons_data)

    # Create annotator metadata
    annotator_metadata = {}
    principle_annotator_cols = []

    for annotator_id, annotator_info in json_data["annotators"].items():
        if annotator_id == json_data["metadata"].get("default_annotator"):
            annotator_metadata[annotator_id] = {
                "variant": "default_annotator",
                "annotator_visible_name": annotator_info.get("name", annotator_id),
                "annotator_in_row_name": annotator_id,
                "annotator_description": annotator_info.get("description", ""),
            }
        elif annotator_info.get("type") == "principle":
            short_principle_text = (
                annotator_info["description"]
                .replace("Select the response that", "")
                .strip(" .")
            )

            annotator_metadata[annotator_id] = {
                "variant": "icai_principle",
                "principle_id": annotator_id,
                "principle_text": annotator_info["description"],
                "annotator_visible_name": PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS
                + short_principle_text,
                "annotator_in_row_name": short_principle_text,
                "annotator_description": annotator_info.get("description", ""),
            }

            principle_annotator_cols.append(annotator_id)
        else:
            annotator_metadata[annotator_id] = {
                "variant": "nondefault_annotator",
                "annotator_visible_name": annotator_info.get("name", annotator_id),
                "annotator_in_row_name": annotator_id,
                "annotator_description": annotator_info.get("description", ""),
            }

    # Add weight column
    full_df["weight"] = 1

    return {
        "df": full_df,
        "shown_annotator_rows": principle_annotator_cols,
        "annotator_metadata": annotator_metadata,
        "reference_annotator_col": DEFAULT_ANNOTATOR_HASH,
    }


def _check_for_nondefault_annotators(df: pd.DataFrame) -> dict:
    """Check for non-default annotators in the dataframe.

    Checks for each column in dataframe if it contains a column
    with values "text_a" or "text_b", and if so, adds it as an
    annotator metadata entry."""

    annotator_metadata = {}

    for col in df.columns:
        if col != DEFAULT_ANNOTATOR_NAME and df[col].isin(["text_a", "text_b"]).any():
            annotator_metadata[col] = {
                "variant": "nondefault_annotation_column",
                "annotator_visible_name": col,
                "annotator_in_row_name": col,
            }

    logger.info(f"Found {len(annotator_metadata)} non-default annotators")
    return annotator_metadata


def create_votes_dict_from_icai_log_files(results_dir: pathlib.Path) -> list[dict]:
    """Create the votes dataframe and voter metadata from ICAI log files.

    Args:
        results_dir (pathlib.Path): Path to the results directory.

    Returns:
       dict: A dictionary containing the votes dataframe and annotator metadata.
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
    annotator_metadata[DEFAULT_ANNOTATOR_HASH] = {
        "variant": "default_annotator",
        "annotator_visible_name": DEFAULT_ANNOTATOR_NAME,
        "annotator_in_row_name": DEFAULT_ANNOTATOR_NAME,
    }

    annotator_metadata.update(_check_for_nondefault_annotators(full_df))

    principle_annotator_cols = []

    # Create separate columns for each principle annotation
    for principle_id, principle_text in principles_by_id.items():
        column_name = hash_string(principle_text)
        short_principle_text = principle_text.replace(
            "Select the response that", ""
        ).strip(" .")

        annotator_metadata[column_name] = {
            "variant": "icai_principle",
            "principle_id": principle_id,
            "principle_text": principle_text,
            "annotator_visible_name": PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS
            + short_principle_text,
            "annotator_in_row_name": short_principle_text,
        }

        principle_annotator_cols.append(column_name)

        # Extract vote for this principle and convert to string
        full_df[column_name] = full_df["votes_dicts"].apply(
            lambda x: convert_vote_to_string(x.get(int(principle_id), None))
        )

        # Vectorized implementation instead of row-by-row apply
        # First check that all preferred_text values are either text_a or text_b
        assert (
            full_df[DEFAULT_ANNOTATOR_NAME].isin(["text_a", "text_b"]).all()
        ), "Tie or other votes currently not supported."

        # Create a Series for the rejected text (opposite of preferred_text)
        rejected_text = pd.Series(
            [
                (
                    "text_b"
                    if pt == "text_a"
                    else "text_a" if pt in ["text_a", "text_b"] else "Not applicable"
                )
                for pt in full_df[DEFAULT_ANNOTATOR_NAME]
            ],
            index=full_df.index,
        )

        # Create masks based on the current column values
        agree_mask = full_df[column_name] == "Agree"
        disagree_mask = full_df[column_name] == "Disagree"

        # Create a copy of the column to store results
        result = pd.Series("Not applicable", index=full_df.index)

        # Set values based on conditions
        result[agree_mask] = full_df.loc[agree_mask, DEFAULT_ANNOTATOR_NAME].values
        result[disagree_mask] = rejected_text.loc[disagree_mask].values

        # Update the column
        full_df[column_name] = result

        # ensure column is categorical
        full_df[column_name] = full_df[column_name].astype("category")
    # add a weight column
    full_df["weight"] = 1

    # Clean up temporary columns if no longer needed
    full_df = full_df.drop(columns=["votes_dicts"])

    # rename preferred_text to default_annotator_hash
    full_df.rename(
        columns={DEFAULT_ANNOTATOR_NAME: DEFAULT_ANNOTATOR_HASH}, inplace=True
    )

    return {
        "df": full_df,
        "shown_annotator_rows": principle_annotator_cols,
        "annotator_metadata": annotator_metadata,
        "reference_annotator_col": DEFAULT_ANNOTATOR_HASH,
    }

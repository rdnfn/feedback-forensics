"""Utilities for working with datasets."""

import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import ijson

import pandas as pd
from loguru import logger

from feedback_forensics.app.constants import PREFIX_OTHER_ANNOTATOR_WITH_VARIANT


def get_available_models(df: pd.DataFrame) -> list:
    """
    Get all available model names from a dataset.

    Args:
        df: DataFrame containing the dataset

    Returns:
        List of unique model names found in the dataset
    """
    required_cols = ["model_a", "model_b"]
    if not all(col in df.columns for col in required_cols):
        return []

    model_a_values = set(x for x in df["model_a"].unique() if pd.notna(x))
    model_b_values = set(x for x in df["model_b"].unique() if pd.notna(x))
    return list(model_a_values.union(model_b_values))


def add_annotators_to_votes_dict(
    votes_dict: dict, annotator_metadata: dict, annotations_df: pd.DataFrame
) -> dict:
    """
    Add annotators to an existing votes_dict.

    Args:
        votes_dict: An existing votes_dict with dataframe and metadata
        annotator_metadata: Dictionary of annotator metadata to add
        annotations_df: DataFrame with only comparison_id and annotator columns

    Returns:
        A new votes_dict with annotators added
    """
    start_time = time.time()

    result = votes_dict.copy()
    result["df"] = result["df"].merge(annotations_df, on="comparison_id", how="left")
    result["annotator_metadata"].update(annotator_metadata)

    elapsed_time = time.time() - start_time
    logger.debug(
        f"Added {len(annotator_metadata)} virtual annotators to the dataset in {elapsed_time:.2f} seconds"
    )
    return result


def get_annotators_by_type(
    votes_dict: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract all annotators grouped by their type from a votes_dict.

    Args:
        votes_dict: Dictionary containing annotator metadata

    Returns:
        Dictionary mapping variant types to dictionaries with "column_ids" and "visible_names" keys
        Automatically handles missing keys with empty lists using defaultdict
    """
    result = defaultdict(lambda: {"column_ids": [], "visible_names": []})

    for col, metadata in votes_dict["annotator_metadata"].items():
        variant = metadata.get("variant", "unknown")

        result[variant]["column_ids"].append(col)

        if "annotator_visible_name" in metadata:
            result[variant]["visible_names"].append(metadata["annotator_visible_name"])
        else:
            # Default visible name indicates variant and column ID
            result[variant]["visible_names"].append(
                PREFIX_OTHER_ANNOTATOR_WITH_VARIANT.format(variant=variant) + col
            )

    return result


def get_first_json_key_value(file_path):
    """Get the first key and value from a JSON file.

    This is useful for extracting metadata from AnnotatedPairs
    file, without loading full dataset.
    """

    with open(file_path, "rb") as f:
        parser = ijson.parse(f)
        first_key = None
        for prefix, event, value in parser:
            if event == "map_key" and first_key is None:
                first_key = value
            elif (
                event in ("string", "number", "boolean", "null")
                and first_key is not None
            ):
                return first_key, value
            elif event == "start_array" and first_key is not None:
                # Load the entire array value
                array_value = list(
                    ijson.items(open(file_path, "rb"), f"{first_key}.item")
                )
                return first_key, array_value
            elif event == "start_map" and first_key is not None and prefix == first_key:
                # Load the entire object value
                obj_value = next(ijson.items(open(file_path, "rb"), first_key))
                return first_key, obj_value
    return None, None

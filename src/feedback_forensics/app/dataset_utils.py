"""Utilities for working with datasets."""

import time
from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger


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
    return model_a_values.union(model_b_values)


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

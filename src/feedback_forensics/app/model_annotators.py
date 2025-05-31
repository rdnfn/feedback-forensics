"""Functionality for creating virtual model identity annotators."""

import time

import pandas as pd
from inverse_cai.data.annotated_pairs_format import hash_string
from loguru import logger

from feedback_forensics.app.constants import (
    MODEL_IDENTITY_ANNOTATOR_TYPE,
    PREFIX_MODEL_IDENTITY_ANNOTATORS,
)
from feedback_forensics.data.dataset_utils import get_available_models
from feedback_forensics.app.utils import iter_to_trunc_str


def generate_model_identity_annotators(
    df: pd.DataFrame, target_models: list, reference_models: list
) -> tuple:
    """
    Generate model identity annotators for target models compared against reference models.

    Args:
        df: DataFrame containing the dataset
        target_models: List of model names to use as target models. Empty list means all models are used as targets.
        reference_models: List of model names to use as reference models. Empty list means all models are used as references.

    Returns:
        Tuple of (annotator_metadata, df_with_annotators)
    """
    start_time = time.time()

    all_targets = len(target_models) == 0
    one_vs_all = len(reference_models) == 0
    if all_targets or one_vs_all:
        all_models = get_available_models(df)
        if all_targets:
            target_models = all_models
        if one_vs_all:
            reference_models = all_models

    annotations_df = df[["comparison_id"]].copy()

    if len(target_models) == 0:
        logger.info(
            "No target models available, creating no model identity annotators."
        )
        return {}, annotations_df

    if len(reference_models) == 0:
        logger.info("No reference models specified. Using all models as references.")

    # Filter out rows with missing model data or same model on both sides
    valid_mask = ~(
        pd.isna(df["model_a"])
        | pd.isna(df["model_b"])
        | (df["model_a"] == df["model_b"])
    )
    valid_model_a = df.loc[valid_mask, "model_a"]
    valid_model_b = df.loc[valid_mask, "model_b"]
    valid_idx = df.index[valid_mask]

    model_a_in_refs = valid_model_a.isin(reference_models)
    model_b_in_refs = valid_model_b.isin(reference_models)

    annotator_start = time.time()
    prep_time = annotator_start - start_time
    logger.debug(f"Model annotator preparation time: {prep_time:.2f} seconds")

    # For each target model, create an annotator that always prefers that model over reference models
    total_annotations = 0
    annotator_metadata = {}

    for model in target_models:
        other_refs = [m for m in reference_models if m != model]

        annotator_name = f"model_identity_{model}_over_references"
        annotator_id = hash_string(annotator_name)

        if one_vs_all:
            description = f"Always prefer {model} over any other model"
        else:
            description = f"Always prefer {model} over reference models ({iter_to_trunc_str(other_refs, 3)})"

        # remove openrouter prefix from model name
        model_name = model.replace("openrouter/", "")

        annotator_metadata[annotator_id] = {
            "variant": MODEL_IDENTITY_ANNOTATOR_TYPE,
            "model_id": model,
            "reference_models": reference_models,
            "annotator_visible_name": f"{PREFIX_MODEL_IDENTITY_ANNOTATORS}{model_name}",
            "annotator_in_row_name": f"{model}-preference",
            "annotator_description": description,
        }

        values = pd.Series("Not applicable", index=df.index)

        # If there are no reference models, all annotations are "Not applicable"
        if len(other_refs) == 0:
            annotations_df[annotator_id] = values
            annotations_df[annotator_id] = annotations_df[annotator_id].astype(
                "category"
            )
            continue

        # Create masks for when this model should be preferred (in column A or B)
        a_preference_mask = (valid_model_a == model) & model_b_in_refs
        b_preference_mask = (valid_model_b == model) & model_a_in_refs

        values.loc[valid_idx[a_preference_mask]] = "text_a"
        values.loc[valid_idx[b_preference_mask]] = "text_b"

        annotations_df[annotator_id] = values
        annotations_df[annotator_id] = annotations_df[annotator_id].astype("category")

        annotation_count = a_preference_mask.sum() + b_preference_mask.sum()
        total_annotations += annotation_count
        logger.debug(
            f"Created {annotation_count} annotations for model {model} (annotator ID: {annotator_id})"
        )

    annotator_time = time.time() - annotator_start
    total_time = time.time() - start_time
    logger.debug(f"Creating annotators time: {annotator_time:.2f} seconds")
    logger.info(
        f"Created {total_annotations} annotations for {len(target_models)} model annotators with {len(reference_models)} reference models in {total_time:.2f} seconds"
    )

    return annotator_metadata, annotations_df

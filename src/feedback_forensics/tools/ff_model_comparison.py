"""Tools to compare models' personalities.

Uses generations created by ff_modelgen."""

import asyncio
import pathlib
import pandas as pd
import feedback_forensics.tools.ff_modelgen as modelgen
from loguru import logger


def create_pairwise_datasets(
    model_names: list[str],
    reference_models: list[str],
    generations: dict[str, pd.DataFrame],
    output_path: str = "output/tmp/pairwise_datasets",
):
    """Create pairwise dataset for each model against the reference models.

    Args:
        model_names (list[str]): List of model names to compare.
        reference_models (list[str]): List of reference model names to compare against.
        generations (dict[str, pd.DataFrame]): Dictionary of generations for each model.
        output_path (str): Path to save the pairwise datasets.
    """
    save_path = pathlib.Path(output_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create pairwise dataset for each model against the reference models
    for model_name in model_names:
        logger.info(f"Creating pairwise dataset for {model_name}")

        data = []

        for reference_model in reference_models:
            # Load generations for the model
            ref_gens = generations[reference_model]


def compare_models(
    prompts: list[str],
    model_names: list[str],
    reference_models: list[str],
    output_path: str = "output/",
):
    """Compare models' personalities.

    First creates generations for the given set of prompts.
    Then creates pairwise dataset for each model against the reference models.
    Then annotates the pairwise dataset using inverse-cai package.

    Args:
        prompts: List of prompts to compare.
        model_names: List of model names to compare.
        reference_models: List of reference model names to compare against.
        output_path: Path to save the results.
    """

    logger.info(f"Comparing models: {model_names} against {reference_models}")
    logger.info(f"Number of prompts: {len(prompts)}")

    # Sanity checks

    # Make sure all reference models are in model_names
    for reference_model in reference_models:
        assert (
            reference_model in model_names
        ), f"Reference model {reference_model} not in model_names"

    # Make sure model_names are unique
    assert len(model_names) == len(set(model_names)), "Model names must be unique"

    # Make sure reference_models are unique
    assert len(reference_models) == len(
        set(reference_models)
    ), "Reference models must be unique"

    # Make sure prompts are unique
    assert len(prompts) == len(set(prompts)), "Prompts must be unique"

    logger.info(f"Stage 1: Creating generations for {model_names}")

    # Create generations for each model
    for model_name in model_names:
        logger.info(f"Creating generations for {model_name}")
        asyncio.run(
            modelgen.run_model_on_prompts_async(
                prompts, model_name, output_path, max_concurrent=10
            )
        )

    # Load generations for each model
    generations = {}
    for model_name in model_names:
        generations[model_name] = modelgen.load_generations(output_path, model_name)

    # Create pairwise dataset for each model against the reference models
    create_pairwise_datasets(model_names, reference_models, generations, output_path)

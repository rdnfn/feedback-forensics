"""Tools to compare models' personalities.

Uses generations created by ff_modelgen."""

import asyncio
import pathlib
import pandas as pd
import subprocess
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
        if model_name in reference_models:
            continue  # Skip self-comparison

        logger.info(f"Creating pairwise dataset for {model_name}")

        data = []
        model_gens = generations[model_name]

        for reference_model in reference_models:
            ref_gens = generations[reference_model]

            # Create pairwise comparisons for each prompt
            for _, model_row in model_gens.iterrows():
                prompt = model_row["prompt"]
                ref_row = ref_gens[ref_gens["prompt"] == prompt].iloc[0]

                if len(ref_row) > 0:
                    data.append(
                        {
                            "prompt": prompt,
                            "text_a": model_row["response"],
                            "text_b": ref_row["response"],
                            "model_a": model_name,
                            "model_b": reference_model,
                        }
                    )

        # Save pairwise dataset
        if data:
            df = pd.DataFrame(data)
            filename = f"{model_name.replace('/', '_')}.csv"
            df.to_csv(save_path / filename, index=False)
            logger.info(f"Saved {len(data)} comparisons to {save_path / filename}")


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
    for reference_model in reference_models:
        assert (
            reference_model in model_names
        ), f"Reference model {reference_model} not in model_names"

    assert len(model_names) == len(set(model_names)), "Model names must be unique"
    assert len(reference_models) == len(
        set(reference_models)
    ), "Reference models must be unique"
    assert len(prompts) == len(set(prompts)), "Prompts must be unique"

    ### STAGE 1: Generation ###
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

    ### STAGE 2: pairwise dataset creation ###
    logger.info("Stage 2: Creating pairwise datasets")
    pairwise_path = pathlib.Path(output_path) / "tmp" / "pairwise_datasets"
    create_pairwise_datasets(model_names, reference_models, generations, pairwise_path)

    ### STAGE 3: annotation ###
    logger.info("Stage 3: Annotating pairwise datasets")
    # Annotate each pairwise dataset
    annotation_dir = pathlib.Path(output_path) / "annotations"

    for csv_file in pairwise_path.glob("*.csv"):
        logger.info(f"Annotating {csv_file}")
        subprocess.run(
            [
                "ff-annotate",
                "--datapath",
                str(csv_file),
                "--output-dir",
                str(annotation_dir / csv_file.stem),
            ],
            check=True,
        )

    logger.info(f"Comparison complete. Results in {output_path}")

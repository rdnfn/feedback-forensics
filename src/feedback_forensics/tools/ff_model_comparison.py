"""Tools to compare models' personalities.

Uses generations created by ff_modelgen."""

import asyncio
import pathlib
import pandas as pd
import subprocess
import shutil
import feedback_forensics.tools.ff_modelgen as modelgen
from loguru import logger
import argparse
import json
from feedback_forensics.data.operations import load_ap, save_ap, merge_ap, csv_to_ap


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

        filename = f"{model_name.replace('/', '_')}.csv"
        if (save_path / filename).exists():
            logger.info(
                f"Pairwise dataset already exists for {model_name}. Skipping..."
            )
            continue

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
            df.to_csv(save_path / filename, index=False)
            logger.info(f"Saved {len(data)} comparisons to {save_path / filename}")
        else:
            logger.warning(f"No data to save for {model_name} vs {reference_models}")


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
    annotation_tmp_dir = pathlib.Path(output_path) / "tmp" / "annotations"
    final_annotations_dir = pathlib.Path(output_path) / "annotations"
    annotation_tmp_dir.mkdir(parents=True, exist_ok=True)
    final_annotations_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in pairwise_path.glob("*.csv"):
        logger.info(f"Annotating {csv_file}")
        tmp_output_dir = annotation_tmp_dir / csv_file.stem
        final_annotation_path = final_annotations_dir / (csv_file.stem + "_ap.json")

        if not final_annotation_path.exists():
            subprocess.run(
                [
                    "ff-annotate",
                    "--datapath",
                    str(csv_file),
                    "--output-dir",
                    str(tmp_output_dir),
                ],
                check=True,
            )
            # Move the annotation to the final output path
            shutil.copyfile(
                src=tmp_output_dir / "results" / "070_annotations_train_ap.json",
                dst=final_annotation_path,
            )
        else:
            logger.info(f"Annotation already exists for {csv_file}. Skipping...")

    # Merge annotations
    logger.info("Merging annotations")

    ap_files = list(final_annotations_dir.glob("*.json"))
    ap_files = [load_ap(file) for file in ap_files]
    combined_ap = ap_files[0]
    if len(ap_files) > 1:
        for ap_file in ap_files[1:]:
            combined_ap = merge_ap(
                combined_ap,
                ap_file,
                dataset_name="🎭 Model Personality Comparison",
                description="Model Personality Comparison dataset between "
                + ", ".join(model_names)
                + ". Using "
                + ", ".join(reference_models)
                + " as reference model(s). "
                + "Created and annotated using Feedback Forensics, "
                + "see https://huggingface.co/datasets/rdnfn/ff-model-personality "
                + "for more details.",
            )
    save_ap(combined_ap, final_annotations_dir / "combined_ap.json")
    logger.info(
        f"Saved combined annotation to {final_annotations_dir / 'combined_ap.json'}"
    )

    logger.info(f"Comparison complete. Results in {output_path}")


def run():
    """CLI entry point for model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare models' personalities using Feedback Forensics pipeline."
    )

    # Prompts input
    prompts_group = parser.add_mutually_exclusive_group(required=True)
    prompts_group.add_argument(
        "-p", "--prompts", nargs="+", help="List of prompts to compare models on"
    )
    prompts_group.add_argument(
        "--prompts-file",
        type=str,
        help="Path to file containing prompts in JSON array format",
    )

    # Model configuration
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        help="List of model names to compare",
    )

    parser.add_argument(
        "-r",
        "--reference-models",
        nargs="+",
        required=True,
        help="List of reference model names to compare against",
    )

    # Output configuration
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="output/",
        help="Path to save results (default: output/)",
    )

    args = parser.parse_args()

    # Parse prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts_file = pathlib.Path(args.prompts_file)
        if not prompts_file.exists():
            logger.error(f"Prompts file not found: {prompts_file}")
            return 1

        try:
            with open(prompts_file) as f:
                prompts = json.load(f)
            if not isinstance(prompts, list):
                logger.error("JSON file must contain an array of strings")
                return 1
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in prompts file: {e}")
            return 1

    if not prompts:
        logger.error("No prompts provided")
        return 1

    compare_models(
        prompts=prompts,
        model_names=args.models,
        reference_models=args.reference_models,
        output_path=args.output_path,
    )
    logger.success(
        f"Model personality comparison data generated successfully! To see results, run\n   feedback-forensics -d {args.output_path}/annotations/combined_ap.json"
    )
    return 0


if __name__ == "__main__":
    exit(run())

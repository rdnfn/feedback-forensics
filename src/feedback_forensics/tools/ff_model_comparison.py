"""Tools to compare models' personalities.

Uses generations created by ff_modelgen."""

import asyncio
import pathlib
import time
import pandas as pd
import shutil
import feedback_forensics.tools.ff_modelgen as modelgen
from loguru import logger
import argparse
import json
from feedback_forensics.data.operations import load_ap, save_ap, merge_ap, csv_to_ap
from feedback_forensics.annotation import annotate_dataset, get_default_principles
from feedback_forensics.annotation.pipeline import USAGE_SUMMARY_FILENAME
from feedback_forensics.tools.ff_annotate import DEFAULT_ANNOTATOR_MODEL


def get_annotation_settings(annotator_model: str, position_debias: bool) -> dict:
    """Annotation settings of compare_models (as in `metadata.annotation` of ff-annotate
    output). Only the annotator model and position debiasing are configurable: the
    other settings are fixed to keep annotations of models added over time comparable.
    """
    return {
        "annotator_model": annotator_model,
        "position_debias": position_debias,
        "max_output_tokens": 3000,
        "reasoning_effort": None,
    }


# Settings of annotation files without `metadata.annotation` (or null): created by
# the former ICAI-based ff-annotate, which always used these settings.
LEGACY_ANNOTATION_SETTINGS = get_annotation_settings(
    annotator_model="openrouter/google/gemini-2.5-flash", position_debias=False
)


class AnnotationSettingsError(ValueError):
    """Existing annotations were created with other settings than requested."""


def check_annotation_settings(annotations_dir: pathlib.Path, settings: dict) -> None:
    """Check that all annotation files in a directory were created with `settings`.

    Existing annotation files are reused (not annotated again) and merged into the
    combined dataset, so files annotated with other settings would be mixed in.

    Raises:
        AnnotationSettingsError: if any file was created with other settings.
    """
    mismatches = []
    for path in sorted(annotations_dir.glob("*.json")):
        recorded = load_ap(path)["metadata"].get("annotation")
        if recorded is None:
            recorded = LEGACY_ANNOTATION_SETTINGS
        differences = [
            f"{key}={recorded.get(key)!r} (requested: {value!r})"
            for key, value in settings.items()
            if recorded.get(key) != value
        ]
        if differences:
            mismatches.append(f"  - {path}: {', '.join(differences)}")
    if mismatches:
        raise AnnotationSettingsError(
            f"Existing annotations in {annotations_dir} were created with other "
            "annotation settings than requested:\n"
            + "\n".join(mismatches)
            + "\nThey would be reused and merged with new annotations. Use a new "
            "output path (--output-path), or delete these files to annotate again "
            "with the requested settings."
        )


def _is_failed(row) -> bool:
    """Whether generation failed (see ff_modelgen.get_failure())."""
    failure = row.get("failure")
    return isinstance(failure, str) and bool(failure)


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
            ref_rows = {
                row["prompt"]: row
                for _, row in generations[reference_model].iterrows()
            }
            skipped = {"model failed": 0, "reference missing": 0, "reference empty": 0}

            # Create pairwise comparisons for each prompt
            for _, model_row in model_gens.iterrows():
                prompt = model_row["prompt"]
                ref_row = ref_rows.get(prompt)
                if _is_failed(model_row):
                    skipped["model failed"] += 1
                    continue
                if ref_row is None:
                    skipped["reference missing"] += 1
                    continue
                if _is_failed(ref_row):
                    # Kept (e.g. reference stuck in a loop is its real behavior),
                    # unless empty, since it is shared by all comparisons
                    if not str(ref_row["response"] or "").strip():
                        skipped["reference empty"] += 1
                        continue
                    logger.warning(
                        f"Keeping failed ({ref_row['failure']}) but non-empty "
                        f"reference generation of {reference_model} for prompt "
                        f"{ref_row['prompt_id']}."
                    )
                data.append(
                    {
                        "prompt": prompt,
                        "text_a": model_row["response"],
                        "text_b": ref_row["response"],
                        "model_a": model_name,
                        "model_b": reference_model,
                    }
                )
            if any(skipped.values()):
                logger.warning(
                    f"Skipped prompts for {model_name} vs {reference_model}: "
                    f"{ {k: v for k, v in skipped.items() if v} }"
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
    batch_max_wait: float = 24 * 60,
    annotator_model: str = DEFAULT_ANNOTATOR_MODEL,
    position_debias: bool = False,
):
    """Compare models' personalities.

    First creates generations for the given set of prompts.
    Then creates pairwise dataset for each model against the reference models.
    Then annotates the pairwise datasets with principle-following AI annotators.

    Args:
        prompts: List of prompts to compare.
        model_names: List of model names to compare. New OpenRouter models are
            generated via batch API if available, unless suffix ':nonbatch' is
            added (e.g. 'openrouter/openai/gpt-6-sol:nonbatch').
        reference_models: List of reference model names to compare against.
        output_path: Path to save the results.
        batch_max_wait: Minutes to wait for batches. Models with batches still
            pending afterwards are skipped (and picked up again in next run).
        annotator_model: Model used to annotate the pairwise datasets.
        position_debias: Annotate in both response orders, only keep agreeing votes.

    Raises:
        AnnotationSettingsError: if existing annotations in the output path were
            created with other annotation settings (checked before any API calls).
    """

    # Existing annotations are reused: check their settings before any API calls
    annotation_settings = get_annotation_settings(annotator_model, position_debias)
    final_annotations_dir = pathlib.Path(output_path) / "annotations"
    check_annotation_settings(final_annotations_dir, annotation_settings)

    batch_models = modelgen.get_batch_models(model_names, output_path)
    model_names = [modelgen.strip_nonbatch_suffix(m) for m in model_names]
    reference_models = [modelgen.strip_nonbatch_suffix(m) for m in reference_models]

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

    # Submit batches first, so they are processed while live generations run
    deadline = time.time() + batch_max_wait * 60
    for model_name in batch_models:
        modelgen.submit_batch(prompts, model_name, output_path)

    # Create generations for each model
    pending_models = []
    for model_name in model_names:
        if model_name in batch_models and not modelgen.collect_batch(
            prompts, model_name, output_path, deadline=deadline
        ):
            pending_models.append(model_name)
            continue
        logger.info(f"Creating generations for {model_name}")
        # Second pass retries prompts that failed in first pass
        # (for batch models: requests that failed within batch)
        for _ in range(2):
            asyncio.run(
                modelgen.run_model_on_prompts_async(
                    prompts, model_name, output_path, max_concurrent=10
                )
            )

    # Collect batches of models no longer compared, so that paid results are saved
    # (and their state files do not trigger follow-up runs forever)
    for model_name in modelgen.get_pending_batch_models(output_path):
        if model_name not in model_names:
            logger.warning(f"Collecting batch of model {model_name} not in models.")
            modelgen.collect_batch(prompts, model_name, output_path, deadline=deadline)

    if pending_models:
        logger.warning(
            f"Batches still pending for {pending_models}. "
            "Skipping these models until next run."
        )
        assert not set(pending_models) & set(
            reference_models
        ), "Batch for reference model still pending, run again later."
        model_names = [m for m in model_names if m not in pending_models]

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
    annotation_tmp_dir.mkdir(parents=True, exist_ok=True)
    final_annotations_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in pairwise_path.glob("*.csv"):
        logger.info(f"Annotating {csv_file}")
        tmp_output_dir = annotation_tmp_dir / csv_file.stem
        final_annotation_path = final_annotations_dir / (csv_file.stem + "_ap.json")

        if not final_annotation_path.exists():
            ap_path = annotate_dataset(
                datapath=csv_file,
                principles=get_default_principles(),
                model=annotation_settings["annotator_model"],
                output_dir=tmp_output_dir,
                position_debias=annotation_settings["position_debias"],
                max_output_tokens=annotation_settings["max_output_tokens"],
                reasoning_effort=annotation_settings["reasoning_effort"],
            )
            # Move the annotation to the final output path
            shutil.copyfile(src=ap_path, dst=final_annotation_path)
            with open(ap_path.parent / USAGE_SUMMARY_FILENAME) as f:
                this_run = json.load(f)["this_run"]
            failed = this_run["num_failed_calls"]
            truncated = this_run["num_truncated_calls"]
            unparsable = this_run["num_unparsable_responses"]
            if failed or truncated or unparsable:
                # Unlike with annotate_dataset, re-running does not repeat them
                logger.warning(
                    f"While annotating {csv_file}, {failed} API calls failed (no "
                    f"response after retries), {truncated} were truncated and "
                    f"{unparsable} had an unparsable response (votes invalid). Later "
                    f"runs reuse its annotations: to repeat these calls, delete "
                    f"{final_annotation_path} and re-run (other calls are not "
                    f"repeated as long as {tmp_output_dir} is kept)."
                )
        else:
            logger.info(f"Annotation already exists for {csv_file}. Skipping...")

    # Merge annotations
    logger.info("Merging annotations")

    ap_files = list(final_annotations_dir.glob("*.json"))
    ap_files = [load_ap(file) for file in ap_files]
    for ap in ap_files:
        # All files were created with these settings (checked above), also those
        # without recorded settings (ICAI-era)
        ap["metadata"]["annotation"] = annotation_settings
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

    parser.add_argument(
        "--batch-max-wait",
        type=float,
        default=24 * 60,
        help=(
            "Minutes to wait for batch generations (used by default for new "
            "OpenRouter models, disable per model with ':nonbatch' suffix). "
            "Models with pending batches are picked up again in next run "
            "(default: 1440)"
        ),
    )
    parser.add_argument(
        "--annotator-model",
        type=str,
        default=DEFAULT_ANNOTATOR_MODEL,
        help=f"Model used to annotate the comparisons (default: {DEFAULT_ANNOTATOR_MODEL})",
    )
    parser.add_argument(
        "--position-debias",
        action="store_true",
        help="Annotate each comparison in both response orders and only keep "
        "votes that agree (~2x annotation cost). See ff-annotate --help.",
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

    try:
        compare_models(
            prompts=prompts,
            model_names=args.models,
            reference_models=args.reference_models,
            output_path=args.output_path,
            batch_max_wait=args.batch_max_wait,
            annotator_model=args.annotator_model,
            position_debias=args.position_debias,
        )
    except AnnotationSettingsError as e:
        logger.error(str(e))
        return 1
    logger.success(
        f"Model personality comparison data generated successfully! To see results, run\n   feedback-forensics -d {args.output_path}/annotations/combined_ap.json"
    )
    return 0


if __name__ == "__main__":
    exit(run())

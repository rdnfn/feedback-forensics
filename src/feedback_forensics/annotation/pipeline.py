"""End-to-end annotation of a pairwise dataset with principle-following annotators.

Output directory layout (kept compatible with the former ICAI-based `ff-annotate`):

    <output_dir>/
        main.log                                 log of the run
        api_calls.jsonl                          start and result of each LLM API
                                                 call (tokens, cost, duration,
                                                 errors)
        results/
            000_train_data.csv                   input data as annotated
            030_distilled_principles_per_cluster.json   principle id -> principle
            040_votes_per_comparison.csv         votes per comparison
            041_votes_per_comparison_per_order.csv      (position debiasing only)
                                                 votes for each response order
            070_annotations_train_ap.json        annotated dataset (AnnotatedPairs)
            080_api_usage_summary.json           tokens, cost and failed calls of
                                                 all runs with this output dir
                                                 (and of the last run: "this_run"),
                                                 number of invalid votes; written
                                                 when a run completes
            cache/                               vote cache (enables resuming runs)
"""

import datetime
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

from feedback_forensics.annotation.usage import (
    UsageTracker,
    log_usage_summary,
    summarise_usage,
)
from feedback_forensics.annotation.voting import (
    INVALID_VOTE,
    VotingConfig,
    get_votes_for_principles,
)
from feedback_forensics.data.ap_format import (
    create_annotated_pairs,
    load_preference_data,
    save_annotated_pairs_to_file,
)
from feedback_forensics.models import check_model_settings

AP_OUTPUT_FILENAME = "070_annotations_train_ap.json"
PER_ORDER_VOTES_FILENAME = "041_votes_per_comparison_per_order.csv"
USAGE_SUMMARY_FILENAME = "080_api_usage_summary.json"
API_CALLS_LOG_FILENAME = "api_calls.jsonl"


def default_output_dir() -> Path:
    return Path("exp/outputs") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def annotate_dataset(
    datapath: str | Path,
    principles: list[str],
    model: str,
    output_dir: str | Path | None = None,
    position_debias: bool = False,
    max_concurrent_tasks: int = 400,
    max_principles_per_prompt: int = 40,
    max_output_tokens: int = 3000,
    reasoning_effort: str | None = None,
    log_api_calls: bool = False,
    dataset_name: str | None = None,
) -> Path:
    """Annotate a dataset with principle-following AI annotators.

    Args:
        datapath: CSV (columns text_a, text_b, optionally preferred_text, prompt,
            model_a, model_b, ...) or AnnotatedPairs JSON file.
        principles: Principles to annotate with
            (e.g. "Select the response that is more concise").
        model: Annotator model, e.g. "openrouter/google/gemini-2.5-flash".
        output_dir: Output directory (default: exp/outputs/<datetime>).
            Re-running with the same output directory resumes from the cache:
            votes are reused for the same model and settings, API calls that
            produced no valid vote (e.g. failed calls) are repeated.
        position_debias: Annotate each comparison in both response orders and
            only keep votes that agree (2x API calls).
        max_concurrent_tasks: Maximum number of concurrent API calls.
        max_principles_per_prompt: Maximum principles per annotation prompt.
        max_output_tokens: Maximum output tokens per annotation call (including
            reasoning tokens of reasoning models).
        reasoning_effort: Reasoning effort for reasoning models (None: default).
        log_api_calls: Also log full request messages and responses
            in api_calls.jsonl (token usage and cost are always logged).
        dataset_name: Name of the resulting dataset.

    Returns:
        Path to the AnnotatedPairs JSON file with the annotations.
    """
    # Validate settings before writing any files
    principles = _check_principles(principles)
    config = VotingConfig(
        model=model,
        max_principles_per_prompt=max_principles_per_prompt,
        max_output_tokens=max_output_tokens,
        max_concurrent_tasks=max_concurrent_tasks,
        reasoning_effort=reasoning_effort,
        position_debias=position_debias,
    )
    check_model_settings(model, reasoning_effort)
    output_dir = Path(output_dir) if output_dir is not None else default_output_dir()
    results_path = output_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    log_handler_id = logger.add(output_dir / "main.log")
    try:
        # Deduplicate principles while keeping order
        principles = list(dict.fromkeys(principles))
        principle_index_to_text = {i: p for i, p in enumerate(principles)}
        logger.info(f"Principles to be tested: {principles}")

        data = load_preference_data(datapath, merge_prompts=True)
        added_pseudo_preferred_text = "preferred_text" not in data.columns
        if added_pseudo_preferred_text:
            logger.warning(
                "Data has no preferred_text column. Adding pseudo preferred_text "
                "column that always selects the first text (text_a)."
            )
            data["preferred_text"] = "text_a"
        data.to_csv(
            results_path / "000_train_data.csv", index=True, index_label="index"
        )
        _save_json(
            principle_index_to_text,
            results_path / "030_distilled_principles_per_cluster.json",
        )

        usage_tracker = UsageTracker(
            output_dir / API_CALLS_LOG_FILENAME, log_content=log_api_calls
        )

        cache_name = "01_principle_votes" + ("_debiased" if position_debias else "")
        result = get_votes_for_principles(
            feedback_df=data,
            principles=principle_index_to_text,
            config=config,
            cache_path=results_path / "cache" / cache_name,
            usage_tracker=usage_tracker,
        )

        pd.Series(result.votes, name="votes", dtype=object).to_csv(
            results_path / "040_votes_per_comparison.csv",
            index=True,
            index_label="index",
        )
        if result.votes_per_order is not None:
            pd.DataFrame.from_dict(result.votes_per_order, orient="index").to_csv(
                results_path / PER_ORDER_VOTES_FILENAME,
                index=True,
                index_label="index",
            )
        else:
            # Remove per-order votes of an earlier position-debiased run
            (results_path / PER_ORDER_VOTES_FILENAME).unlink(missing_ok=True)

        annotated_pairs = create_annotated_pairs(
            df=data,
            principle_index_to_text=principle_index_to_text,
            comparison_votes=result.votes,
            dataset_name=dataset_name
            or f"Feedback Forensics Dataset - {output_dir.name}",
            auto_detect_annotators=True,
        )
        # Also read by ff-model-comparison, to only reuse annotations created
        # with the same settings (see `check_annotation_settings` there)
        annotated_pairs["metadata"]["annotation"] = {
            "annotator_model": model,
            "position_debias": position_debias,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
        }
        ap_path = results_path / AP_OUTPUT_FILENAME
        save_annotated_pairs_to_file(annotated_pairs, ap_path)

        all_votes = [v for votes in result.votes.values() for v in votes.values()]
        usage_summary = summarise_usage(
            usage_tracker,
            num_votes=len(all_votes),
            num_invalid_votes=sum(v == INVALID_VOTE for v in all_votes),
        )
        log_usage_summary(usage_summary)
        _save_json(usage_summary, results_path / USAGE_SUMMARY_FILENAME)

        if added_pseudo_preferred_text:
            logger.warning(
                "Used synthetic preferred_text column (always text_a). Interpret "
                "principle votes relative to text_a."
            )
        logger.info(
            f"Annotation finished. To inspect the results, run:\n\n"
            f"    feedback-forensics -d {ap_path}\n"
        )
        return ap_path
    finally:
        logger.remove(log_handler_id)


def _check_principles(principles) -> list[str]:
    """Principles as a list; TypeError if not a list (or other iterable) of strings,
    e.g. a string or a dict, which would be iterated over as characters or keys."""
    if not isinstance(principles, (str, Mapping)):
        principles = list(principles)
        if all(isinstance(principle, str) for principle in principles):
            return principles
    raise TypeError(
        f"principles must be a list of strings, got {principles!r:.200} (for a "
        "dict of id -> principle, pass its values)"
    )


def _save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

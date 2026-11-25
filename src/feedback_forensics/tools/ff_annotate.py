"""CLI for annotating datasets with principle-following AI annotators."""

import argparse
import json

from loguru import logger

from feedback_forensics.annotation import (
    AnnotatorAccountError,
    PromptDataError,
    annotate_dataset,
)
from feedback_forensics.annotation.batch import BatchError, BatchPendingError
from feedback_forensics.annotation.pipeline import default_output_dir
from feedback_forensics.annotation.principles import get_default_principles
from feedback_forensics.models.main import check_model_settings
from feedback_forensics.models.openrouter import check_batch_model

DEFAULT_ANNOTATOR_MODEL = "openrouter/google/gemini-2.5-flash"
EXIT_CODE_BATCH_PENDING = 3

__all__ = ["run", "get_default_principles"]


def parse_principles(value: str) -> list[str]:
    """Parse --principles: a JSON list of principles, or a JSON object mapping ids
    to principles (e.g. ICAI's results/030_distilled_principles_per_cluster.json)."""
    try:
        principles = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"invalid JSON ({e})") from e
    if isinstance(principles, dict):
        principles = list(principles.values())
    if not isinstance(principles, list) or not all(
        isinstance(principle, str) for principle in principles
    ):
        raise argparse.ArgumentTypeError(
            "expected a JSON list of principles (strings), or a JSON object "
            "mapping ids to principles"
        )
    return principles


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if not number >= 0:
        raise argparse.ArgumentTypeError(f"must not be negative, got {value}")
    return number


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI command for annotating datasets for Feedback Forensics."
    )
    parser.add_argument(
        "-d",
        "--datapath",
        type=str,
        required=True,
        help="Path to AnnotatedPairs JSON file or CSV file with columns text_a, text_b, and preferred_text.",
    )
    parser.add_argument(
        "-v",
        "--principles-version",
        type=str,
        default="v1",
        help=(
            "Version of standard principles to test (default: v1). "
            "A principle is an instruction for the AI annotator to "
            "select a text according to a given personality trait."
        ),
    )
    parser.add_argument(
        "-p",
        "--principles",
        type=parse_principles,
        default=None,
        help=(
            "Custom principles to test (default: None): a JSON list of principles, "
            'e.g. ["Select the response that is more confident", '
            '"Select the response that is more friendly"], or a JSON object mapping '
            "ids to principles (e.g. the content of ICAI's "
            "results/030_distilled_principles_per_cluster.json)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (default: exp/outputs/DATETIME)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_ANNOTATOR_MODEL,
        help=f"Model to use to annotate the data (default: {DEFAULT_ANNOTATOR_MODEL})",
    )
    parser.add_argument(
        "--position-debias",
        action="store_true",
        help=(
            "Annotate each comparison twice, with both response orders (A/B and B/A), "
            "and only keep votes that agree across orders. Reduces position bias of "
            "the AI annotator at ~2x annotation cost. Votes that disagree are stored "
            "without preference (reason: 'position_inconsistent')."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_int,
        default=3000,
        help=(
            "Maximum output tokens per annotation call, including reasoning tokens "
            "of reasoning models (default: 3000). Calls hitting the limit produce "
            "invalid votes and are reported at the end of the run."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help=(
            "Reasoning effort for reasoning models (default: provider default). "
            "Not supported for anthropic/ models: use openrouter/anthropic/... "
            "instead."
        ),
    )
    parser.add_argument(
        "--log-api-calls",
        action="store_true",
        help=(
            "Log full request messages and responses of all API calls to "
            "<output-dir>/api_calls.jsonl (token usage and cost are always logged)."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=positive_int,
        default=400,
        help="Maximum number of concurrent API calls (default: 400)",
    )
    parser.add_argument(
        "--without-separate-prompt",
        action="store_true",
        help=(
            "Annotate with the former annotation prompt, which shows only the two "
            "texts (a prompt column is merged into both texts, i.e. sent twice per "
            "API call). Needed for data without prompt column, or with the prompts "
            "already contained in the texts. By default, the user prompt is stated "
            "once before the two responses, and the data must have a prompt for "
            "every row. Votes of the two prompts are not comparable."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Annotate via OpenRouter's batch API: at reduced price, but results may "
            "take up to 24 hours (openrouter/ models with a batch variant only). If "
            "the batch is still pending after --batch-max-wait, the command exits "
            f"with code {EXIT_CODE_BATCH_PENDING}: re-run it with the same "
            "--output-dir and settings to collect the results."
        ),
    )
    parser.add_argument(
        "--batch-max-wait",
        type=non_negative_float,
        default=24 * 60,
        help=(
            "Minutes to wait for the results of --batch (default: 1440). "
            "0: only submit the batch, or collect it if it has completed."
        ),
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    """Annotate your data with principle-following AI annotators."""
    parser = get_parser()
    args = parser.parse_args(argv)
    try:
        check_model_settings(args.model, args.reasoning_effort)
        if args.batch:
            check_batch_model(args.model)
    except ValueError as e:
        parser.error(str(e))

    principles = []
    if args.principles_version is not None:
        principles.extend(get_default_principles(args.principles_version))
    if args.principles is not None:
        principles.extend(args.principles)

    if not principles:
        logger.error("No principles to annotate with. Set --principles(-version).")
        return 1

    # Set here (not by annotate_dataset), to tell how to resume a pending batch
    output_dir = args.output_dir or str(default_output_dir())
    try:
        annotate_dataset(
            datapath=args.datapath,
            principles=principles,
            model=args.model,
            output_dir=output_dir,
            position_debias=args.position_debias,
            max_concurrent_tasks=args.max_concurrent,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            log_api_calls=args.log_api_calls,
            batch=args.batch,
            batch_max_wait=args.batch_max_wait,
            without_separate_prompt=args.without_separate_prompt,
        )
    except (AnnotatorAccountError, PromptDataError) as e:
        # Message states the problem and how to continue: shown without traceback
        logger.error(str(e))
        return 1
    except BatchPendingError as e:
        logger.warning(
            f"{e}\nTo collect the results, re-run this command later with "
            f"--output-dir {output_dir}"
        )
        return EXIT_CODE_BATCH_PENDING
    except BatchError as e:
        logger.error(f"{e} (output directory: {output_dir})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

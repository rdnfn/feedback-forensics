"""CLI for annotating datasets with principle-following AI annotators."""

import argparse
import json

from loguru import logger

from feedback_forensics.annotation import annotate_dataset
from feedback_forensics.annotation.principles import get_default_principles
from feedback_forensics.models import check_model_settings

DEFAULT_ANNOTATOR_MODEL = "openrouter/google/gemini-2.5-flash"

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
    return parser


def run(argv: list[str] | None = None) -> int:
    """Annotate your data with principle-following AI annotators."""
    parser = get_parser()
    args = parser.parse_args(argv)
    try:
        check_model_settings(args.model, args.reasoning_effort)
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

    annotate_dataset(
        datapath=args.datapath,
        principles=principles,
        model=args.model,
        output_dir=args.output_dir,
        position_debias=args.position_debias,
        max_concurrent_tasks=args.max_concurrent,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        log_api_calls=args.log_api_calls,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

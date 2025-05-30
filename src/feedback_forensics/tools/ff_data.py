"""Swiss army knife for AnnotatedPairs datasets"""

import argparse
import sys
from pathlib import Path
from loguru import logger
import json

from feedback_forensics.data.operations import load_ap, save_ap, merge_ap


def merge_command(args):
    """Handle merge subcommand."""
    logger.info(f"Merging AnnotatedPairs: {args.first} + {args.second}")

    try:
        first_data = load_ap(args.first)
        second_data = load_ap(args.second)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1

    merged_data = merge_ap(first_data, second_data, args.name, args.desc)

    if args.output == "-":
        # Output to stdout if output is "-"
        logger.info("Outputting merged dataset to stdout")
        print(json.dumps(merged_data, indent=2))
    else:
        try:
            save_ap(merged_data, args.output)
        except FileNotFoundError as e:
            logger.error(f"Output directory does not exist.")
            return 1
        logger.info(f"Saved merged dataset to {args.output}")

    logger.info(f"Successfully merged datasets to {args.output}")
    logger.info(
        f"Result contains {len(merged_data['comparisons'])} comparisons and {len(merged_data['annotators'])} annotators"
    )
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Swiss army knife for AnnotatedPairs datasets", prog="ff-data"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    merge_parser = subparsers.add_parser(
        "merge", help="Merge two AnnotatedPairs datasets"
    )
    merge_parser.add_argument(
        "first", help="First dataset file (takes precedence in conflicts)"
    )
    merge_parser.add_argument("second", help="Second dataset file")
    merge_parser.add_argument(
        "output",
        help='Output file (use "-" for stdout)',
    )
    merge_parser.add_argument("--name", help="Override dataset name for merged result")
    merge_parser.add_argument("--desc", help="Override description for merged result")
    merge_parser.set_defaults(func=merge_command)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


def run():
    """Entry point for scripts."""
    sys.exit(main())


if __name__ == "__main__":
    run()

"""CLI for running ICAI experiments with simplified parameters."""

import argparse
import json
import subprocess
from loguru import logger

from inverse_cai.experiment.config import default_principles


def get_default_principles(version: str = "v1") -> list[str]:
    """Get the default principles."""
    match version:
        case "v1":
            return default_principles.DEFAULT_PRINCIPLES["v4"]
        case "null" | "None" | "":
            return []
        case _:
            raise ValueError(f"Unknown principles version: {version}.")


def run():
    """Annotate your data using the ICAI experiment pipeline with simplified parameters."""
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="CLI command for annotating datasets for Feedback Forensics with ICAI pipeline."
    )
    parser.add_argument(
        "-d",
        "--datapath",
        type=str,
        required=True,
        help="Path to dataset CSV file with columns text_a, text_b, and preferred_text",
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
        type=json.loads,
        default=None,
        help=(
            "List of custom principles to test (default: None). "
            "E.g. \"['Select the response that is more confident', "
            "'Select the response that is more friendly']\""
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
        default="openrouter/openai/gpt-4o-mini-2024-07-18",
        help="Model to use to annotate the data (default: openrouter/openai/gpt-4o-mini-2024-07-18)",
    )

    args = parser.parse_args()

    logger.info(
        f"Feedback Forensics is using the Inverse Constitutional AI (ICAI) pipeline to annotate your data."
    )

    # Set up the ICAI command with fixed parameters
    icai_cmd = [
        "icai-exp",
        f'data_path="{args.datapath}"',
        "annotator.skip=true",
        "s0_skip_principle_generation=true",
    ]

    if args.output_dir is not None:
        icai_cmd.append(f'hydra.run.dir="{args.output_dir}"')

    if args.model is not None:
        icai_cmd.append(f'alg_model="{args.model}"')

    # Collect principles to test
    principles_to_test = []
    if args.principles_version is not None:
        principles_to_test.extend(get_default_principles(args.principles_version))
    if args.principles is not None:
        principles_to_test.extend(args.principles)

    principle_str = json.dumps(principles_to_test, separators=(",", ":")).replace(
        '"', '\\"'
    )
    icai_cmd.append(f'"s0_added_principles_to_test={principle_str}"')

    cmd_str = " ".join(icai_cmd)

    logger.info(f"Running ICAI experiment: {cmd_str}")

    try:
        # Run the ICAI command
        process = subprocess.run(cmd_str, shell=True, check=True)

        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"ICAI experiment failed with exit code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

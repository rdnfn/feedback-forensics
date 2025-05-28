"""CLI for running ICAI experiments with simplified parameters."""

import argparse
import subprocess
from loguru import logger


def run():
    """Run the ICAI experiment CLI with simplified parameters."""
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run an ICAI experiment with simplified parameters"
    )
    parser.add_argument(
        "--datapath",
        "-d",
        type=str,
        required=True,
        help="Path to dataset CSV file with columns text_a, text_b, and preferred_text",
    )
    parser.add_argument(
        "--principles-version",
        "-p",
        type=str,
        default="v4",
        help="Version of standard principles to test (default: v2)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Path to output directory (default: exp/outputs/DATETIME)",
    )

    args = parser.parse_args()

    logger.info(
        f"Feedback Forensics is using the Inverse Constitutional AI (ICAI) pipeline to annotate your data."
    )

    # Set up the ICAI command with fixed parameters
    icai_cmd = [
        "icai-exp",
        f'data_path="{args.datapath}"',
        f's0_added_standard_principles_to_test="[{args.principles_version}]"',
        "annotator.skip=true",
        "s0_skip_principle_generation=true",
    ]

    if args.output_dir is not None:
        icai_cmd.append(f'hydra.run.dir="{args.output_dir}"')

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

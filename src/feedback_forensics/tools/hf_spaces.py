#!/usr/bin/env python3
"""
Command line tool to trigger a rebuild of a Hugging Face Space.
"""

import argparse
import os
import sys
from huggingface_hub import HfApi
from loguru import logger


def rebuild_space(repo_id: str) -> None:
    """
    Trigger a rebuild of a Hugging Face Space.

    Args:
        repo_id: The repository ID of the Hugging Face Space (format: username/space-name)
        token: Hugging Face API token. If None, will use the token from the Hugging Face CLI.
    """
    try:
        # Initialize the Hugging Face API client
        api = HfApi(token=os.environ["HF_REBUILD_TOKEN"])

        # Trigger the rebuild
        logger.info(f"Triggering rebuild for Space: {repo_id}")
        api.restart_space(repo_id=repo_id, factory_reboot=True)
        logger.success(f"Successfully triggered rebuild for Space: {repo_id}")

    except Exception as e:
        logger.error(f"Failed to trigger rebuild: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point for the command line tool."""
    parser = argparse.ArgumentParser(
        description="Trigger a rebuild of a Hugging Face Space."
    )

    parser.add_argument(
        "repo_id",
        type=str,
        help="The repository ID of the Hugging Face Space (format: username/space-name)",
    )

    args = parser.parse_args()

    rebuild_space(repo_id=args.repo_id)


if __name__ == "__main__":
    main()

import subprocess
import os
import pathlib
from loguru import logger

from feedback_forensics.app.constants import GITHUB_TOKEN


DATA_DIR = pathlib.Path("icai-tmp-data")


def clone_repo(username, token, repo_name, clone_directory):
    """
    Clones a GitHub repository into a specified directory using subprocess.

    Args:
    username (str): GitHub username
    token (str): Personal access token for GitHub
    repo_name (str): Name of the repository to be cloned
    clone_directory (str): Local directory where the repository should be cloned
    """
    # ensure the clone directory exists
    dir_exists = os.path.exists(clone_directory / repo_name)

    if dir_exists:
        logger.info(
            f"Standard data repository '{repo_name}' already exists in '{clone_directory}'. Skipping loading data from repo. Delete directory to re-clone standard data."
        )
        return None

    pathlib.Path(clone_directory).mkdir(parents=True, exist_ok=True)

    # Form the complete GitHub URL with credentials

    # Form the complete GitHub URL with credentials
    if token:
        logger.info("Using token and https to clone data from repository.")
        git_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
        # Execute the git clone command
        subprocess.run(
            [f"git clone {git_url}"], shell=True, check=True, cwd=clone_directory
        )
    else:
        # try via ssh
        logger.info("Using SSH to clone data from repository.")
        git_url = f"git@github.com:{username}/{repo_name}.git"
        subprocess.run(
            [f"git clone {git_url}"], shell=True, check=True, cwd=clone_directory
        )
    logger.info("Data loaded from repo successfully.")


def load_icai_data():
    """
    Load the data from the InverseCAI repository.
    """
    # Define the repository name
    username = "rdnfn"
    repo_name = "icai-data"

    # Define the local directory where the repository should be cloned
    # get package directory
    clone_directory = DATA_DIR

    try:
        # Clone the repository
        clone_repo(username, GITHUB_TOKEN, repo_name, clone_directory)
    except Exception as e:
        logger.warning(
            f"Failed to load standard data from repo (error: '{e}'). Skipping this step."
        )

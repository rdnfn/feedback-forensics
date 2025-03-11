import subprocess
import os
import shutil
import pathlib
from loguru import logger

from feedback_forensics.app.constants import HF_TOKEN


CLONE_DIR = pathlib.Path("forensics-data")
REPO_USERNAME = "rdnfn"
REPO_NAME = "feedback-forensics-public-results"
REPO_PROVIDER = "huggingface.co/datasets"
DATA_DIR = CLONE_DIR / REPO_NAME


def clone_repo(username, token, repo_name, clone_directory, provider="github.com"):
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

    # check if git-lfs is installed
    if not shutil.which("git-lfs"):
        logger.warning(
            (
                "git-lfs is not installed. Skipping data loading from repo. "
                "Check https://github.com/git-lfs/git-lfs for installation instructions."
            )
        )
        return None

    pathlib.Path(clone_directory).mkdir(parents=True, exist_ok=True)

    # Form the complete GitHub URL with credentials
    if token:
        logger.info("Using token and https to clone data from repository.")
        git_url = f"https://{username}:{token}@{provider}/{username}/{repo_name}.git"
        # Execute the git clone command
        subprocess.run(
            [f"git clone {git_url}"], shell=True, check=True, cwd=clone_directory
        )
    else:
        # try via ssh
        logger.info("Using SSH to clone data from repository.")
        git_url = f"git@{provider}:{username}/{repo_name}.git"
        subprocess.run(
            [f"git clone {git_url}"], shell=True, check=True, cwd=clone_directory
        )
    logger.info("Data loaded from repo successfully.")


def load_icai_data():
    """
    Load the public results from HuggingFace repository.
    """
    # Define the repository name
    username = REPO_USERNAME
    repo_name = REPO_NAME

    # Define the local directory where the repository should be cloned
    # get package directory
    clone_directory = CLONE_DIR

    if not HF_TOKEN:
        logger.warning(
            "HF_TOKEN environment variable not set. Cannot load datasets from HuggingFace."
        )
        return False

    try:
        # Clone the repository
        logger.info(
            f"Attempting to clone repository {username}/{repo_name} from {REPO_PROVIDER}..."
        )
        clone_repo(
            username,
            HF_TOKEN,
            repo_name,
            clone_directory,
            provider=REPO_PROVIDER,
        )
        return True
    except Exception as e:
        logger.error(
            f"Failed to load standard data from repo (error: '{e}'). "
            "Please verify your HF_TOKEN has permissions to access the repository."
        )
        return False

import subprocess
import os
import shutil
import pathlib
from loguru import logger


def clone_repo(username, repo_name, clone_directory, provider="github.com", token=None):
    """
    Clones a GitHub repository into a specified directory using subprocess.

    Args:
    username (str): GitHub username
    token (str): Personal access token for GitHub
    repo_name (str): Name of the repository to be cloned
    clone_directory (str): Local directory where the repository should be cloned
    """
    logger.info(
        f"Attempting to clone repository {username}/{repo_name} from {provider}..."
    )

    # check if already cloned
    if os.path.exists(clone_directory / repo_name):
        logger.info(
            f"Repository '{repo_name}' already exists in '{clone_directory}'. Skipping loading data from repo. Delete directory to re-clone data."
        )
        return True

    # check if git-lfs is installed
    if not shutil.which("git-lfs"):
        logger.warning(
            (
                "git-lfs is not installed. Skipping data loading from repo. "
                "Check https://github.com/git-lfs/git-lfs for installation instructions."
            )
        )
        return False

    pathlib.Path(clone_directory).mkdir(parents=True, exist_ok=True)

    # set up git url
    if token:
        logger.info("Using token and https to clone data from repository.")
        git_url = f"https://{username}:{token}@{provider}/{username}/{repo_name}.git"
    else:
        logger.info("Using https to clone data from repository.")
        git_url = f"https://{provider}/{username}/{repo_name}.git"

    try:
        logger.warning(
            "Note: With large datasets, this may take a while! Git often underestimates overall size due to omitting large files (when using git lfs). Patience may be required."
        )
        subprocess.run(
            [f"git clone {git_url}"], shell=True, check=True, cwd=clone_directory
        )
        logger.info("Data loaded from repo successfully.")
        return True
    except Exception as e:
        logger.error(
            f"Failed to load standard data from repo (error: '{e}'). "
            "Please verify your HF_TOKEN has permissions to access the repository."
        )
        return False

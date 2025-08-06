import subprocess
import os
import shutil
import pathlib
import requests
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


def clone_file(
    username,
    repo_name,
    file_path,
    destination_dir,
    provider="github.com",
    token=None,
    ref="HEAD",
):
    """
    Clones a single file from a git repository using git archive.
    """
    logger.info(
        f"Attempting to clone file '{file_path}' from {username}/{repo_name}..."
    )

    if ref is None:
        ref = "HEAD"

    pathlib.Path(destination_dir).mkdir(parents=True, exist_ok=True)

    if token:
        url_base = f"https://{username}:{token}@{provider}/{username}/{repo_name}"
    else:
        url_base = f"https://{provider}/{username}/{repo_name}"

    download_url = f"{url_base}/resolve/{ref}/{file_path}"
    file_name = file_path.split("/")[-1] if "/" in file_path else file_path

    try:
        logger.info(f"Downloading file from {download_url}...")
        response = requests.get(
            download_url,
            timeout=60 * 5,  # max 5 mins
        )
        with open(destination_dir / file_name, "wb") as f:
            f.write(response.content)
        logger.info("File cloned successfully.")
        return True

    except Exception as e:
        logger.error(f"Failed to clone file (error: '{e}').")
        return False

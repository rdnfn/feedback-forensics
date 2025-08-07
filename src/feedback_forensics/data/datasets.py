"""Module with configurations for built-in datasets."""

from dataclasses import dataclass
import pathlib
import re
import copy
from loguru import logger

from feedback_forensics.app.constants import DEFAULT_DATASET_NAMES, DATA_DIR, HF_TOKEN
from feedback_forensics.data.fetcher import clone_repo, clone_file
from feedback_forensics.data.dataset_utils import get_first_json_key_value


@dataclass
class BuiltinDataset:
    """Class to represent a built-in dataset."""

    name: str
    path: str | None = None
    description: str | None = None
    filterable_columns: list[str] | None = None
    source: str | None = None

    @property
    def url_name(self) -> str:
        """Get the name of the dataset for the URL."""
        # remove special characters and spaces, strictly alphanumeric lowercase, allow _ in the middle
        return (
            re.sub(r"[^a-zA-Z0-9_]", "", self.name.replace(" ", "_"))
            .lower()
            .strip(" _-")
        )


_available_datasets = []


def get_dataset_from_ap_json(file_path: str | pathlib.Path) -> BuiltinDataset:
    """Get a dataset config from AnnotatedPairs json file."""

    # load metadata from AnnotatedPairs file (first key value pair)
    key, metadata = get_first_json_key_value(file_path)

    if key != "metadata":
        logger.warning(
            f"Failed to load dataset from: '{file_path}'. "
            f"Expected first key to be 'metadata', got {key}. "
            f"This is not a valid AnnotatedPairs file. Skipping..."
        )
        return None

    logger.info(f"Loaded dataset from: '{file_path}'")
    return BuiltinDataset(
        name=metadata.get("dataset_name", f"Unnamed dataset ({file_path})"),
        path=file_path,
        description=metadata.get("description", "No description"),
        source=metadata.get("source", "Unknown source"),
        filterable_columns=metadata.get("filterable_columns", None),
    )


def get_datasets_from_dir(dir_path: str | pathlib.Path) -> list[BuiltinDataset]:
    """Get all AnnotatedPairs datasets inside dir."""
    logger.info(f"Loading datasets from directory: {dir_path}")
    datasets = []
    for file_path in pathlib.Path(dir_path).glob("*.json"):
        dataset = get_dataset_from_ap_json(file_path)
        if dataset is not None:
            datasets.append(dataset)
    logger.info(f"Loaded {len(datasets)} datasets from: '{dir_path}'")
    return datasets


def get_urlname_from_stringname(stringname: str, datasets: list[BuiltinDataset]) -> str:
    """Get the URL name from the string name."""
    for dataset in datasets:
        if dataset.name == stringname:
            return dataset.url_name
    logger.warning(f"Dataset with name '{stringname}' not found.")
    return None


def get_stringname_from_urlname(urlname: str, datasets: list[BuiltinDataset]) -> str:
    """Get the string name from the URL name."""
    for dataset in datasets:
        if dataset.url_name == urlname:
            return dataset.name
    logger.warning(f"Dataset with URL name '{urlname}' not found.")
    return None


def load_datasets_from_repo(
    repo_username: str,
    repo_name: str,
    repo_provider: str,
    token: str | None = None,
    data_extraction_subdir: str | None = None,
    file_path: str | None = None,
    ref: str | None = None,
):
    """
    Load datasets from git repo, e.g. on HuggingFace (HF).

    Args:
        repo_username (str): Username of repo
        repo_name (str): Name of repo
        repo_provider (str): Provider of the git repo (e.g. "huggingface.co/datasets")
        token (str): Token for the repo (optional, required for private repos)
        data_extraction_subdir (str): Subdirectory of repo to extract datasets from (optional)
        file_path (str): Path to file to extract from repo (optional), if provided, will only extract the file
        ref (str): Reference to use for the repo (optional)

    Returns:
        int: Number of datasets successfully loaded
    """
    global _available_datasets

    # Try to load the datasets from repo
    logger.info(f"Attempting to load datasets from {repo_provider}...")

    if not file_path:
        success = clone_repo(
            username=repo_username,
            repo_name=repo_name,
            clone_directory=DATA_DIR,
            provider=repo_provider,
            token=token,
        )
    else:
        success = clone_file(
            username=repo_username,
            repo_name=repo_name,
            file_path=file_path,
            destination_dir=DATA_DIR / repo_name,
            provider=repo_provider,
            token=token,
            ref=ref,
        )

    if success:
        extraction_dir = (
            DATA_DIR / repo_name / data_extraction_subdir
            if data_extraction_subdir
            else DATA_DIR / repo_name
        )
        loaded_datasets = get_datasets_from_dir(extraction_dir)
        _available_datasets += loaded_datasets
        loaded_count = len(loaded_datasets)
    else:
        loaded_count = 0

    if success and loaded_count > 0:
        logger.info(
            f"Successfully loaded {loaded_count} datasets from {repo_provider}."
        )
    elif success and loaded_count == 0:
        logger.error(
            "No datasets found in HuggingFace repository despite successful clone. Check repository contents."
        )
    elif not success:
        logger.error(
            f"Failed to load datasets from {repo_provider}. Check your HF_TOKEN permissions."
        )

    return loaded_count


def load_standard_web_datasets():
    """Load standard openly accessible datasets from HuggingFace.

    Triggered via command line flag or webapp mode.
    """
    load_datasets_from_repo(
        repo_username="rdnfn",
        repo_name="ff-public-results",
        repo_provider="huggingface.co/datasets",
        data_extraction_subdir="data/main",
    )


def load_webapp_datasets():
    """Load special gated datasets for the webapp from HuggingFace.

    Triggered via webapp mode only.
    """
    if HF_TOKEN is None:
        logger.error("HF_TOKEN is not set. Skipping loading gated datasets.")
    else:
        # load gated datasets
        load_datasets_from_repo(
            repo_username="rdnfn",
            repo_name="ff-gated-results",
            repo_provider="huggingface.co/datasets",
            data_extraction_subdir="data/main",
            token=HF_TOKEN,
        )

    # load model personality dataset
    # (autogenerated, saved to its own repo)
    load_datasets_from_repo(
        repo_username="rdnfn",
        repo_name="ff-model-personality",
        repo_provider="huggingface.co/datasets",
        file_path="data/v1/annotations/combined_ap.json",
    )


def get_available_datasets():
    """
    Get the current list of all available datasets.

    This function will always return the most up-to-date list of available datasets
    that have been loaded, including both standard datasets from HuggingFace
    and any locally added datasets.

    Returns:
        list[BuiltinDataset]: List of all available datasets
    """
    return _available_datasets


def get_available_datasets_names():
    """Get the names of all available datasets."""
    avail_datasets = copy.deepcopy(get_available_datasets())
    avail_datasets.sort(key=lambda x: x.url_name)
    names = [dataset.name for dataset in avail_datasets]
    return names


def get_default_dataset_names():
    """Get the names of the default datasets."""
    dataset_names = get_available_datasets_names()
    if len(DEFAULT_DATASET_NAMES) > 0:
        return DEFAULT_DATASET_NAMES
    else:
        return [dataset_names[-1]] if dataset_names else []


def add_dataset(dataset):
    """
    Add a dataset to the list of available datasets.

    Args:
        dataset (BuiltinDataset): Dataset to add
    """
    global _available_datasets
    _available_datasets.append(dataset)

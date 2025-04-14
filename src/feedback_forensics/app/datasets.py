"""Module with configurations for built-in datasets."""

from dataclasses import dataclass, field
from loguru import logger
import pathlib
import re
from feedback_forensics.app.constants import NONE_SELECTED_VALUE, DEFAULT_DATASET_NAMES
from feedback_forensics.app.data_loader import load_icai_data, DATA_DIR


@dataclass
class BuiltinDataset:
    """Class to represent a built-in dataset."""

    name: str
    path: str | None = None
    description: str | None = None
    options: list | None = None
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


@dataclass
class Config:
    """Class to represent a configuration."""

    name: str
    show_individual_prefs: bool = False
    pref_order: str = "By reconstruction success"
    plot_col_name: str = NONE_SELECTED_VALUE
    plot_col_values: list = field(default_factory=lambda: [NONE_SELECTED_VALUE])
    filter_col: str = NONE_SELECTED_VALUE
    filter_value: str = NONE_SELECTED_VALUE
    filter_col_2: str = NONE_SELECTED_VALUE
    filter_value_2: str = NONE_SELECTED_VALUE
    metrics: list = field(default_factory=lambda: ["perf", "relevance", "acc"])


# Builtin datasets

ANTHROPIC_HELPFUL = BuiltinDataset(
    name="🚑 Anthropic helpful",
    path=DATA_DIR / "anthropic_helpful",
    description="5k subsample of human preference pairs favouring helpful responses from RLHF dataset by Anthropic.",
    source="https://github.com/anthropics/hh-rlhf",
)


ANTHROPIC_HARMLESS = BuiltinDataset(
    name="🕊️ Anthropic harmless",
    path=DATA_DIR / "anthropic_harmless",
    description="5k subsample of human preference pairs favouring harmless responses from RLHF dataset by Anthropic.",
    source="https://github.com/anthropics/hh-rlhf",
)

ARENA_V2 = BuiltinDataset(
    name="🏟️ Chatbot Arena",
    path=DATA_DIR / "arena",
    description="10k subsample of Chatbot Arena dataset (100k) released alongside Arena Explorer work, crowdsourced human annotations from between June and August 2024 in English.",
    source="https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k",
)

ALPACA_EVAL_V2 = BuiltinDataset(
    name="🦙 AlpacaEval",
    path=DATA_DIR / "alpacaeval_human",
    description="648 cross-annotated human preference pairs used to validate AlpacaEval annotators.",
    source="https://huggingface.co/datasets/tatsu-lab/alpaca_eval/",
)

PRISM_V2 = BuiltinDataset(
    name="💎 PRISM",
    path=DATA_DIR / "prism",
    description="~8k human preference pairs from PRISM dataset, focused on controversial topics with extensive annotator information. Originally four-way annotations, subsampled using 1-of-3 rejected responses to get pairwise preferences.",
    source="https://huggingface.co/datasets/HannahRoseKirk/prism-alignment",
    filterable_columns=[
        "age",
        "education",
        "chosen_model",
        "rejected_model",
        "conversation_type",
        "lm_familiarity",
        "location_reside_region",
        "english_proficiency",
    ],
)

OLMO2_0325 = BuiltinDataset(
    name="🏋️ OLMo-2 0325 pref-mix",
    path=DATA_DIR / "olmo2-0325-32b",
    description="10k preference pairs subsampled randomly from original 378k pairs used for fine-tuning OLMo 2 model by Ai2. Synthetically generated via multiple different pipelines.",
    source="https://huggingface.co/datasets/allenai/olmo-2-0325-32b-preference-mix",
)

MULTIPREF = BuiltinDataset(
    name="🔄 MultiPref",
    path=DATA_DIR / "multipref_10k_v3.json",
    description="10k preference pairs annotated by 4 human annotators, as well as GPT-4-based AI annotators.",
    source="https://huggingface.co/datasets/allenai/multipref",
)

LLAMA4_ARENA = BuiltinDataset(
    name="🏟️ Arena (special)",
    path=DATA_DIR / "arena_llama4.json",
    description="Llama-4-Maverick-03-26-Experimental arena results, combined with public weights version of Llama-4-Maverick.",
    source="https://huggingface.co/spaces/lmarena-ai/Llama-4-Maverick-03-26-Experimental_battles/tree/main/data",
)

# List of all built-in datasets
_BUILTIN_DATASETS = [
    ARENA_V2,
    ALPACA_EVAL_V2,
    PRISM_V2,
    ANTHROPIC_HELPFUL,
    ANTHROPIC_HARMLESS,
    OLMO2_0325,
    MULTIPREF,
    LLAMA4_ARENA,
]
_available_datasets = []


# utility functions
def get_config_from_name(name: str, config_options: list) -> Config:
    """Get a configuration from its name."""
    if name == NONE_SELECTED_VALUE or name is None:  # default config
        return Config(name=name)

    for config in config_options:
        if config.name == name:
            return config

    raise ValueError(f"Configuration with name '{name}' not found.")


def get_dataset_from_name(name: str) -> BuiltinDataset:
    """Get a dataset from its name."""
    for dataset in get_available_datasets():
        if dataset.name == name:
            logger.info(f"Loading dataset '{name}'", duration=5)
            return dataset

    raise ValueError(f"Dataset with name '{name}' not found.")


def get_available_builtin_datasets() -> list[BuiltinDataset]:
    """Get all built-in datasets."""
    # validate that the relevant data is present for each dataset
    available_datasets = []
    for dataset in _BUILTIN_DATASETS:
        if dataset.path is not None and dataset.path.exists():
            logger.info(f"Found dataset: {dataset.name} at {dataset.path}")
            available_datasets.append(dataset)
        else:
            if dataset.path is not None:
                logger.warning(
                    f"Dataset path does not exist: {dataset.path} for {dataset.name}"
                )
    return available_datasets


def create_local_dataset(path: str, name: str = "🏠 Local dataset") -> BuiltinDataset:
    """Create a local dataset."""
    return BuiltinDataset(
        name=name,
        path=pathlib.Path(path),
        description=f"Local dataset from path {path}.",
        source=f"{path}",
    )


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


def load_datasets_from_hf():
    """
    Load datasets from HuggingFace.

    This function attempts to clone the HuggingFace repository containing datasets.

    Returns:
        int: Number of datasets successfully loaded
    """
    global _available_datasets

    # Try to load the datasets from HuggingFace
    logger.info("Attempting to load datasets from HuggingFace...")
    success = load_icai_data()

    # Refresh the available datasets
    _available_datasets += get_available_builtin_datasets()

    loaded_count = len(_available_datasets)
    if success and loaded_count > 0:
        logger.info(f"Successfully loaded {loaded_count} datasets from HuggingFace.")
    elif success and loaded_count == 0:
        logger.error(
            "No datasets found in HuggingFace repository despite successful clone. Check repository contents."
        )
    elif not success:
        logger.error(
            "Failed to load datasets from HuggingFace. Check your HF_TOKEN permissions."
        )

    return loaded_count


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
    return [dataset.name for dataset in _available_datasets]


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

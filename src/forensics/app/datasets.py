"""Module with configurations for built-in datasets."""

from dataclasses import dataclass, field
from loguru import logger
import pathlib
import re
from forensics.app.constants import NONE_SELECTED_VALUE
from forensics.app.data_loader import load_icai_data, DATA_DIR

load_icai_data()


@dataclass
class BuiltinDataset:
    """Class to represent a built-in dataset."""

    name: str
    path: str | None = None
    description: str | None = None
    options: list | None = None
    filterable_columns: list[str] | None = None

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
    description="",
)


ANTHROPIC_HARMLESS = BuiltinDataset(
    name="🕊️ Anthropic harmless",
    path=DATA_DIR / "anthropic_harmless",
    description="",
)

ARENA_V2 = BuiltinDataset(
    name="🏟️ Chatbot Arena",
    path=DATA_DIR / "arena",
    description="",
)

ALPACA_EVAL_V2 = BuiltinDataset(
    name="🦙 AlpacaEval",
    path=DATA_DIR / "alpacaeval_human",
    description="",
)

PRISM_V2 = BuiltinDataset(
    name="💎 PRISM",
    path=DATA_DIR / "prism",
    description="",
)

# List of all built-in datasets
_BUILTIN_DATASETS = [
    ARENA_V2,
    ALPACA_EVAL_V2,
    PRISM_V2,
    ANTHROPIC_HELPFUL,
    ANTHROPIC_HARMLESS,
]


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
    for dataset in BUILTIN_DATASETS:
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
            available_datasets.append(dataset)
    return available_datasets


def create_local_dataset(path: str) -> BuiltinDataset:
    """Create a local dataset."""
    return BuiltinDataset(
        name="🏠 Local dataset",
        path=pathlib.Path(path),
        description="Local dataset.",
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


BUILTIN_DATASETS = get_available_builtin_datasets()

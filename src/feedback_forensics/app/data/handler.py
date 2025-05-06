"""Module with the main data handler for the app.

The data handler allows efficient loading of AnnotatedPairs datasets
and computation of annotation metrics. It provides a unified interface
for loading data from different sources and computing metrics.
"""

import copy
import pandas as pd
from loguru import logger

from feedback_forensics.app.data.loader import add_virtual_annotators, get_votes_dict


def _get_annotator_df_col_names(
    annotator_visible_names: list[str], votes_dicts: dict[str, dict]
) -> list[str]:
    """Get the column names of the annotators in votes_df from a list of visible names.

    Note that this fn can be used both for annotators shown in rows and columns of
    the final output plot. All annotators are columns in the original votes_df.

    This also gives warning if not all annotators are available across all datasets.
    """
    # get mappings from visible names to column names for each dataset
    visible_to_cols = {}
    for dataset_name, votes_dict in votes_dicts.items():
        metadata = votes_dict["annotator_metadata"]
        visible_to_col = {
            value["annotator_visible_name"]: col for col, value in metadata.items()
        }
        visible_to_cols[dataset_name] = visible_to_col

    updated_annotator_visible_names = copy.deepcopy(annotator_visible_names)

    for annotator_name in annotator_visible_names:
        for dataset_name, visible_to_col in visible_to_cols.items():
            if annotator_name not in visible_to_col:
                logger.warning(
                    f"Annotator '{annotator_name}' (visible name) not found in dataset '{dataset_name}'. Skipping this annotator."
                )
                # remove annotator from annotator_visible_names
                updated_annotator_visible_names.remove(annotator_name)

    # get column names for the remaining annotators
    col_names = [
        visible_to_col[annotator_name]
        for annotator_name in updated_annotator_visible_names
    ]
    return col_names


def _split_votes_dict(
    votes_dict: dict,
    split_col: str,
    selected_vals: list[str] | None = None,
) -> dict:
    """Split votes data by split_col.

    First assert that only one votes_df in vote_dfs, and split that votes_df into multiple, based on the unique values of split_col.

    Args:
        votes_dict: Dictionary with keys "df" and "annotator_metadata"
        split_col: Column to split on
        selected_vals: Optional list of values to filter split_col by. If None, use all values.

    Returns:
        Dictionary mapping split values to filtered DataFrames
    """

    votes_df = votes_dict["df"]
    votes_df[split_col] = votes_df[split_col].astype(str)

    if selected_vals:
        # Filter to only selected values before grouping
        votes_df = votes_df[votes_df[split_col].isin(selected_vals)]

    grouped_df = votes_df.groupby(split_col)

    split_dicts = {}
    for name, group in grouped_df:
        split_dicts[name] = {
            "df": group,
            "annotator_metadata": votes_dict["annotator_metadata"],
            "reference_annotator_col": votes_dict["reference_annotator_col"],
            "shown_annotator_rows": votes_dict["shown_annotator_rows"],
        }

    return split_dicts


class DatasetHandler:
    """Class to handle data operations (loading, computing metrics) of a single annotation dataset."""

    def __init__(self, cache: dict | None = None, avail_datasets: dict | None = None):

        self.cache = cache
        self.avail_datasets = avail_datasets

        self._votes_dict = {}
        self.data_path = None

    @property
    def votes_dict(self):
        return self._votes_dict

    @property
    def default_annotator_rows(self):
        """Default annotator rows for the votes_dict."""
        return self.votes_dict["shown_annotator_rows"]

    @property
    def default_annotator_cols(self):
        """Default annotator cols for the votes_dict."""
        return [self.votes_dict["reference_annotator_col"]]

    def _get_data_path(self, dataset_name: str):
        """Get the data path for a given dataset name.

        The path is extracted from the available dataset configs.

        Args:
            dataset_name (str): The name of the dataset to get the path for.

        Returns:
            str: The path to the dataset.
        """

        if dataset_name not in self.avail_datasets:
            raise ValueError(
                f"Dataset {dataset_name} not found in avail_datasets ({self.avail_datasets})"
            )

        dataset_config = self.avail_datasets[dataset_name]
        return dataset_config.path

    def load_data_from_name(self, name: str):
        """Load data from a given dataset name."""

        assert self.avail_datasets is not None, (
            "Avail_datasets must be provided to load data from name. "
            "Use load_data_from_path instead if you have a path."
        )

        logger.info(f"Loading data from name: {name}")
        self.data_path = self._get_data_path(name)
        self.load_data_from_path(self.data_path)

        return self.votes_dict

    def load_data_from_path(self, dataset_path: str):
        """Load data from a given path."""

        base_votes_dict = get_votes_dict(dataset_path, cache=self.cache)
        votes_dict = add_virtual_annotators(
            base_votes_dict,
            cache=self.cache,
            dataset_cache_key=dataset_path,
            reference_models=[],
            target_models=[],
        )

        self.votes_dict = votes_dict
        logger.info(f"Loaded data from path: {dataset_path}")

    def load_from_votes_dict(self, votes_dict: dict):
        """Load data from a given votes_dict."""
        self.votes_dict = votes_dict

    def set_visible_annotator_rows(self, annotator_rows_visible_names: list[str]):
        """Change the visible annotator rows for the votes_dict."""
        if not annotator_rows_visible_names or len(annotator_rows_visible_names) == 0:
            logger.warning(
                "No annotator rows visible names provided. "
                "Showing all annotator rows."
            )
            return

        annotator_row_keys = _get_annotator_df_col_names(
            annotator_rows_visible_names, self.votes_dict
        )
        self._votes_dict["shown_annotator_rows"] = annotator_row_keys

    def compute_metrics(self, data: pd.DataFrame):
        """Compute metrics from a given dataset."""
        pass


class MultiDatasetHandler:
    """Class to handle data operations (loading, computing metrics) of multiple annotation datasets."""

    def __init__(self, cache: dict | None = None, avail_datasets: dict | None = None):
        self.cache = cache
        self.avail_datasets = avail_datasets

        self._handlers = {}

    @property
    def handlers(self):
        """Get dataset handlers."""
        return self._handlers

    @property
    def first_handler(self):
        """Get the first dataset handler."""
        return list(self._handlers.values())[0]

    def add_data_from_name(self, name: str):
        """Load data from a given dataset name."""
        assert (
            name not in self._handlers
        ), f"Dataset {name} already loaded or using duplicate name"

        self._handlers[name] = DatasetHandler(
            cache=self.cache, avail_datasets=self.avail_datasets
        )
        self._handlers[name].load_data_from_name(name)

    def load_data_from_names(self, names: list[str]):
        """Load data from a given list of dataset names."""
        for name in names:
            self.add_data_from_name(name)

    def add_data_from_votes_dict(self, votes_dict: dict, name: str):
        """Add data from a given votes_dict."""
        assert (
            name not in self._handlers
        ), f"Dataset {name} already loaded or using duplicate name"

        self._handlers[name] = DatasetHandler(
            cache=self.cache, avail_datasets=self.avail_datasets
        )
        self._handlers[name].load_from_votes_dict(votes_dict)

    def load_data_from_votes_dicts(self, votes_dicts: dict[str, dict]):
        """Load data from a given list of votes_dicts."""
        for name, votes_dict in votes_dicts.items():
            self.add_data_from_votes_dict(votes_dict=votes_dict, name=name)

    def get_handler(self, name: str):
        """Get the dataset handler for a given dataset name."""
        return self._handlers[name]

    def set_visible_annotator_rows(self, annotator_rows_visible_names: list[str]):
        """Change the visible annotator rows for all dataset handlers."""
        for handler in self._handlers.values():
            handler.set_visible_annotator_rows(annotator_rows_visible_names)


def split_dataset_by_col(
    handler: DatasetHandler, col: str, selected_vals: list[str] | None = None
) -> MultiDatasetHandler:
    """Split a single dataset by a given column to create multiple datasets."""
    split_votes_dicts = _split_votes_dict(handler.votes_dict, col, selected_vals)
    multi_dataset_handler = MultiDatasetHandler(
        cache=handler.cache, avail_datasets=handler.avail_datasets
    )
    multi_dataset_handler.load_data_from_votes_dicts(split_votes_dicts)
    return multi_dataset_handler

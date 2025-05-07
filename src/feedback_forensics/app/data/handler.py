"""Module with the main data handler for the app.

The data handler allows efficient loading of AnnotatedPairs datasets
and computation of annotation metrics. It provides a unified interface
for loading data from different sources and computing metrics.
"""

import copy
import pandas as pd
from pathlib import Path
from loguru import logger

from feedback_forensics.app.data.loader import add_virtual_annotators, get_votes_dict
from feedback_forensics.app.metrics import (
    get_overall_metrics,
    compute_annotator_metrics,
)


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

    updated_annotator_visible_names = []
    for annotator_name in annotator_visible_names:
        add_annotator = True
        for dataset_name, visible_to_col in visible_to_cols.items():
            if annotator_name not in visible_to_col:
                logger.warning(
                    f"Annotator '{annotator_name}' (visible name) not found in dataset '{dataset_name}'. Skipping this annotator. Available annotators in this dataset: {list(visible_to_col.keys())}"
                )
                add_annotator = False
        if add_annotator:
            updated_annotator_visible_names.append(annotator_name)

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


class ColumnHandler:
    """Class to handle data operations (loading, computing metrics) of a single annotation column dataset."""

    def __init__(self, cache: dict | None = None, avail_datasets: dict | None = None):

        self.cache = cache
        self.avail_datasets = avail_datasets
        self.df = None
        self.annotator_metadata = None
        self.reference_annotator_col = None
        self.shown_annotator_rows = None

        self.data_path = None

    @property
    def votes_dict(self):
        return {
            "df": self.df,
            "annotator_metadata": self.annotator_metadata,
            "reference_annotator_col": self.reference_annotator_col,
            "shown_annotator_rows": self.shown_annotator_rows,
        }

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

        self.load_from_votes_dict(votes_dict)
        logger.info(f"Loaded data from path: {dataset_path}")

    def load_from_votes_dict(self, votes_dict: dict):
        """Load data from a given votes_dict."""
        self.df = votes_dict["df"]
        self.annotator_metadata = votes_dict["annotator_metadata"]
        self.reference_annotator_col = votes_dict["reference_annotator_col"]
        self.shown_annotator_rows = votes_dict["shown_annotator_rows"]

    def set_visible_annotator_rows(self, annotator_rows_keys: list[str]):
        """Change the visible annotator rows for the votes_dict."""
        if not annotator_rows_keys or len(annotator_rows_keys) == 0:
            logger.warning(
                "No annotator rows keys provided. " "Showing all annotator rows."
            )
            return

        self.shown_annotator_rows = annotator_rows_keys

    def compute_overall_metrics(self):
        """Compute the overall metrics for the votes_dict."""
        return get_overall_metrics(self.df, self.reference_annotator_col)

    def compute_annotator_metrics(self):
        """Compute the annotator metrics for the votes_dict."""
        return compute_annotator_metrics(
            votes_df=self.df,
            annotator_metadata=self.annotator_metadata,
            annotator_cols=self.shown_annotator_rows,
            ref_annotator_col=self.reference_annotator_col,
        )


class DatasetHandler:
    """Class to handle data operations (loading, computing metrics) of one or multiple annotation datasets."""

    def __init__(self, cache: dict | None = None, avail_datasets: dict | None = None):
        self.cache = cache
        self.avail_datasets = avail_datasets

        self._col_handlers = {}

    @property
    def col_handlers(self):
        """Get column handlers (each handler is a single dataset)."""
        return self._col_handlers

    @property
    def first_handler(self):
        """Get the first column handler."""
        return list(self._col_handlers.values())[0]

    @property
    def first_handler_name(self):
        """Get the name of the first dataset handler."""
        return list(self._col_handlers.keys())[0]

    @property
    def is_single_dataset(self):
        """Check if the dataset handler is a single dataset."""
        return len(self._col_handlers) == 1

    @property
    def votes_dicts(self):
        """Get the votes_dicts for all dataset handlers."""
        return {
            name: handler.votes_dict for name, handler in self._col_handlers.items()
        }

    @property
    def num_cols(self):
        """Get the number of columns in the dataset."""
        return len(self._col_handlers)

    def add_col_handler(self, name: str, handler: ColumnHandler):
        """Add a column handler."""
        assert (
            name not in self._col_handlers
        ), f"Dataset {name} already loaded or using duplicate name"
        self._col_handlers[name] = handler

    def reset_handlers(self):
        """Reset the dataset handlers."""
        self._col_handlers = {}

    def add_data_from_name(self, name: str):
        """Load data from a given dataset name."""

        handler = ColumnHandler(cache=self.cache, avail_datasets=self.avail_datasets)
        handler.load_data_from_name(name)
        self.add_col_handler(name, handler)

    def load_data_from_names(self, names: list[str]):
        """Load data from a given list of dataset names."""
        for name in names:
            self.add_data_from_name(name)

    def add_data_from_path(self, path: str | Path, name: str):
        """Add data from a given path."""
        handler = ColumnHandler(cache=self.cache, avail_datasets=self.avail_datasets)
        handler.load_data_from_path(path)
        self.add_col_handler(name, handler)

    def load_data_from_paths(self, paths: list[str | Path]):
        """Load data from a given list of paths."""
        for path in paths:
            self.add_data_from_path(path, name=path.split("/")[-1])

    def add_data_from_votes_dict(self, votes_dict: dict, name: str):
        """Add data from a given votes_dict."""
        handler = ColumnHandler(cache=self.cache, avail_datasets=self.avail_datasets)
        handler.load_from_votes_dict(votes_dict)
        self.add_col_handler(name, handler)

    def load_data_from_votes_dicts(self, votes_dicts: dict[str, dict]):
        """Load data from a given list of votes_dicts."""
        for name, votes_dict in votes_dicts.items():
            self.add_data_from_votes_dict(votes_dict=votes_dict, name=name)

    def get_col_handler(self, name: str):
        """Get the column handler for a given dataset name."""
        return self._col_handlers[name]

    def set_annotator_rows(self, annotator_rows_visible_names: list[str]):
        """Change the visible annotator rows for all dataset handlers."""
        if len(annotator_rows_visible_names) == 0:
            logger.warning("No annotator rows provided, no changes to annotator rows.")
            return

        annotator_row_keys = _get_annotator_df_col_names(
            annotator_rows_visible_names, self.votes_dicts
        )
        for handler in self._col_handlers.values():
            handler.set_visible_annotator_rows(annotator_row_keys)

    def split_by_col(self, col: str, selected_vals: list[str] | None = None):
        """Split the dataset by a given column to create multiple datasets."""
        if self.is_single_dataset:
            votes_dict = self.first_handler.votes_dict
            split_votes_dicts = _split_votes_dict(votes_dict, col, selected_vals)
            self.reset_handlers()  # removes previous single dataset handler
            self.load_data_from_votes_dicts(
                split_votes_dicts
            )  # creates new dataset handlers for each split
        else:
            raise ValueError("Cannot split multi-dataset handler.")

    def set_annotator_cols(self, annotator_cols: list[str]):
        """Change the annotator columns for all dataset handlers."""
        if len(annotator_cols) == 0:
            logger.warning(
                "No annotator columns provided, no changes to annotator columns."
            )
            return

        annotator_cols_df_names = _get_annotator_df_col_names(
            annotator_visible_names=annotator_cols,
            votes_dicts=self.votes_dicts,
        )

        if self.is_single_dataset:
            # if only has single dataset, just duplicate
            # the single dataset (handler) with new annotator columns
            dataset_name = self.first_handler_name
            votes_dict = self.first_handler.votes_dict
            dataset_names = [dataset_name] * len(annotator_cols)
            votes_dicts = [votes_dict] * len(annotator_cols)
        else:
            # if is not single dataset, can currently only handle one annotator column
            if len(annotator_cols) > 1:
                logger.warning(
                    "Only one annotator column is supported for multi-dataset handler. "
                    "Ignoring additional annotator columns. "
                    f"Currently {len(self._col_handlers)} datasets are loaded with the following"
                    f"annotators: {annotator_cols}. Only using the first annotator column "
                    f"({annotator_cols[0]})."
                )
                annotator_cols = [annotator_cols[0]]
                annotator_cols_df_names = [annotator_cols_df_names[0]]

            annotator_cols = annotator_cols * len(self._col_handlers)
            annotator_cols_df_names = annotator_cols_df_names * len(self._col_handlers)
            # if has multiple datasets, duplicate each dataset with new annotator column
            dataset_names = list(self._col_handlers.keys())
            votes_dicts = [
                handler.votes_dict for handler in self._col_handlers.values()
            ]

        # create new votes_dicts with the new annotator columns
        votes_dicts = {
            f"{dataset_name}\n({annotator_name.replace('-', ' ')})": {
                "df": votes_dict["df"],
                "annotator_metadata": votes_dict["annotator_metadata"],
                "reference_annotator_col": annotator_col,
                "shown_annotator_rows": votes_dict["shown_annotator_rows"],
            }
            for annotator_col, annotator_name, dataset_name, votes_dict in zip(
                annotator_cols_df_names,
                annotator_cols,
                dataset_names,
                votes_dicts,
            )
        }

        self.reset_handlers()
        self.load_data_from_votes_dicts(votes_dicts)

    def get_overall_metrics(self):
        """Get the overall metrics for all dataset handlers."""
        return {
            name: handler.compute_overall_metrics()
            for name, handler in self._col_handlers.items()
        }

    def get_annotator_metrics(self):
        """Get the annotator metrics for all dataset handlers."""
        return {
            name: handler.compute_annotator_metrics()
            for name, handler in self._col_handlers.items()
        }

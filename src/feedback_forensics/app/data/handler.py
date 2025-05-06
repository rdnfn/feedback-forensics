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


class DatasetHandler:
    """Class to handle data operations (loading, computing metrics) of a single annotation dataset."""

    def __init__(self, cache: dict | None = None, avail_datasets: dict | None = None):

        self.cache = cache
        self.avail_datasets = avail_datasets

        self._votes_dict = None
        self.data_path = None

    @property
    def votes_dict(self):
        return self._votes_dict

    @votes_dict.setter
    def votes_dict(self, value):
        self._votes_dict = value

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

    def compute_metrics(self, data: pd.DataFrame):
        """Compute metrics from a given dataset."""
        pass

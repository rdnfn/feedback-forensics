"""Call backs to be used in the app."""

import pathlib
import copy
import gradio as gr
import pandas as pd

from loguru import logger

from feedback_forensics.data.loader import add_virtual_annotators, get_votes_dict
import feedback_forensics.app.plotting
from feedback_forensics.data.dataset_utils import (
    get_annotators_by_type,
    get_available_models,
)
from feedback_forensics.app.utils import (
    get_csv_columns,
    load_json_file,
)
from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    APP_BASE_URL,
    DEFAULT_ANNOTATOR_VISIBLE_NAME,
    MODEL_IDENTITY_ANNOTATOR_TYPE,
    PRINCIPLE_ANNOTATOR_TYPE,
    PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS,
    PREFIX_MODEL_IDENTITY_ANNOTATORS,
)
from feedback_forensics.data.datasets import (
    get_available_datasets_names,
    get_default_dataset_names,
)

from feedback_forensics.app.url_parser import (
    get_config_from_query_params,
    get_url_with_query_params,
    get_list_member_from_url_string,
    transfer_url_str_to_nonurl_str,
    transfer_url_list_to_nonurl_list,
    parse_list_param,
)
from feedback_forensics.data.handler import (
    DatasetHandler,
    _get_annotator_df_col_names,
)

from feedback_forensics.app.metrics import DEFAULT_METRIC_NAME


def generate(inp: dict, state: dict, out: dict) -> dict:
    """Generate callbacks for loading data and plots."""

    def get_columns_in_dataset(dataset_name, data) -> str:
        """Get the columns in a dataset."""

        dataset_config = data[state["avail_datasets"]][dataset_name]
        results_dir = pathlib.Path(dataset_config.path)
        base_votes_dict = get_votes_dict(results_dir, cache=data[state["cache"]])
        avail_cols = base_votes_dict["available_metadata_keys"]

        if dataset_config.filterable_columns:
            avail_cols = [
                col for col in avail_cols if col in dataset_config.filterable_columns
            ]
        return avail_cols

    def get_avail_col_values(col_name, data):
        """Get the available values for a given column."""

        datasets = data[inp["active_datasets_dropdown"]]

        # Normalize datasets to always be a list for processing
        if not isinstance(datasets, list):
            datasets = [datasets] if datasets is not None else []

        dataset = datasets[0]  # Use first dataset for column values
        dataset_config = data[state["avail_datasets"]][dataset]
        results_dir = pathlib.Path(dataset_config.path)
        cache = data[state["cache"]]
        votes_dict = get_votes_dict(results_dir, cache=cache)
        votes_df = votes_dict["df"]
        value_counts = votes_df[col_name].value_counts()
        avail_values = [
            (count, f"{val} ({count})", str(val)) for val, count in value_counts.items()
        ]
        # sort by count descending
        avail_values = sorted(avail_values, key=lambda x: x[0], reverse=True)
        # remove count from avail_values
        avail_values = [(val[1], val[2]) for val in avail_values]
        return avail_values

    return {
        "get_columns_in_dataset": get_columns_in_dataset,
        "get_avail_col_values": get_avail_col_values,
    }

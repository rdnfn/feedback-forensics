"""Call backs to be used in the app."""

import pathlib

from feedback_forensics.data.loader import get_votes_dict


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

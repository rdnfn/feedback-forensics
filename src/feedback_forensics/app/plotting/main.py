import pandas as pd
from loguru import logger
import gradio as gr
import matplotlib as mpl
import matplotlib.colors
import numpy as np

import feedback_forensics.app.metrics
from feedback_forensics.app.plotting.table import create_fig_with_tables
from feedback_forensics.app.plotting.table import get_table_contents_from_metrics


def generate_dataframe(
    votes_dicts: dict[str, dict],
):

    # compute metrics for each dataset
    overall_metrics = {}
    metrics = {}
    for dataset_name, votes_dict in votes_dicts.items():
        overall_metrics[dataset_name] = (
            feedback_forensics.app.metrics.get_overall_metrics(
                votes_dict["df"],
                ref_annotator_col=votes_dict["reference_annotator_col"],
            )
        )
        metrics[dataset_name] = feedback_forensics.app.metrics.compute_metrics(
            votes_dict
        )

    overall_metrics_df = pd.DataFrame(overall_metrics)

    table_df = get_table_df(metrics)

    return table_df


def get_table_df(
    metrics: dict[str, dict],
    sort_by: str | None = None,
    sort_ascending: bool = False,
    color_scale: str = "berlin",
    neutral_value: float = 0.0,
) -> pd.DataFrame:

    initial_dataset = list(metrics.keys())[0]
    metric = "strength"
    metric_columns = {
        dataset_name: list(dataset_dict["metrics"][metric]["by_annotator"].values())
        for dataset_name, dataset_dict in metrics.items()
    }
    annotator_keys = list(
        metrics[initial_dataset]["metrics"][metric]["by_annotator"].keys()
    )

    # sanity check
    for dataset_name, dataset_dict in metrics.items():
        assert (
            list(dataset_dict["metrics"][metric]["by_annotator"].keys())
            == annotator_keys
        )

    shown_df = pd.DataFrame(
        {
            "annotator": annotator_keys,
            **metric_columns,
        }
    )
    # get max numerical value in the dataframe (ignoring non-numeric values)
    max_value = shown_df.select_dtypes(include=[np.number]).max().max()
    # get min numerical value in the dataframe (ignoring non-numeric values)
    min_value = shown_df.select_dtypes(include=[np.number]).min().min()

    # sort by
    if sort_by is None:
        sort_by = list(metric_columns.keys())[0]

    shown_df = shown_df.sort_values(by=sort_by, ascending=sort_ascending)

    shown_values = shown_df.to_numpy()

    def get_styling(values):
        print(f"values: {values}")
        cmap = mpl.colormaps[color_scale]
        display_values = []
        for i, row in enumerate(values):
            display_row = []
            for j, col in enumerate(row):
                if isinstance(col, float):
                    val = col * 100
                    # Use the berlin colormap for the background
                    # Normalize the value relative to neutral_value
                    # If val > neutral_value, scale from neutral to 100
                    # If val < neutral_value, scale from 0 to neutral
                    color_hex = matplotlib.colors.rgb2hex(
                        cmap(
                            0.5
                            + 0.5 * (col - neutral_value) / (max_value - neutral_value)
                            if col > neutral_value
                            else 0.5 * (col - min_value) / (neutral_value - min_value)
                        )
                    )
                    display_row.append(f"background-color: {color_hex}; color: white;")
                else:
                    display_row.append("")
            display_values.append(display_row)
        return display_values

    def get_display_value(values):
        print(f"values: {values}")
        display_values = []
        for i, row in enumerate(values):
            display_row = []
            for j, col in enumerate(row):
                if isinstance(col, float):
                    display_row.append(f"{col:.2f}")
                else:
                    display_row.append(col)
            display_values.append(display_row)
        return display_values

    styling = get_styling(shown_values)
    display_value = get_display_value(shown_values)

    value = {
        "data": shown_values,
        "headers": ["Annotator"] + list(metric_columns.keys()),
        "metadata": {
            "styling": styling,
            "display_value": display_value,
        },
    }

    return gr.Dataframe(
        value,
        datatype=["str"] + ["number"] * len(metric_columns),
        # show_search="filter", TODO: reactivate once sorting issue by Gradio is fixed
        interactive=False,
    )

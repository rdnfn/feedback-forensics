import pandas as pd
from loguru import logger
import gradio as gr
import matplotlib as mpl
import matplotlib.colors
import numpy as np

import feedback_forensics.app.metrics
from feedback_forensics.app.plotting.table import create_fig_with_tables
from feedback_forensics.app.plotting.table import get_table_contents_from_metrics


def generate_dataframes(
    annotator_metrics: dict[str, dict],
    overall_metrics: dict[str, dict],
    metric_name: str = "strength",
    sort_by: str = None,
    sort_ascending: bool = False,
):

    overall_df = pd.DataFrame(overall_metrics)
    overall_df.insert(0, "Metric", overall_df.index)  # insert metric name as col
    overall_metrics_df = gr.Dataframe(overall_df)

    annotator_table_df = get_annotator_table_df(
        annotator_metrics,
        metric_name=metric_name,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
    )

    return {
        "overall_metrics": overall_metrics_df,
        "annotator": annotator_table_df,
    }


def get_annotator_table_df(
    annotator_metrics: dict[str, dict],
    metric_name: str = "strength",
    sort_by: str = None,
    sort_ascending: bool = False,
    color_scale: str = "berlin",
    neutral_value: float = 0.0,
) -> pd.DataFrame:

    initial_dataset = list(annotator_metrics.keys())[0]
    metric = metric_name  # Use the provided metric_name instead of hardcoded "strength"
    metric_columns = {
        dataset_name: list(dataset_dict["metrics"][metric]["by_annotator"].values())
        for dataset_name, dataset_dict in annotator_metrics.items()
    }
    annotator_keys = list(
        annotator_metrics[initial_dataset]["metrics"][metric]["by_annotator"].keys()
    )

    # sanity check
    for dataset_name, dataset_dict in annotator_metrics.items():
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
    cmap = mpl.colormaps[color_scale]

    # sort by
    if sort_by is None:
        sort_by = list(metric_columns.keys())[0]

    shown_df = shown_df.sort_values(by=sort_by, ascending=sort_ascending)

    shown_values = shown_df.to_numpy()

    def get_styling(values):
        display_values = []
        for row in values:
            display_row = []
            for col in row:
                if isinstance(col, float):
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
        display_values = []
        for row in values:
            display_row = []
            for col in row:
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

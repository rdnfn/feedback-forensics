import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib as mpl
import plotly.subplots
import plotly.graph_objects as go
import plotly.colors


# table constants
AVAIL_METRICS = {
    "strength": {
        "name": "Principle strength (relevance-weighted Cohen's kappa)",
        "color_scale": "berlin",
        "neutral_value": 0.0,
    },
    "acc": {
        "name": "Accuracy",
        "color_scale": "berlin",
        "neutral_value": 0.0,
    },
    "relevance": {
        "name": "Relevance",
        "color_scale": "berlin",
        "neutral_value": 0.0,
    },
    "cohens_kappa": {
        "name": "Cohen's kappa",
        "color_scale": "berlin",
        "neutral_value": 0.0,
    },
}

GREY_10 = "#5f5f5f"
GREY_20 = "#3a3a3a"

PAPER_BG_COLOR = "#0f0f11"

FONT_FAMILY = "IBM Plex Sans"

TABLE_TITLE_FONT_SIZE = 20
TABLE_TITLE_FONT_COLOR = "#98989a"
TABLE_TITLE_FONT_FAMILY = "IBM Plex Sans"


HEADER_BG_COLOR = "#52525b"
HEADER_FONT_COLOR = "white"
HEADER_FONT_SIZE = 12
HEADER_FONT_FAMILY = None  # "open-sans"

EVEN_ROW_BG_COLOR = "#18181b"
ODD_ROW_BG_COLOR = "#27272a"
CELL_FONT_COLOR = "white"
CELL_FONT_SIZE = 12
CELL_FONT_FAMILY = None  # "open-sans"

TABLE_HEADING_HEIGHT = 30
TABLE_ROW_HEIGHT = 22
TABLE_ALIGN = ["left", "right"]
TABLE_HEADER_ALIGN = ["left", "right"]

TABLE_HEADER_CONFIG = dict(
    fill_color=HEADER_BG_COLOR,
    font=dict(
        color=HEADER_FONT_COLOR,
        size=HEADER_FONT_SIZE,
        family=HEADER_FONT_FAMILY,
    ),
    line_color=HEADER_BG_COLOR,
    align=TABLE_HEADER_ALIGN,
    height=TABLE_HEADING_HEIGHT,
)

ANNOTATION_FONT_FAMILY = None  # "open-sans"

SPACING_BETWEEN_TABLES = 80


# function that transforms a matplotlib colormap to a Plotly colorscale
# from https://notebook.community/empet/Plotly-plots/Plotly-asymmetric-colorscales
def colormap_to_colorscale(cmap):
    return [[k * 0.1, matplotlib.colors.rgb2hex(cmap(k * 0.1))] for k in range(11)]


BERLIN_CELL_COLOR_SCALE = colormap_to_colorscale(mpl.colormaps["berlin"])


### TABLE UTILITIES ###


def get_table_height(table_data: pd.DataFrame):
    if isinstance(table_data, dict):
        num_rows = len(table_data[list(table_data.keys())[0]]["data"].index)
    else:
        num_rows = len(table_data.index)
    table_height = TABLE_HEADING_HEIGHT + TABLE_ROW_HEIGHT * num_rows
    return table_height * 1.3


def get_alternating_cell_colors(num_cols, num_rows):
    colors = []
    for i in range(num_rows * num_cols):
        if i % 2 == 0:
            colors.append(EVEN_ROW_BG_COLOR)
        else:
            colors.append(ODD_ROW_BG_COLOR)
    return colors


def get_variable_cell_colors(
    color_scale,
    values,
    max_value,
    neutral_value,
    min_value,
    flip_orientation: bool = True,
):
    # normalize values between min_value and neutral_value to be between 0 and 0.5
    # normalize values between neutral_value and max_value to be between 0.5 and 1
    # then use the color_scale to get the colors
    values = np.clip(values, min_value, max_value)

    # deal with nan
    values = np.where(np.isnan(values), neutral_value, values)
    if neutral_value < max_value:
        above_neutral_norm_values = 0.5 + 0.5 * (values - neutral_value) / (
            max_value - neutral_value
        )
    else:
        above_neutral_norm_values = np.ones_like(values) * 0.5
    if neutral_value > min_value:
        below_neutral_norm_values = (
            0.5 * (values - min_value) / (neutral_value - min_value)
        )
    else:
        below_neutral_norm_values = np.ones_like(values) * 0.5
    normalized_values = np.where(
        values > neutral_value,
        above_neutral_norm_values,
        below_neutral_norm_values,
    )
    if flip_orientation:
        normalized_values = 1 - normalized_values

    return plotly.colors.sample_colorscale(color_scale, normalized_values)


def get_cell_colors(
    table_data: pd.DataFrame,
    color_scale: str | None,
    neutral_value: float,
):
    """Get colors for each cell.

    Returns a long list of colors, going by columns:
    e.g. [col1_row1, col1_row2, col1_row3, col2_row1, col2_row2, col2_row3, ...]
    """

    if color_scale is None:
        cell_colors = [
            get_alternating_cell_colors(
                len(table_data.columns) + 1, len(table_data.index)
            )
        ]
    else:
        if color_scale == "berlin":
            color_scale = BERLIN_CELL_COLOR_SCALE
        else:
            raise ValueError(f"Non supported color scale: {color_scale}")

        max_value = table_data.max().max()
        min_value = table_data.min().min()

        cell_colors = [get_alternating_cell_colors(1, len(table_data.index))] + [
            get_variable_cell_colors(
                color_scale,
                table_data[col],
                max_value=max_value,
                neutral_value=neutral_value,
                min_value=min_value,
            )
            for col in table_data.columns
        ]

    return cell_colors


def get_table_values(table_data: pd.DataFrame) -> list[list[str]]:
    return [table_data.index] + [table_data[col] for col in table_data.columns]


def get_values_as_strings(values: list[list[float | str]]) -> list[list[str]]:
    return [
        [
            (
                f"{value:.2f}"
                if isinstance(value, float) or isinstance(value, int)
                else str(value)
            )
            for value in row
        ]
        for row in values
    ]


def get_updatemenus_and_annotation(
    table_title: str,
    table_data: pd.DataFrame | dict,
    table_index: int,
    relative_table_heights: list[float],
    spacing: float,
    index_col_heading: str,
) -> tuple[list[dict], dict]:
    """Get the updatemenus and info annotation for a table.
    Returns None for both if table_data is not a dict."""

    # y position of updatemenus from the bottom
    y_updatemenus = sum(relative_table_heights[table_index:]) + spacing * (
        len(relative_table_heights[table_index:]) - 0.9
    )
    y_table_title = y_updatemenus

    table_title_annotation = dict(
        x=0.0,
        y=y_table_title,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        text=table_title,
        showarrow=False,
        font=dict(
            color=TABLE_TITLE_FONT_COLOR,
            size=TABLE_TITLE_FONT_SIZE,
            family=TABLE_TITLE_FONT_FAMILY,
        ),
    )

    # if not multiple views, return None for updatemenus and table title annotation
    if not isinstance(table_data, dict):
        return None, [table_title_annotation]

    buttons = []
    for metric_name in table_data.keys():
        values = table_data[metric_name]["values"]
        colors = table_data[metric_name]["colors"]
        header_values = [index_col_heading] + list(
            table_data[metric_name]["data"].columns
        )
        buttons.append(
            dict(
                label=metric_name,
                method="restyle",
                args=[
                    {
                        "header": {
                            "values": header_values,
                            "fill": dict(color=HEADER_BG_COLOR),
                            "font": dict(
                                color=HEADER_FONT_COLOR,
                                size=HEADER_FONT_SIZE,
                                family=HEADER_FONT_FAMILY,
                            ),
                            "line": dict(color=HEADER_BG_COLOR),
                            "align": TABLE_HEADER_ALIGN,
                        },
                        "cells": {
                            "values": values,
                            "fill": dict(color=colors),
                            "line": dict(color=colors),
                            "font": dict(
                                color=CELL_FONT_COLOR,
                                size=CELL_FONT_SIZE,
                                family=CELL_FONT_FAMILY,
                            ),
                            "align": TABLE_ALIGN,
                        },
                    },
                    [table_index],
                ],
            )
        )

    updatemenus = [
        dict(
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1,
            y=y_updatemenus,
            xanchor="right",
            yanchor="bottom",
            font=dict(color=TABLE_TITLE_FONT_COLOR, family=ANNOTATION_FONT_FAMILY),
            bordercolor=HEADER_BG_COLOR,
        )
    ]

    annotations = [table_title_annotation]

    return updatemenus, annotations


def get_table_trace(
    table_title: str,
    table_data: pd.DataFrame | dict,
    color_scale: str | None,
    index_col_heading: str = "",
    neutral_value: float = 0.0,
    table_index: int = 0,
    relative_table_heights: list[float] = None,
    spacing: float = 0.0,
):
    """Get the table trace and its associated menu and annotation if applicable."""
    if isinstance(table_data, dict):
        available_metrics = list(table_data.keys())
        initial_metric_name = available_metrics[0]
        initial_metric_dict = table_data[initial_metric_name]
        columns = list(initial_metric_dict["data"].columns)
        initial_values = initial_metric_dict["values"]
        initial_cell_colors = initial_metric_dict["colors"]
    else:
        columns = list(table_data.columns)
        initial_values = get_values_as_strings(get_table_values(table_data))
        initial_cell_colors = get_cell_colors(table_data, color_scale, neutral_value)

    num_cols = len(columns)
    widths = [(1 / 2) * num_cols] + [1] * num_cols

    table = go.Table(
        columnwidth=widths,
        header=dict(
            values=[index_col_heading] + columns,
            **TABLE_HEADER_CONFIG,
        ),
        cells=dict(
            values=initial_values,
            fill_color=initial_cell_colors,
            line_color=initial_cell_colors,
            font=dict(
                color=CELL_FONT_COLOR,
                size=CELL_FONT_SIZE,
                family=CELL_FONT_FAMILY,
            ),
            align=TABLE_ALIGN,
            height=TABLE_ROW_HEIGHT,
        ),
    )

    updatemenus, annotations = get_updatemenus_and_annotation(
        table_title=table_title,
        table_data=table_data,
        table_index=table_index,
        relative_table_heights=relative_table_heights,
        spacing=spacing,
        index_col_heading=index_col_heading,
    )

    return table, updatemenus, annotations


def add_combined_metric_to_table_contents(
    perf_data: pd.DataFrame,
    acc_data: pd.DataFrame,
    rel_data: pd.DataFrame,
    colors: list[str],
    sort_by: str = None,
) -> dict:
    """Adds a metric that combines perf, acc and relevance into a single view.

    Returns the same dict with the combined metric added.
    """

    original_dataset_cols = [str(col) for col in perf_data.columns]

    # copy dataframes
    perf_data = perf_data.copy()
    acc_data = acc_data.copy()
    rel_data = rel_data.copy()

    # merge dataset based on index
    combined_data = pd.merge(
        perf_data,
        acc_data,
        left_index=True,
        right_index=True,
        suffixes=("_perf", "_acc"),
    )
    combined_data = pd.merge(
        combined_data,
        rel_data.add_suffix("_rel"),
        left_index=True,
        right_index=True,
    )
    for col in original_dataset_cols:
        combined_data[col] = combined_data.apply(
            lambda row: (
                f"{row[col + '_perf']:.2f} ({row[col + '_acc']:.2f}|{row[col + '_rel']:.2f})"
                if isinstance(row[col + "_perf"], float)
                else row[col + "perf"]
            ),
            axis=1,
        )
    if sort_by == "max diff":
        subset_perf_cols = [
            col
            for col in combined_data.columns
            if "perf" in col and "max diff" not in col
        ]
        combined_data["max diff"] = abs(
            combined_data[subset_perf_cols].max(axis=1)
            - combined_data[subset_perf_cols].min(axis=1)
        )

    # limit cols to original dataset cols
    combined_data = combined_data[original_dataset_cols]

    metric_dict = {}

    metric_dict["data"] = combined_data
    metric_dict["values"] = get_values_as_strings(get_table_values(combined_data))
    metric_dict["colors"] = colors

    return metric_dict


def get_table_contents_from_metrics(metrics: dict[str, dict]) -> dict:
    """Compute the contents of the table based on metrics dict.

    Args:
        metrics: A dictionary where keys are dataset names and values are
                metric dictionaries containing metrics for that dataset.
                Structure: {dataset_name: {metrics: {metric_name: {by_principle: ...}}}}

    Returns:
        A dictionary of table views, where each view contains data, values, and colors
        for different sorting and metric combinations. These views are used as
        alternative table versions controlled by updatemenus.
    """

    principles_metrics_dfs = {}

    for metric_name, metric_info in list(AVAIL_METRICS.items()) + [
        (
            "strength (acc|rel)",
            None,
        )
    ]:
        # get metrics data
        if not metric_name == "strength (acc|rel)":
            # Create DataFrame with principles as rows and datasets as columns
            data = pd.DataFrame(
                {
                    dataset_name: metrics_dict["metrics"][metric_name]["by_principle"]
                    for dataset_name, metrics_dict in metrics.items()
                }
            )

        dataset_names = list(metrics.keys())
        if len(metrics) > 1:
            # Insert virtual max diff column first to make it the default view.
            dataset_names.insert(0, "max diff")

        for dataset_name in dataset_names:
            data = data.copy()

            view_name = f"{metric_name}  ({dataset_name} ↓)"

            if metric_name == "strength (acc|rel)":
                principles_metrics_dfs[view_name] = (
                    add_combined_metric_to_table_contents(
                        principles_metrics_dfs[f"strength  ({dataset_name} ↓)"]["data"],
                        principles_metrics_dfs[f"acc  ({dataset_name} ↓)"]["data"],
                        principles_metrics_dfs[f"relevance  ({dataset_name} ↓)"][
                            "data"
                        ],
                        principles_metrics_dfs[f"strength  ({dataset_name} ↓)"][
                            "colors"
                        ],
                        sort_by=dataset_name,
                    )
                )
                continue

            # process data to be viewed in table
            if dataset_name == "max diff":
                data["max diff"] = abs(data.max(axis=1) - data.min(axis=1))
                data = data.sort_values(by="max diff", ascending=False)
                # data = data.drop(columns=["max diff"])
            else:
                data = data.sort_values(by=dataset_name, ascending=False)
            data.index = data.index.str.replace(
                "Select the response that", ""
            ).str.strip(" .")
            values = get_values_as_strings(get_table_values(data))
            colors = get_cell_colors(
                data, metric_info["color_scale"], metric_info["neutral_value"]
            )

            # store data for later use
            principles_metrics_dfs[view_name] = {}
            principles_metrics_dfs[view_name]["data"] = data.copy()
            principles_metrics_dfs[view_name]["values"] = values
            principles_metrics_dfs[view_name]["colors"] = colors

    return principles_metrics_dfs


def create_fig_with_tables(
    table_titles: list[str],
    table_dfs: list[pd.DataFrame | dict],
    color_scales: list[str] | None = None,
    index_col_headings: list[str] | None = None,
    neutral_values: list[float] | None = None,
):
    """Create a figure with multiple tables.

    Args:
        table_dfs: List of tables to be displayed. The list can contain dataframes or dictionaries.
            The dictionaries are used to create tables with updatemenus (multiple views of the same table).
            Each dictionary must have the values and colors keys.
        color_scales: List of color scales to be used for the tables.
        index_col_headings: List of headings for the index column of each table.
        neutral_values: List of neutral values for the tables.

    Returns:
        A plotly figure with multiple tables.
    """
    if index_col_headings is None:
        index_col_headings = [None] * len(table_dfs)
    if neutral_values is None:
        neutral_values = [None] * len(table_dfs)
    if color_scales is None:
        color_scales = [None] * len(table_dfs)

    table_heights = [get_table_height(table_df) for table_df in table_dfs]
    spacing = SPACING_BETWEEN_TABLES / sum(table_heights)
    total_height = sum(table_heights) + SPACING_BETWEEN_TABLES * (
        len(table_heights) - 1
    )
    relative_table_heights = [height / total_height for height in table_heights]
    fig = plotly.subplots.make_subplots(
        rows=len(table_dfs),
        cols=1,
        specs=[[{"type": "domain"}]] * len(table_dfs),
        row_heights=relative_table_heights,
        vertical_spacing=spacing,
    )

    # Initialize empty lists for menus and annotations
    all_menus = []
    all_annotations = []

    # add table traces to the figure
    tables = [
        get_table_trace(
            table_title,
            table_df,
            color_scale,
            index_col_heading,
            neutral_value,
            table_index,
            relative_table_heights,
            spacing,
        )
        for table_title, table_df, color_scale, index_col_heading, neutral_value, table_index in zip(
            table_titles,
            table_dfs,
            color_scales,
            index_col_headings,
            neutral_values,
            range(len(table_dfs)),
        )
    ]
    for i, (table, menu, annotations) in enumerate(tables):
        fig.add_trace(table, row=i + 1, col=1)
        if menu is not None:
            all_menus.extend(menu)
        if annotations is not None:
            all_annotations.extend(annotations)

    # Add all menus and annotations to the figure
    if all_menus:
        fig.update_layout(updatemenus=all_menus)
    if all_annotations:
        fig.update_layout(annotations=all_annotations)

    fig.update_layout(
        paper_bgcolor=PAPER_BG_COLOR,
        height=sum(table_heights),
        margin=dict(l=20, r=20, t=75, b=20),
    )

    return fig

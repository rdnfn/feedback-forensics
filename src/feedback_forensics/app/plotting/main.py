"""Plotting functions for the Inverse CAI app."""

import gradio as gr
import plotly.graph_objects as go
import pandas as pd
from loguru import logger

import feedback_forensics.app.metrics
from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    PRINCIPLE_SHORT_LENGTH,
    END_RECONSTRUCTION_PLOT_X,
    PRINCIPLE_END_X,
    METRICS_START_X,
    MENU_X,
    FONT_FAMILY,
    FONT_COLOR,
    PAPER_BACKGROUND_COLOR,
    PLOT_BACKGROUND_COLOR,
    PLOTLY_MODEBAR_POSSIBLE_VALUES,
    SPACE_PER_NUM_COL,
    get_fig_proportions_y,
)
from feedback_forensics.app.plotting.single import _plot_examples, _plot_aggregated
from feedback_forensics.app.plotting.multiple import _plot_multiple_values
from feedback_forensics.app.plotting.metrics_table import (
    _generate_metrics_table_annotations,
)

# based on official plotly example
# https://plotly.com/python/horizontal-bar-charts/


def generate_plot(
    votes_df: pd.DataFrame,
    unfiltered_df: pd.DataFrame,
    show_examples: bool = False,
    shown_metric_names: list[str] | None = None,
    default_ordering_metric: str = "perf",
    sort_examples_by_agreement: bool = True,
    plot_col_name: str = NONE_SELECTED_VALUE,
    plot_col_values: list = None,
) -> go.Figure:

    if plot_col_name is not None and plot_col_name != NONE_SELECTED_VALUE:
        # plot per principle and per multiple column values
        gr.Warning(
            "Plots of the selected configuration (based on column values) are currently experimental, results may vary."
        )
        return _plot_multiple_values(
            votes_df=votes_df,
            plot_col_name=plot_col_name,
            plot_col_values=plot_col_values,
            shown_metric_names=shown_metric_names,
        )
    else:
        # plot only per principle (for entire dataset)
        return _generate_hbar_chart(
            votes_df=votes_df,
            unfiltered_df=unfiltered_df,
            show_examples=show_examples,
            shown_metric_names=shown_metric_names,
            default_ordering_metric=default_ordering_metric,
            sort_examples_by_agreement=sort_examples_by_agreement,
            plot_col_name=plot_col_name,
            plot_col_values=plot_col_values,
        )


def _generate_hbar_chart(
    votes_df: pd.DataFrame,
    unfiltered_df: pd.DataFrame,
    show_examples: bool = False,
    shown_metric_names: list[str] | None = None,
    default_ordering_metric: str = "perf",
    sort_examples_by_agreement: bool = True,
    plot_col_name: str = NONE_SELECTED_VALUE,
    plot_col_values: list = None,
) -> go.Figure:

    if shown_metric_names is None:
        shown_metric_names = [
            "perf",
            "perf_base",
            "perf_diff",
            "acc",
            "relevance",
        ]

    logger.debug("Computing metrics...")
    full_metrics: dict = feedback_forensics.app.metrics.compute_metrics(unfiltered_df)
    metrics: dict = feedback_forensics.app.metrics.compute_metrics(
        votes_df, baseline_metrics=full_metrics
    )
    principles = metrics["principles"]
    logger.debug("Metrics computed.")

    overall_metrics = feedback_forensics.app.metrics.get_overall_metrics(votes_df)

    proportions_y: dict = get_fig_proportions_y(len(principles), len(overall_metrics))

    logger.debug(f"Proportions y: {proportions_y}")

    TABLE_PROPORTIONS_Y = [
        proportions_y["principle_table"]["relative"][
            "table_bottom_y"
        ],  # order is important, lower y value first (x goes up from bottom of figure)
        proportions_y["principle_table"]["relative"]["table_top_y"],
    ]

    HEADING_HEIGHT_Y = proportions_y["principle_table"]["relative"]["heading_y"]
    HEADING_ANCHOR_Y = "middle"
    MENU_Y = HEADING_HEIGHT_Y
    # SPACE_ALL_NUM_COL = FIG_PROPORTIONS_X[0] - METRICS_START_X - 0.01
    # SPACE_PER_NUM_COL = SPACE_ALL_NUM_COL / len(shown_metric_names)

    SPACE_ALL_NUM_COL = SPACE_PER_NUM_COL * len(shown_metric_names)

    FIG_PROPORTIONS_X = [
        PRINCIPLE_END_X + SPACE_ALL_NUM_COL,
        END_RECONSTRUCTION_PLOT_X,
    ]

    fig = go.Figure()

    if sort_examples_by_agreement:
        votes_df = votes_df.sort_values(by=["principle", "vote"], axis=0)

    # bar plots for each principle
    if plot_col_name == NONE_SELECTED_VALUE:

        # add plots for single set of values
        if show_examples:
            _plot_examples(fig, votes_df, principles)
        else:
            _plot_aggregated(fig, principles, metrics)

    else:
        logger.warning(
            f"Plotting multiple values for '{plot_col_name}' should not happen here."
        )

    # set up general layout configurations
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=FIG_PROPORTIONS_X,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=TABLE_PROPORTIONS_Y,
        ),
        barmode="stack",
        paper_bgcolor=PAPER_BACKGROUND_COLOR,
        plot_bgcolor=PLOT_BACKGROUND_COLOR,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=proportions_y["principle_table"]["absolute"]["total_height"],
        font_family=FONT_FAMILY,
        template="plotly_dark",
    )

    annotations = []
    headings_added = []

    metrics_annotations = _generate_metrics_table_annotations(
        overall_metrics,
        PRINCIPLE_END_X,
        row_height=proportions_y["principle_table"]["relative"]["row_height"],
        start_y=proportions_y["principle_table"]["relative"]["metrics_table_top_y"],
        end_y=proportions_y["principle_table"]["relative"]["metrics_table_bottom_y"],
    )

    annotations.extend(metrics_annotations)

    # adding principle labels to y-axis
    for principle in principles:
        principle_short = principle.replace("Select the response that", "").strip(" .")
        principle_short = (
            principle_short[:PRINCIPLE_SHORT_LENGTH] + "..."
            if len(principle_short) > PRINCIPLE_SHORT_LENGTH
            else principle_short
        )

        # principle
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=PRINCIPLE_END_X,
                y=principle,
                xanchor="right",
                text=principle_short,
                font=dict(size=12, color=FONT_COLOR),
                showarrow=False,
                align="right",
                hovertext=f"Tested principle: {principle}",
            )
        )

        # add metric values in own columns
        for (
            start,
            value,
            label,
            hovertext,
        ) in feedback_forensics.app.metrics.get_metric_cols_by_principle(
            principle,
            metrics,
            shown_metric_names,
            METRICS_START_X,
            FIG_PROPORTIONS_X[0] - METRICS_START_X - 0.01,
        ):
            annotations.append(
                dict(
                    xref="paper",
                    yref="y",
                    x=start,
                    y=principle,
                    xanchor="right",
                    text=f"{value:.2f}",
                    font=dict(size=14, color=FONT_COLOR),
                    showarrow=False,
                    align="right",
                )
            )
            # value
            if label not in headings_added:
                headings_added.append(label)
                annotations.append(
                    dict(
                        xref="paper",
                        yref="paper",
                        x=start,
                        y=HEADING_HEIGHT_Y,
                        xanchor="right",
                        yanchor=HEADING_ANCHOR_Y,
                        text=f"<i>{label}</i>",
                        font=dict(size=14, color=FONT_COLOR),
                        showarrow=False,
                        align="right",
                        hovertext=hovertext,
                    )
                )

    # Add heading for principle and vote count columns
    # the first one is the principle heading is aligned right
    # the Preference reconstruction results heading is aligned left (as its the last one)
    for start, label, hovertext, align in [
        [PRINCIPLE_END_X, "Annotations prefer a response that ...", None, "right"],
        [
            FIG_PROPORTIONS_X[0],
            f"Preference reconstruction results ({metrics['num_pairs']} comparisons)",
            "Reconstruction results with LLM-as-a-Judge following the hypothesis.",
            "left",
        ],
    ]:
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=start,
                y=HEADING_HEIGHT_Y,
                xanchor=align,
                yanchor=HEADING_ANCHOR_Y,
                text=f"<i>{label}</i>",
                font=dict(size=14, color=FONT_COLOR),
                showarrow=False,
                align=align,
                hovertext=hovertext,
            )
        )

    # sort by agreement
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=metrics["metrics"][default_ordering_metric]["principle_order"],
    )

    # add sorting menu
    update_method = "relayout"  # "update"  # or "relayout"
    options = feedback_forensics.app.metrics.get_ordering_options(
        metrics, shown_metric_names, initial=default_ordering_metric
    )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    {
                        "label": label,
                        "method": update_method,
                        "args": [
                            {
                                "yaxis.categoryorder": "array",
                                "yaxis.categoryarray": option,
                            }
                        ],
                    }
                    for label, option in options
                ],
                x=MENU_X,
                xanchor="left",
                y=MENU_Y,
                yanchor=HEADING_ANCHOR_Y,
                font=dict(size=10, color=FONT_COLOR),
                bgcolor="#3f3f46",
                showactive=False,
                # pad={"r": 0, "t": 0, "b": 0, "l": 0},
            )
        ],
    )

    # annotations.append(
    #     dict(
    #         xref="paper",
    #         yref="paper",
    #         x=MENU_X,
    #         y=MENU_Y,
    #         xanchor="right",
    #         yanchor="middle",
    #         text="Sort by:",
    #         font=dict(size=11, color=FONT_COLOR),
    #         showarrow=False,
    #         align="left",
    #     )
    # )

    fig.update_layout(
        annotations=annotations,
    )

    # remove/hide modebar
    fig.update_layout(
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="rgba(0,0,0,0)",
            activecolor="rgba(0,0,0,0)",
            remove=PLOTLY_MODEBAR_POSSIBLE_VALUES,
        )
    )

    gr.Info("Plotting complete, uploading to interface.", duration=3)

    return fig

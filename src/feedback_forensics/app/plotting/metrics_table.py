import plotly.graph_objects as go

from feedback_forensics.app.constants import FONT_COLOR, SPACE_PER_NUM_COL


def _generate_metrics_table_annotations(
    metrics: dict,
    name_x_right: float,
    row_height: float,
    start_y: float,
    end_y: float,
    heading_texts: list[str] = ["Metric", "Value"],
) -> go.Figure:

    annotations = []

    heading_x_positions = [name_x_right, name_x_right + SPACE_PER_NUM_COL]

    # add heading for the table
    for heading_text, heading_x_position in zip(heading_texts, heading_x_positions):
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=heading_x_position,
                y=start_y,
                xanchor="right",
                text=heading_text,
                font=dict(size=14, color=FONT_COLOR, weight="bold"),
                showarrow=False,
            )
        )

    # add metrics to the table
    for i, (metric, value) in enumerate(metrics.items()):

        # add metric name
        row_y = start_y - row_height * (i + 1.5)
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=name_x_right,
                y=row_y,
                xanchor="right",
                text=metric,
                font=dict(size=12, color=FONT_COLOR),
                showarrow=False,
                align="right",
            )
        )

        # add metric value
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=name_x_right + SPACE_PER_NUM_COL,
                y=row_y,
                xanchor="right",
                text=f"{value:.2f}",
                font=dict(size=14, color=FONT_COLOR),
                showarrow=False,
                align="left",
            )
        )

    return annotations

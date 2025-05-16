"""Plotting functions for the paper"""

from typing import Callable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scienceplots


plt.style.use(["science", "nature"])

TITLE_FONT_SIZE = 14
TITLE_FONT_WEIGHT = "bold"
# Use the same colors as in the main plotting module
POSITIVE_COLOR = "#9eb0ff"  # Light blue
NEGATIVE_COLOR = "#ffadad"  # Light red
ALTERNATE_ROW_COLOR = "#f5f5f5"  # Light grey for alternating rows


def get_top_and_bottom_annotators(
    annotator_metrics: dict,
    top_n: int = 5,
    bottom_n: int = 5,
    format_values: bool = True,
):
    """Extract top and bottom annotators from metrics.

    Args:
        annotator_metrics: Dictionary mapping annotator names to metric values
        top_n: Number of top annotators to show
        bottom_n: Number of bottom annotators to show
        format_values: If True, format values as strings with 3 decimal places

    Returns:
        tuple: (top_n_annotators, bottom_n_annotators, max_abs_value)
    """
    metric_series = pd.Series(annotator_metrics)

    if format_values:
        top_n_annotators = [
            [annotator, f"{float(value):.3f}"]
            for annotator, value in metric_series.sort_values(ascending=False)
            .head(top_n)
            .items()
        ]
        bottom_n_annotators = [
            [annotator, f"{float(value):.3f}"]
            for annotator, value in metric_series.sort_values(ascending=True)
            .head(bottom_n)
            .items()
        ]
    else:
        top_n_annotators = [
            [annotator, float(value)]
            for annotator, value in metric_series.sort_values(ascending=False)
            .head(top_n)
            .items()
        ]
        bottom_n_annotators = [
            [annotator, float(value)]
            for annotator, value in metric_series.sort_values(ascending=True)
            .head(bottom_n)
            .items()
        ]

    # Find max absolute value for normalization
    all_values = [
        float(row[1]) if isinstance(row[1], str) else row[1]
        for row in top_n_annotators + bottom_n_annotators
    ]
    max_abs_value = abs(max(all_values))
    min_abs_value = abs(min(all_values))

    return top_n_annotators, bottom_n_annotators, max_abs_value, min_abs_value


def generate_latex_table(
    annotators_data: list[list[str | float]],
    metric_names: list[str],
    title: str | None,
    minipage_width: float,
    first_col_width: float,
    metric_col_width: float,
    vertical_spacing: float,
    get_color_intensity: Callable[[float], float] | None = None,
    special_configs: dict | None = None,
    precision: int = 2,
):
    """Generate LaTeX code for a table of annotators.

    Args:
        annotators_data (list[list[str | float]]): List of [annotator, value1, value2, ...] lists where each value
                        corresponds to a metric in metric_names
        metric_names (list[str]): List of names for the metrics being displayed
        title (str | None): Title for this table section
        minipage_width (float): Width of the minipage as fraction of textwidth
        first_col_width (float): Width of first column as fraction of linewidth
        metric_col_width (float): Width of each metric column as fraction of linewidth
        vertical_spacing (float): Vertical spacing between rows in pt
        get_color_intensity (Callable[[float], float]): Function to calculate color intensity

    Returns:
        List of LaTeX code lines
    """
    latex = []

    # Start minipage
    latex.append(r"\begin{minipage}[t]{" + str(minipage_width) + r"\textwidth}")
    latex.append(r"\centering")
    latex.append(r"\sffamily")  # Sans serif font for the entire table
    latex.append(r"\tablefontsize")  # Apply the font size

    # Add title
    if title is not None:
        latex.append(f"\\textbf{{{title}}}\\\\[0.5em]")

    if special_configs is None:
        special_configs = {}

    # Begin table
    latex.append(r"\begin{tabular}{")
    # First column for annotator names
    latex.append(
        r"    >{\raggedright\arraybackslash}p{" + str(first_col_width) + r"\linewidth}"
    )
    # Add a column for each metric
    for _ in metric_names:
        latex.append(r"    @{\hspace{" + str(vertical_spacing) + r"pt}} ")
        latex.append(
            r"    >{\centering\arraybackslash}p{"
            + str(metric_col_width)
            + r"\linewidth}"
        )
    latex.append(r"}")
    latex.append(r"")

    # Column headers
    header = "\\textbf{Generating a response that...}"
    for metric_name in metric_names:
        header += f" & \\textbf{{{metric_name}}}"
    header += " \\\\"
    latex.append(header)
    latex.append(r"\toprule")

    # Data rows
    for i, row_data in enumerate(annotators_data):
        annotator = row_data[0]
        values = row_data[1:]

        # Start the row with annotator name and row color if needed
        if i % 2 == 0:
            row = f"\\rowcolor{{altrow}} {annotator}"
        else:
            row = annotator

        for j, value in enumerate(values):
            col_name = metric_names[j]

            intensity = get_color_intensity(value)
            pos_color = "poscolor"
            neg_color = "negcolor"

            # If column has special configs, apply them to
            # overwrite default values
            if col_name in special_configs:
                if "get_color_intensity" in special_configs[col_name]:
                    intensity = special_configs[col_name]["get_color_intensity"](value)
                if "pos_color" in special_configs[col_name]:
                    pos_color = special_configs[col_name]["pos_color"]
                if "neg_color" in special_configs[col_name]:
                    neg_color = special_configs[col_name]["neg_color"]

            color_name = pos_color if value >= 0 else neg_color
            # Default precision is 3 decimal places
            row += (
                f" & \\cellcolor{{{color_name}!{intensity}}}{{{value:.{precision}f}}}"
            )

        row += " \\\\"
        latex.append(row)

        # Add spacing after each row (except the last one)
        if i < len(annotators_data) - 1:
            latex.append(r"\addlinespace[\rowspacing]")

    # End table
    latex.append(r"\end{tabular}")
    latex.append(r"\end{minipage}")

    return latex


def get_latex_doc_preamble():
    """Get the LaTeX preamble for a document."""

    latex = []

    latex.append(r"\usepackage{colortbl}")
    latex.append(r"\usepackage{xcolor}")
    latex.append(r"\usepackage{array}")
    latex.append(r"\usepackage{booktabs}")
    latex.append(
        r"\definecolor{poscolor}{HTML}{9eb0ff} % Light blue for positive values"
    )
    latex.append(
        r"\definecolor{negcolor}{HTML}{ffadad} % Light red for negative values"
    )
    latex.append(
        r"\definecolor{altrow}{HTML}{f5f5f5}   % Light grey for alternating rows"
    )
    latex.append(r"\definecolor{headercolor}{HTML}{fcf1cf} % Light yellow for headers")
    latex.append(
        r"\definecolor{lightgrey}{RGB}{210,210,210}  % Less light grey as alternative value color"
    )

    latex.append(r"\newlength{\rowspacing}")
    latex.append(r"\setlength{\rowspacing}{1.5pt}")
    latex.append(r"\newcommand{\tablefontsize}{\scriptsize}")

    return "\n".join(latex)


def add_table_preamble(latex: list, title: str):
    """Add the table preamble to the LaTeX code."""

    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{" + title + r"}")
    latex.append(r"\vspace{0.5em}")  # Add vertical space after caption
    latex.append(r"\renewcommand{\arraystretch}{1.1}")
    latex.append(r"\setlength{\rowspacing}{1.5pt}")
    latex.append(r"\renewcommand{\tablefontsize}{\scriptsize}")

    return latex


def add_table_postamble(latex: list):
    """Add the table postamble to the LaTeX code."""
    latex.append(r"\end{table}")
    return latex


def get_intensity_callable(max_abs_value: float, min_abs_value: float):
    """Get a callable that calculates the intensity of a value."""

    def get_color_intensity(value):
        if value == 0:
            return 0
        elif value > 0:
            normalized_val = value / max_abs_value if max_abs_value > 0 else 0
        else:
            normalized_val = abs(value) / min_abs_value if min_abs_value > 0 else 0
        intensity = normalized_val * 100
        return float(f"{intensity:.1f}")

    return get_color_intensity


def get_latex_top_and_bottom_annotators(
    annotator_metrics: dict,
    metric_name: str,
    top_n: int = 5,
    bottom_n: int = 5,
    top_title: str = "Five most encouraged traits",
    bottom_title: str = "Five most discouraged traits",
) -> str:
    """Generate LaTeX code for just the table content showing top and bottom annotators.

    This version doesn't include document preamble or \begin{document} tags,
    making it suitable for inclusion in an existing LaTeX document.

    Args:
        annotator_metrics: Dictionary mapping annotator names to metric values
        metric_name: Name of the metric being displayed
        top_n: Number of top annotators to show
        bottom_n: Number of bottom annotators to show
        top_title: Title of the top annotators table
        bottom_title: Title of the bottom annotators table

    Returns:
        String containing LaTeX code for just the table content
    """
    MINIPAGE_WIDTH = 0.48
    FIRST_COLUMN_WIDTH = 0.7
    SECOND_COLUMN_WIDTH = 0.18

    top_n_annotators, bottom_n_annotators, max_abs_value, min_abs_value = (
        get_top_and_bottom_annotators(
            annotator_metrics, top_n, bottom_n, format_values=False
        )
    )

    # Start building the LaTeX code
    latex = []
    # latex = add_table_preamble(latex, title)

    # Generate top annotators table
    top_table = generate_latex_table(
        annotators_data=top_n_annotators,
        metric_names=[metric_name],
        title=top_title,
        minipage_width=MINIPAGE_WIDTH,
        first_col_width=FIRST_COLUMN_WIDTH,
        metric_col_width=SECOND_COLUMN_WIDTH,
        vertical_spacing=10,
        get_color_intensity=get_intensity_callable(max_abs_value, min_abs_value),
    )
    latex.extend(top_table)

    # Add spacing between tables
    latex.append(r"\hfill")

    # Generate bottom annotators table
    # get number as written out string
    bottom_table = generate_latex_table(
        annotators_data=bottom_n_annotators,
        metric_names=[metric_name],
        title=bottom_title,
        minipage_width=MINIPAGE_WIDTH,
        first_col_width=FIRST_COLUMN_WIDTH,
        metric_col_width=SECOND_COLUMN_WIDTH,
        vertical_spacing=10,
        get_color_intensity=get_intensity_callable(max_abs_value, min_abs_value),
    )
    latex.extend(bottom_table)

    # latex = add_table_postamble(latex)

    return "\n".join(latex)


def get_latex_table_from_metrics_df(
    metrics_df: pd.DataFrame,
    title: str,
    first_col_width: float = 0.2,
):
    latex = []
    # latex = add_table_preamble(latex, title=title)

    metric_col_width = (0.8 - first_col_width) / (len(metrics_df.columns[1:]) * 1.1)

    max_abs_value = abs(metrics_df.iloc[:, 1:-1].max(axis=1).max())
    min_abs_value = abs(metrics_df.iloc[:, 1:-1].min(axis=1).min())

    max_diff_value = metrics_df["Max diff"].max()

    get_intensities = get_intensity_callable(max_abs_value, min_abs_value)

    table = generate_latex_table(
        annotators_data=metrics_df.to_numpy(),
        metric_names=list(metrics_df.columns[1:]),
        title=None,
        minipage_width=1,
        first_col_width=first_col_width,
        metric_col_width=metric_col_width,
        get_color_intensity=get_intensities,
        vertical_spacing=13.5,
        special_configs={
            "Max diff": {
                "get_color_intensity": get_intensity_callable(max_diff_value, 0),
                "pos_color": "lightgrey",
                "neg_color": "lightgrey",
            }
        },
    )
    latex.extend(table)
    # latex = add_table_postamble(latex)

    latex_str = "\n".join(latex)
    return latex_str

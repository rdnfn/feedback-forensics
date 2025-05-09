"""Plotting functions for the paper"""

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


def plot_top_and_bottom_annotators(
    annotator_metrics: dict,
    metric_name: str,
    top_n: int = 5,
    bottom_n: int = 5,
) -> None:

    top_n_annotators, bottom_n_annotators, max_abs_value = (
        get_top_and_bottom_annotators(
            annotator_metrics, top_n, bottom_n, format_values=True
        )
    )

    # Calculate an appropriate figure height based on the number of rows
    row_height = 0.25  # inches per row
    header_height = 0.4  # extra inches for header and padding
    title_height = 0.5  # inches for title
    fig_height = (
        max(len(top_n_annotators), len(bottom_n_annotators)) * row_height
        + header_height
        + title_height
    )

    # Create figure with adaptive height
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, fig_height))

    # Function to get color based on value
    def get_color(value):
        normalized_val = abs(float(value)) / max_abs_value if max_abs_value > 0 else 0
        # Ensure color intensity is visible even for small values
        opacity = 0.3 + (normalized_val * 0.7)

        if float(value) >= 0:
            return mcolors.to_rgba(POSITIVE_COLOR, opacity)
        else:
            return mcolors.to_rgba(NEGATIVE_COLOR, opacity)

    # Create tables
    top_table = ax1.table(
        cellText=top_n_annotators,
        colLabels=["Annotator", metric_name],
        loc="center",
        cellLoc="left",  # Left align all cells by default
        colWidths=[0.6, 0.4],
    )
    bottom_table = ax2.table(
        cellText=bottom_n_annotators,
        colLabels=["Annotator", metric_name],
        loc="center",
        cellLoc="left",  # Left align all cells by default
        colWidths=[0.6, 0.4],
    )

    # Style tables
    for table in [top_table, bottom_table]:
        table.auto_set_font_size(False)
        table.set_fontsize(12)  # Larger font size
        table.scale(1, 1.5)  # Scale to make cells taller

        # Style header row
        for j, cell in table._cells.items():
            # Remove black borders by setting edge color to white
            cell.set_edgecolor("white")

            if j[0] == 0:  # Header row
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("grey")
                cell.set_text_props(ha="center")  # Center align headers
            elif j[1] == 0:  # First column (annotators)
                cell.set_text_props(ha="left")  # Left align annotators
                # Add alternating background color for odd rows in annotator column
                if j[0] % 2 == 1:  # Odd rows (1, 3, 5...)
                    cell.set_facecolor(ALTERNATE_ROW_COLOR)
            elif j[1] == 1:  # Value column
                cell.set_text_props(ha="center")  # Center align values

                # Add gradient color to values
                value_text = cell.get_text().get_text()
                cell.set_facecolor(get_color(value_text))

    # Set titles with minimal padding and position manually
    title1 = ax1.set_title(
        f"Top {top_n} Annotators",
        fontsize=TITLE_FONT_SIZE,
        fontweight=TITLE_FONT_WEIGHT,
        pad=0,  # No padding
    )
    title2 = ax2.set_title(
        f"Bottom {bottom_n} Annotators",
        fontsize=TITLE_FONT_SIZE,
        fontweight=TITLE_FONT_WEIGHT,
        pad=0,  # No padding
    )

    # Adjust vertical position of titles to be closer to tables
    title1.set_position([0.5, 1.0])
    title2.set_position([0.5, 1.0])

    ax1.axis("off")
    ax2.axis("off")

    # Apply tight layout with less vertical space
    plt.tight_layout(pad=0.5)

    # Fine-tune the vertical margins
    plt.subplots_adjust(top=0.95, bottom=0.05)

    plt.show()

    return fig


def generate_latex_table(
    annotators_data,
    metric_name,
    title,
    minipage_width,
    first_col_width,
    second_col_width,
    get_color_intensity,
):
    """Generate LaTeX code for a table of annotators.

    Args:
        annotators_data: List of [annotator, value] pairs
        metric_name: Name of the metric being displayed
        title: Title for this table section
        minipage_width: Width of the minipage as fraction of textwidth
        first_col_width: Width of first column as fraction of linewidth
        second_col_width: Width of second column as fraction of linewidth
        get_color_intensity: Function to calculate color intensity

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
    latex.append(f"\\textbf{{{title}}}\\\\[0.5em]")

    # Begin table
    latex.append(r"\begin{tabular}{")
    latex.append(
        r"    >{\raggedright\arraybackslash}p{" + str(first_col_width) + r"\linewidth}"
    )
    latex.append(
        r"    >{\centering\arraybackslash}p{" + str(second_col_width) + r"\linewidth}"
    )
    latex.append(r"}")
    latex.append(r"")

    # Column headers
    latex.append(
        f"\\textbf{{Generating a response that...}} & \\textbf{{{metric_name}}} \\\\"
    )
    latex.append(r"\toprule")

    # Data rows
    for i, (annotator, value) in enumerate(annotators_data):
        intensity = get_color_intensity(value)
        # Choose color based on the sign of the value
        color_name = "poscolor" if value >= 0 else "negcolor"

        if i % 2 == 0:
            latex.append(
                f"\\rowcolor{{altrow}} {annotator} & \\cellcolor{{{color_name}!{intensity}}}{{{value:.3f}}} \\\\"
            )
        else:
            latex.append(
                f"{annotator} & \\cellcolor{{{color_name}!{intensity}}}{{{value:.3f}}} \\\\"
            )

        # Add spacing after each row (except the last one)
        if i < len(annotators_data) - 1:
            latex.append(r"\addlinespace[\rowspacing]")

    # End table
    latex.append(r"\end{tabular}")
    latex.append(r"\end{minipage}")

    return latex


def get_latex_top_and_bottom_annotators(
    annotator_metrics: dict,
    metric_name: str,
    top_n: int = 5,
    bottom_n: int = 5,
    title: str = "Encouraged Personality Traits",
) -> str:
    """Generate LaTeX code for just the table content showing top and bottom annotators.

    This version doesn't include document preamble or \begin{document} tags,
    making it suitable for inclusion in an existing LaTeX document.

    Args:
        annotator_metrics: Dictionary mapping annotator names to metric values
        metric_name: Name of the metric being displayed
        top_n: Number of top annotators to show
        bottom_n: Number of bottom annotators to show

    Returns:
        String containing LaTeX code for just the table content
    """
    MINIPAGE_WIDTH = 0.45
    FIRST_COLUMN_WIDTH = 0.82
    SECOND_COLUMN_WIDTH = 0.18

    top_n_annotators, bottom_n_annotators, max_abs_value, min_abs_value = (
        get_top_and_bottom_annotators(
            annotator_metrics, top_n, bottom_n, format_values=False
        )
    )

    # Function to calculate color intensity based on value
    def get_color_intensity(value):
        if value == 0:
            return 0
        elif value > 0:
            normalized_val = value / max_abs_value if max_abs_value > 0 else 0
        else:
            normalized_val = abs(value) / min_abs_value if min_abs_value > 0 else 0
        intensity = normalized_val * 100
        return float(f"{intensity:.1f}")

    # Start building the LaTeX code
    latex = []

    # Begin table
    latex.append(r"\begin{table}")
    latex.append(r"\centering")
    latex.append(r"\caption{" + title + r"}")
    latex.append(r"\renewcommand{\arraystretch}{1.5}")

    # Define row spacing variable
    latex.append(r"\newlength{\rowspacing}")
    latex.append(r"\setlength{\rowspacing}{1.5pt}")

    # Define font size command that can be changed
    latex.append(r"\newcommand{\tablefontsize}{\scriptsize}")

    # Define colors
    latex.append(
        r"\definecolor{poscolor}{RGB}{158,176,255} % Light blue for positive values"
    )
    latex.append(
        r"\definecolor{negcolor}{RGB}{255,173,173} % Light red for negative values"
    )
    latex.append(
        r"\definecolor{altrow}{RGB}{245,245,245}   % Light grey for alternating rows"
    )

    # Generate top annotators table
    top_table = generate_latex_table(
        top_n_annotators,
        metric_name,
        "Five most encouraged traits",
        MINIPAGE_WIDTH,
        FIRST_COLUMN_WIDTH,
        SECOND_COLUMN_WIDTH,
        get_color_intensity,
    )
    latex.extend(top_table)

    # Add spacing between tables
    latex.append(r"\hfill")

    # Generate bottom annotators table
    # get number as written out string
    bottom_table = generate_latex_table(
        bottom_n_annotators,
        metric_name,
        "Five most discouraged traits",
        MINIPAGE_WIDTH,
        FIRST_COLUMN_WIDTH,
        SECOND_COLUMN_WIDTH,
        get_color_intensity,
    )
    latex.extend(bottom_table)

    # End the table environment
    latex.append(r"\end{table}")

    return "\n".join(latex)

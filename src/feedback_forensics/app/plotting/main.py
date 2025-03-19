import pandas as pd

import feedback_forensics.app.metrics
from feedback_forensics.app.plotting.table import create_fig_with_tables
from feedback_forensics.app.plotting.table import get_table_contents_from_metrics


def generate_plot(
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

    principles_metrics_dfs = get_table_contents_from_metrics(metrics)

    fig = create_fig_with_tables(
        table_titles=["Basic Statistics", "Implicit Objectives"],
        table_dfs=[overall_metrics_df, principles_metrics_dfs],
        color_scales=[None, "berlin"],
        index_col_headings=["Metric", "Annotations prefer a response that ..."],
        neutral_values=[0.0, 0.0],
    )

    return fig

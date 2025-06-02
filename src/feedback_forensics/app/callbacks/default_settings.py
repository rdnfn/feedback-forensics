"""Callbacks to set the default settings for all input, output and state components."""

import pandas as pd
from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    ENABLE_EXAMPLE_VIEWER,
    EXAMPLE_VIEWER_NO_DATA_MESSAGE,
)
from feedback_forensics.data.datasets import get_default_dataset_names


def _get_default_values():
    return {
        "inp": {
            "active_datasets_dropdown": {
                "value": lambda: (
                    get_default_dataset_names()[0]
                    if get_default_dataset_names()
                    else None
                )
            },
            "analysis_type_radio": {"value": "model_analysis"},
            "enable_dataviewer_checkbox": {"value": ENABLE_EXAMPLE_VIEWER},
            "enable_multiple_datasets_checkbox": {"value": False},
            "split_col_dropdown": {"value": NONE_SELECTED_VALUE},
            "split_col_selected_vals_dropdown": {"value": None},
            "models_to_compare_dropdown": {"value": None},
            "annotations_to_compare_dropdown": {"value": None},
            "annotator_cols_dropdown": {"value": None},
            "annotator_rows_dropdown": {"value": None},
            "reference_models_dropdown": {"value": None},
            "metric_name_dropdown": {"value": "strength"},
            "sort_by_dropdown": {"value": None},
            "sort_order_dropdown": {"value": "Descending"},
            "example_annotator_1": {"value": None},
            "example_annotator_2": {"value": None},
            "example_subset_dropdown": {"value": "all"},
            "example_index_slider": {"value": 0},
            "results_view_radio": {"value": "numerical_results"},
            "example_dataset_dropdown": {"value": None},
        },
        "out": {
            "overall_metrics_table": {"value": lambda: pd.DataFrame()},
            "annotator_table": {"value": lambda: pd.DataFrame()},
            "example_message": {"value": EXAMPLE_VIEWER_NO_DATA_MESSAGE},
            "share_link": {"value": ""},
            "example_comparison_id": {"value": ""},
            "example_prompt": {"value": ""},
            "example_response_a_model": {"value": ""},
            "example_response_b_model": {"value": ""},
            "example_response_a": {"value": ""},
            "example_response_b": {"value": ""},
            "example_annotator_1_result": {"value": ""},
            "example_annotator_2_result": {"value": ""},
            "example_metadata": {"value": {}},
        },
        "state": {
            "app_url": {"value": ""},
            "datapath": {"value": ""},
            "df": {"value": lambda: pd.DataFrame()},
            "unfiltered_df": {"value": lambda: pd.DataFrame()},
            "dataset_name": {"value": ""},
            "active_dataset": {"value": ""},
            "cache": {"value": {}},
            "avail_datasets": {"value": {}},
            "computed_annotator_metrics": {"value": {}},
            "computed_overall_metrics": {"value": {}},
            "default_annotator_cols": {"value": []},
            "default_annotator_rows": {"value": []},
            "votes_dicts": {"value": {}},
        },
    }


def generate(
    inp: dict,
    state: dict,
    out: dict,
) -> dict:
    """Generate callbacks for default settings."""

    def set_default_settings(
        inp: dict,
        state: dict,
        out: dict,
    ) -> dict:
        """Set the default settings for all input, output and state components."""
        defaults = _get_default_values()
        for component_dict, category in zip([inp, state, out], ["inp", "state", "out"]):
            for key in defaults[category].items():
                component_dict[key] = defaults[category][key]["value"]

        return inp, state, out

    return {"set_default_settings": set_default_settings}

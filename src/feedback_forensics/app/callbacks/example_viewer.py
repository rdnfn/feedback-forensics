"""Callbacks for the example viewer functionality."""

import pandas as pd
import gradio as gr
from loguru import logger

from feedback_forensics.app.constants import (
    EXAMPLE_VIEWER_NO_DATA_MESSAGE,
    EXAMPLE_VIEWER_MULTIPLE_DATASETS_MESSAGE,
    NONE_SELECTED_VALUE,
)

from feedback_forensics.app.metrics import ensure_categories_identical


def generate(inp: dict, state: dict, out: dict) -> dict:
    """Generate callbacks for the example viewer."""

    def _generate_non_functional_slider():
        return gr.Slider(value=0, interactive=False)

    def _get_empty_viewer_option_dict():
        return {
            inp["example_dataset_dropdown"]: gr.Dropdown(
                choices=[], value=None, interactive=False
            ),
            inp["example_annotator_1"]: gr.Dropdown(
                choices=[], value=None, interactive=False
            ),
            inp["example_annotator_2"]: gr.Dropdown(
                choices=[], value=None, interactive=False
            ),
            inp["example_index_slider"]: _generate_non_functional_slider(),
            inp["example_subset_dropdown"]: gr.Dropdown(
                choices=[], value=None, interactive=False
            ),
        }

    def _get_dataset_col_name(
        dataset_name: str, dataset_names: list[str], votes_dicts: dict
    ) -> str:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        if len(dataset_names) >= 2:
            dataset_col_name = [
                col_name for col_name in votes_dicts.keys() if dataset_name in col_name
            ][0]
        else:
            # Note that if it's only a single dataset
            # the dataset name is not included in the column
            dataset_col_name = list(votes_dicts.keys())[0]
        return dataset_col_name

    def update_example_viewer_options(data):
        """Update the example viewer dropdown options based on loaded data.

        Triggered by the active datasets dropdown."""

        votes_dicts = data.get(state["votes_dicts"], {})
        dataset_names = data[inp["active_datasets_dropdown"]]
        selected_dataset = data[inp["example_dataset_dropdown"]]
        annotator_1 = data[inp["example_annotator_1"]]
        annotator_2 = data[inp["example_annotator_2"]]
        subset_filter = data[inp["example_subset_dropdown"]]
        slider_value = data[inp["example_index_slider"]]

        if not votes_dicts or not dataset_names:
            return _get_empty_viewer_option_dict()

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        if len(dataset_names) > 1:
            return {
                **_empty_example_display(
                    out, message=EXAMPLE_VIEWER_MULTIPLE_DATASETS_MESSAGE
                ),
                **_get_empty_viewer_option_dict(),
            }

        if not selected_dataset or selected_dataset not in dataset_names:
            selected_dataset = dataset_names[0]

        selected_dataset_col_name = _get_dataset_col_name(
            selected_dataset, dataset_names, votes_dicts
        )

        selected_votes_dict = votes_dicts[selected_dataset_col_name]
        annotator_metadata = selected_votes_dict.get("annotator_metadata", {})

        # Get annotator choices (visible names)
        annotator_visible_names = [
            metadata["annotator_visible_name"]
            for metadata in annotator_metadata.values()
        ]
        annotator_choices = annotator_visible_names

        if annotator_1 is None:
            annotator_1 = annotator_choices[0]

        if annotator_2 is None:
            if len(annotator_choices) > 1:
                annotator_2 = annotator_choices[1]
            else:
                annotator_2 = annotator_choices[0]

        # Get number of examples for slider
        max_examples = 0
        votes_dict = votes_dicts[selected_dataset_col_name]
        df = votes_dict.get("df")
        if df is not None:
            filtered_df = _filter_dataframe(
                df=df,
                votes_dict=votes_dict,
                annotator_1=annotator_1,
                annotator_2=annotator_2,
                subset_filter=subset_filter,
            )
            max_examples = max(0, len(filtered_df) - 1)
            slider_value = min(slider_value, max_examples)

        data[inp["example_dataset_dropdown"]] = selected_dataset
        data[inp["example_annotator_1"]] = annotator_1
        data[inp["example_annotator_2"]] = annotator_2
        data[inp["example_index_slider"]] = slider_value

        return {
            inp["example_dataset_dropdown"]: gr.Dropdown(
                choices=dataset_names, value=selected_dataset, interactive=True
            ),
            inp["example_annotator_1"]: gr.Dropdown(
                choices=annotator_choices,
                value=annotator_1,
                interactive=True,
            ),
            inp["example_annotator_2"]: gr.Dropdown(
                choices=annotator_choices,
                value=annotator_2,
                interactive=True,
            ),
            inp["example_index_slider"]: gr.Slider(
                minimum=0,
                maximum=max(1, max_examples),
                value=slider_value,
                interactive=max_examples > 0,
            ),
            **display_example(data),
        }

    def display_example(data):
        """Display the selected example details."""
        selected_dataset = data[inp["example_dataset_dropdown"]]
        annotator_1 = data[inp["example_annotator_1"]]
        annotator_2 = data[inp["example_annotator_2"]]
        subset_filter = data[inp["example_subset_dropdown"]]
        example_index = int(data[inp["example_index_slider"]])
        votes_dicts = data.get(state["votes_dicts"], {})
        dataset_names = data[inp["active_datasets_dropdown"]]

        dataset_col_name = _get_dataset_col_name(
            selected_dataset, dataset_names, votes_dicts
        )

        if dataset_col_name not in votes_dicts:
            logger.warning(
                "Example datapoint viewer: selected_dataset not in votes_dicts"
                f"selected_dataset: {dataset_col_name}, "
                f"votes_dicts: {list(votes_dicts.keys())}"
            )
            return _empty_example_display(out)

        votes_dict = votes_dicts[dataset_col_name]
        df = votes_dict.get("df")
        annotator_metadata = votes_dict.get("annotator_metadata", {})

        if df is None or len(df) == 0:
            logger.warning("Example datapoint viewer: df is None or empty")
            return _empty_example_display(
                out,
            )

        # Filter dataframe based on subset selection
        filtered_df = _filter_dataframe(
            df, votes_dict, annotator_1, annotator_2, subset_filter
        )

        if len(filtered_df) == 0 or example_index >= len(filtered_df):
            logger.warning(
                "Example datapoint viewer: filtered_df is empty or example_index is out of range"
                f"filtered_df: {filtered_df}"
                f"example_index: {example_index}"
            )
            return _empty_example_display(
                out,
                gr_warning_message="No examples found.",
            )

        # Get the selected row
        row = filtered_df.iloc[example_index]

        # Extract basic information
        comparison_id = str(row.get("comparison_id", "N/A"))
        prompt = str(row.get("prompt", "N/A"))
        model_a = str(row.get("model_a", "N/A"))
        model_b = str(row.get("model_b", "N/A"))

        # Handle both format versions (1.0 and 2.0)
        text_a = "N/A"
        text_b = "N/A"

        if "text_a" in row and "text_b" in row:
            # Format 1.0 or processed format 2.0
            text_a = str(row.get("text_a", "N/A"))
            text_b = str(row.get("text_b", "N/A"))
        else:
            # Try to find text in other columns (format 2.0 might have different column names)
            for col in row.index:
                if col.endswith("_a") and "text" in col:
                    text_a = str(row.get(col, "N/A"))
                elif col.endswith("_b") and "text" in col:
                    text_b = str(row.get(col, "N/A"))

        # Get annotator results
        annotator_1_result = "N/A"
        annotator_2_result = "N/A"

        if annotator_1 and annotator_2:
            # Find annotator column names from visible names
            ann1_col_name = _get_annotator_col_from_visible_name(
                annotator_metadata, annotator_1
            )
            ann2_col_name = _get_annotator_col_from_visible_name(
                annotator_metadata, annotator_2
            )

            if ann1_col_name and ann1_col_name in row:
                annotator_1_result = str(row[ann1_col_name])

            if ann2_col_name and ann2_col_name in row:
                annotator_2_result = str(row[ann2_col_name])

        # Extract metadata (exclude main columns)
        metadata_keys = ["comparison_id", "prompt", "text_a", "text_b"] + list(
            annotator_metadata.keys()
        )
        metadata = {
            key: str(value)
            for key, value in row.items()
            if key not in metadata_keys and not pd.isna(value)
        }

        return {
            out["example_comparison_id"]: comparison_id,
            out["example_prompt"]: gr.Textbox(value=prompt, visible=prompt != "N/A"),
            out["example_response_a_model"]: model_a,
            out["example_response_b_model"]: model_b,
            out["example_response_a"]: gr.Textbox(
                value=text_a,
            ),
            out["example_response_b"]: gr.Textbox(
                value=text_b,
            ),
            out["example_annotator_1_result"]: gr.Textbox(
                label=f"Annotator 1 preference ({annotator_1})",
                value=annotator_1_result,
                interactive=False,
                text_align="right" if annotator_1_result == "text_b" else "left",
            ),
            out["example_annotator_2_result"]: gr.Textbox(
                label=f"Annotator 2 preference ({annotator_2})",
                value=annotator_2_result,
                interactive=False,
                text_align="right" if annotator_2_result == "text_b" else "left",
            ),
            out["example_metadata"]: metadata,
            out["example_message"]: gr.Markdown(visible=False, value=""),
            out["example_details_group"]: gr.Group(visible=True),
        }

    def launch_example_viewer_from_annotator_table(evt: gr.EventData, data):
        annotator_cols = data[inp["annotator_cols_dropdown"]]
        annotator_rows = data[inp["annotator_rows_dropdown"]]
        df_values: gr.DataFrame = data[out["annotator_table"]].values

        empty_return = {out["annotator_table"]: gr.DataFrame()}

        is_multiple_datasets = (
            not isinstance(data[inp["active_datasets_dropdown"]], str)
            and len(list(data[inp["active_datasets_dropdown"]])) > 1
        )
        if is_multiple_datasets:
            gr.Warning(
                "Data viewer: multiple datasets are active. "
                "Data viewer is currently not supported for multiple datasets. "
                "Please select a single dataset to view examples."
            )
            return empty_return

        if not data[inp["enable_dataviewer_checkbox"]]:
            gr.Info(
                "Data viewer currently not enabled. "
                "Set 'Enable dataviewer' checkbox in "
                "advanced settings to view datapoints."
            )
            return empty_return

        if (
            data[inp["split_col_dropdown"]] != NONE_SELECTED_VALUE
            and data[inp["split_col_selected_vals_dropdown"]] is not None
        ):
            gr.Warning(
                "Example datapoint viewer: currently not available when splitting by column."
            )
            return empty_return

        index = evt._data["index"]
        value = evt._data["value"]
        y_idx = index[0]
        x_idx = index[1]
        if index[1] == 0:
            # selected annotator column, skipping going to example viewer
            return empty_return
        else:
            selected_annotator_col = annotator_cols[x_idx - 1]
            selected_annotator_row_shown_name = df_values[y_idx][0]
            selected_annotator_row_potential_name = [
                annotator
                for annotator in annotator_rows
                if selected_annotator_row_shown_name in annotator
            ]
            if len(selected_annotator_row_potential_name) == 0:
                logger.warning(
                    (
                        f"Selected annotator row {selected_annotator_row_shown_name} not "
                        f"found in annotator rows ({annotator_rows})"
                    )
                )
                return empty_return
            elif len(selected_annotator_row_potential_name) > 1:
                logger.warning(
                    (
                        f"Selected annotator row {selected_annotator_row_shown_name} "
                        f"found multiple times in annotator rows ({annotator_rows})"
                        " Annotator names shown in table are not unique."
                    )
                )
                return empty_return
            else:
                selected_annotator_row = selected_annotator_row_potential_name[0]

        data[inp["example_annotator_1"]] = selected_annotator_row
        data[inp["example_annotator_2"]] = selected_annotator_col

        if value >= 0:
            subset_val = "agree"
        else:
            subset_val = "disagree"

        data[inp["example_subset_dropdown"]] = subset_val

        gr.Info(
            f"Showing example datapoints where annotations by '{selected_annotator_row}' and '{selected_annotator_col}' {subset_val}."
        )

        return {
            inp["results_view_radio"]: "example_viewer",
            out["example_details_group"]: gr.Group(visible=True),
            inp["numerical_results_col"]: gr.Column(visible=False),
            inp["example_subset_dropdown"]: gr.Dropdown(
                value=subset_val,
            ),
            **update_example_viewer_options(data),
        }

    return {
        "update_example_viewer_options": update_example_viewer_options,
        "display_example": display_example,
        "launch_example_viewer_from_annotator_table": launch_example_viewer_from_annotator_table,
    }


def _get_annotator_col_from_visible_name(
    annotator_metadata: dict, visible_name: str
) -> str:
    """Get the annotator column name from the visible name."""
    for col_name, metadata in annotator_metadata.items():
        if metadata.get("annotator_visible_name") == visible_name:
            return col_name
    return None


def _filter_dataframe(
    df: pd.DataFrame,
    votes_dict: dict,
    annotator_1: str,
    annotator_2: str,
    subset_filter: str,
) -> pd.DataFrame:
    """Filter dataframe based on subset selection."""
    if subset_filter == "all":
        return df

    if not annotator_1 or not annotator_2:
        return df

    annotator_metadata = votes_dict.get("annotator_metadata", {})
    ann1_col_name = _get_annotator_col_from_visible_name(
        annotator_metadata, annotator_1
    )
    ann2_col_name = _get_annotator_col_from_visible_name(
        annotator_metadata, annotator_2
    )

    if not ann1_col_name or not ann2_col_name:
        return df

    if ann1_col_name not in list(df.columns) or ann2_col_name not in list(df.columns):
        return df

    df = ensure_categories_identical(df=df, col_a=ann1_col_name, col_b=ann2_col_name)

    # Create filter masks
    if subset_filter == "agree":
        mask = df[ann1_col_name] == df[ann2_col_name]
    elif subset_filter == "disagree":
        mask = (
            (df[ann1_col_name] != df[ann2_col_name])
            & (df[ann1_col_name].isin(["text_a", "text_b"]))
            & (df[ann2_col_name].isin(["text_a", "text_b"]))
        )
    elif subset_filter == "only annotator 1 does not apply":
        mask = (~df[ann1_col_name].isin(["text_a", "text_b"])) & (
            df[ann2_col_name].isin(["text_a", "text_b"])
        )
    elif subset_filter == "only annotator 2 does not apply":
        mask = (df[ann1_col_name].isin(["text_a", "text_b"])) & (
            ~df[ann2_col_name].isin(["text_a", "text_b"])
        )
    elif subset_filter == "neither apply":
        mask = (~df[ann1_col_name].isin(["text_a", "text_b"])) & (
            ~df[ann2_col_name].isin(["text_a", "text_b"])
        )
    else:
        return df

    return df[mask].reset_index(drop=True)


def _empty_example_display(
    out: dict,
    message: str = EXAMPLE_VIEWER_NO_DATA_MESSAGE,
    gr_warning_message: str = None,
) -> dict:
    """Return empty example display values."""
    if gr_warning_message:
        gr.Warning(gr_warning_message)
    return {
        out["example_comparison_id"]: "",
        out["example_prompt"]: "",
        out["example_response_a_model"]: "",
        out["example_response_b_model"]: "",
        out["example_response_a"]: "",
        out["example_response_b"]: "",
        out["example_annotator_1_result"]: "",
        out["example_annotator_2_result"]: "",
        out["example_metadata"]: {},
        out["example_message"]: gr.Markdown(visible=True, value=message),
        out["example_details_group"]: gr.Group(visible=False),
    }

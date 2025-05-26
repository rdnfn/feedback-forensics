"""Callbacks for the example viewer functionality."""

import pandas as pd
import gradio as gr
from loguru import logger


def generate(inp: dict, state: dict, out: dict) -> dict:
    """Generate callbacks for the example viewer."""

    def _generate_non_functional_slider():
        return gr.Slider(value=0, interactive=False)

    def _get_empty_viewer_option_dict():
        return {
            inp["example_dataset_dropdown"]: gr.Dropdown(choices=[], value=None),
            inp["example_annotator_row_dropdown"]: gr.Dropdown(choices=[], value=None),
            inp["example_annotator_col_dropdown"]: gr.Dropdown(choices=[], value=None),
            inp["example_index_slider"]: _generate_non_functional_slider(),
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
        annotator_row = data[inp["example_annotator_row_dropdown"]]
        annotator_col = data[inp["example_annotator_col_dropdown"]]
        subset_filter = data[inp["example_subset_dropdown"]]
        slider_value = data[inp["example_index_slider"]]

        if not votes_dicts or not dataset_names:
            return _get_empty_viewer_option_dict()

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

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

        if annotator_row is None:
            annotator_row = annotator_choices[0]

        if annotator_col is None:
            if len(annotator_choices) > 1:
                annotator_col = annotator_choices[1]
            else:
                annotator_col = annotator_choices[0]

        # Get number of examples for slider
        max_examples = 0
        votes_dict = votes_dicts[selected_dataset_col_name]
        df = votes_dict.get("df")
        if df is not None:
            filtered_df = _filter_dataframe(
                df=df,
                votes_dict=votes_dict,
                annotator_row=annotator_row,
                annotator_col=annotator_col,
                subset_filter=subset_filter,
            )
            max_examples = max(0, len(filtered_df) - 1)
            slider_value = min(slider_value, max_examples)

        data[inp["example_dataset_dropdown"]] = selected_dataset
        data[inp["example_annotator_row_dropdown"]] = annotator_row
        data[inp["example_annotator_col_dropdown"]] = annotator_col
        data[inp["example_index_slider"]] = slider_value

        logger.info(f"Updating example viewer")
        logger.info(f"Selected dataset: {selected_dataset}")
        logger.info(f"Dataset names: {dataset_names}")

        return {
            inp["example_dataset_dropdown"]: gr.Dropdown(
                choices=dataset_names, value=selected_dataset, interactive=True
            ),
            inp["example_annotator_row_dropdown"]: gr.Dropdown(
                choices=annotator_choices,
                value=annotator_row,
                interactive=True,
            ),
            inp["example_annotator_col_dropdown"]: gr.Dropdown(
                choices=annotator_choices,
                value=annotator_col,
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
        annotator_row = data[inp["example_annotator_row_dropdown"]]
        annotator_col = data[inp["example_annotator_col_dropdown"]]
        subset_filter = data[inp["example_subset_dropdown"]]
        example_index = int(data[inp["example_index_slider"]])
        votes_dicts = data.get(state["votes_dicts"], {})
        dataset_names = data[inp["active_datasets_dropdown"]]

        dataset_col_name = _get_dataset_col_name(
            selected_dataset, dataset_names, votes_dicts
        )

        if dataset_col_name not in votes_dicts:
            logger.warning(
                "Example viewer: selected_dataset not in votes_dicts"
                f"selected_dataset: {dataset_col_name}, "
                f"votes_dicts: {list(votes_dicts.keys())}"
            )
            return _empty_example_display(out)

        votes_dict = votes_dicts[dataset_col_name]
        df = votes_dict.get("df")
        annotator_metadata = votes_dict.get("annotator_metadata", {})

        if df is None or len(df) == 0:
            logger.warning("Example viewer: df is None or empty")
            return _empty_example_display(out)

        # Filter dataframe based on subset selection
        filtered_df = _filter_dataframe(
            df, votes_dict, annotator_row, annotator_col, subset_filter
        )

        if len(filtered_df) == 0 or example_index >= len(filtered_df):
            logger.warning(
                "Example viewer: filtered_df is empty or example_index is out of range"
                f"filtered_df: {filtered_df}"
                f"example_index: {example_index}"
            )
            return _empty_example_display(out)

        # Get the selected row
        row = filtered_df.iloc[example_index]

        # Extract basic information
        comparison_id = str(row.get("comparison_id", "N/A"))
        prompt = str(row.get("prompt", "N/A"))

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
        annotator_row_result = "N/A"
        annotator_col_result = "N/A"

        if annotator_row and annotator_col:
            # Find annotator column names from visible names
            row_col_name = _get_annotator_col_from_visible_name(
                annotator_metadata, annotator_row
            )
            col_col_name = _get_annotator_col_from_visible_name(
                annotator_metadata, annotator_col
            )

            if row_col_name and row_col_name in row:
                annotator_row_result = str(row[row_col_name])

            if col_col_name and col_col_name in row:
                annotator_col_result = str(row[col_col_name])

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
            out["example_prompt"]: prompt,
            out["example_response_a"]: text_a,
            out["example_response_b"]: text_b,
            out["example_annotator_row_result"]: annotator_row_result,
            out["example_annotator_col_result"]: annotator_col_result,
            out["example_metadata"]: metadata,
            out["example_no_examples_message"]: gr.Markdown(visible=False),
            out["example_details_group"]: gr.Group(visible=True),
        }

    return {
        "update_example_viewer_options": update_example_viewer_options,
        "display_example": display_example,
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
    annotator_row: str,
    annotator_col: str,
    subset_filter: str,
) -> pd.DataFrame:
    """Filter dataframe based on subset selection."""
    if subset_filter == "all":
        return df

    if not annotator_row or not annotator_col:
        return df

    annotator_metadata = votes_dict.get("annotator_metadata", {})
    row_col_name = _get_annotator_col_from_visible_name(
        annotator_metadata, annotator_row
    )
    col_col_name = _get_annotator_col_from_visible_name(
        annotator_metadata, annotator_col
    )

    if not row_col_name or not col_col_name:
        return df

    if row_col_name not in df.columns or col_col_name not in df.columns:
        return df

    # Create filter masks
    if subset_filter == "agree":
        mask = df[row_col_name] == df[col_col_name]
    elif subset_filter == "disagree":
        mask = (
            (df[row_col_name] != df[col_col_name])
            & (df[row_col_name].isin(["text_a", "text_b"]))
            & (df[col_col_name].isin(["text_a", "text_b"]))
        )
    elif subset_filter == "only annotator row does not apply":
        mask = (~df[row_col_name].isin(["text_a", "text_b"])) & (
            df[col_col_name].isin(["text_a", "text_b"])
        )
    elif subset_filter == "only annotator column does not apply":
        mask = (df[row_col_name].isin(["text_a", "text_b"])) & (
            ~df[col_col_name].isin(["text_a", "text_b"])
        )
    elif subset_filter == "neither apply":
        mask = (~df[row_col_name].isin(["text_a", "text_b"])) & (
            ~df[col_col_name].isin(["text_a", "text_b"])
        )
    else:
        return df

    return df[mask].reset_index(drop=True)


def _empty_example_display(out: dict) -> dict:
    """Return empty example display values."""
    gr.Warning("No examples found")
    return {
        out["example_comparison_id"]: "",
        out["example_prompt"]: "",
        out["example_response_a"]: "",
        out["example_response_b"]: "",
        out["example_annotator_row_result"]: "",
        out["example_annotator_col_result"]: "",
        out["example_metadata"]: {},
        out["example_no_examples_message"]: gr.Markdown(visible=True),
        out["example_details_group"]: gr.Group(visible=False),
    }

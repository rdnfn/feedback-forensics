"""Callbacks for the example viewer functionality."""

import pandas as pd
import gradio as gr


def generate(inp: dict, state: dict, out: dict) -> dict:
    """Generate callbacks for the example viewer."""

    def _generate_non_functional_slider():
        return gr.Slider(value=0, interactive=False)

    def update_example_viewer_options(data):
        """Update the example viewer dropdown options based on loaded data."""
        votes_dicts = data.get(state["votes_dicts"], {})

        if not votes_dicts:
            return {
                inp["example_dataset_dropdown"]: gr.Dropdown(choices=[], value=None),
                inp["example_annotator_row_dropdown"]: gr.Dropdown(
                    choices=[], value=None
                ),
                inp["example_annotator_col_dropdown"]: gr.Dropdown(
                    choices=[], value=None
                ),
                inp["example_index_slider"]: _generate_non_functional_slider(),
            }

        # Get dataset names
        dataset_names = list(votes_dicts.keys())

        # Get first dataset to initialize annotator options
        first_dataset = dataset_names[0] if dataset_names else None
        annotator_choices = []
        annotator_visible_names = []

        if first_dataset:
            first_votes_dict = votes_dicts[first_dataset]
            annotator_metadata = first_votes_dict.get("annotator_metadata", {})

            # Get annotator choices (visible names)
            annotator_visible_names = [
                metadata["annotator_visible_name"]
                for metadata in annotator_metadata.values()
            ]
            annotator_choices = annotator_visible_names

        # Get number of examples for slider
        max_examples = 0
        if first_dataset:
            df = votes_dicts[first_dataset].get("df")
            if df is not None:
                max_examples = max(0, len(df) - 1)

        return {
            inp["example_dataset_dropdown"]: gr.Dropdown(
                choices=dataset_names, value=first_dataset, interactive=True
            ),
            inp["example_annotator_row_dropdown"]: gr.Dropdown(
                choices=annotator_choices,
                value=annotator_choices[0] if annotator_choices else None,
                interactive=True,
            ),
            inp["example_annotator_col_dropdown"]: gr.Dropdown(
                choices=annotator_choices,
                value=(
                    annotator_choices[1]
                    if len(annotator_choices) > 1
                    else annotator_choices[0] if annotator_choices else None
                ),
                interactive=True,
            ),
            inp["example_index_slider"]: gr.Slider(
                minimum=0, maximum=max_examples, value=0, interactive=True
            ),
        }

    def update_example_viewer_annotators(data):
        """Update annotator dropdowns when dataset changes."""
        selected_dataset = data[inp["example_dataset_dropdown"]]
        votes_dicts = data.get(state["votes_dicts"], {})

        if not selected_dataset or selected_dataset not in votes_dicts:
            return {
                inp["example_annotator_row_dropdown"]: gr.Dropdown(
                    choices=[], value=None
                ),
                inp["example_annotator_col_dropdown"]: gr.Dropdown(
                    choices=[], value=None
                ),
                inp["example_index_slider"]: _generate_non_functional_slider(),
            }

        votes_dict = votes_dicts[selected_dataset]
        annotator_metadata = votes_dict.get("annotator_metadata", {})

        # Get annotator choices (visible names)
        annotator_choices = [
            metadata["annotator_visible_name"]
            for metadata in annotator_metadata.values()
        ]

        # Get number of examples for slider
        df = votes_dict.get("df")
        max_examples = max(0, len(df) - 1) if df is not None else 0

        return {
            inp["example_annotator_row_dropdown"]: gr.Dropdown(
                choices=annotator_choices,
                value=annotator_choices[0] if annotator_choices else None,
                interactive=True,
            ),
            inp["example_annotator_col_dropdown"]: gr.Dropdown(
                choices=annotator_choices,
                value=(
                    annotator_choices[1]
                    if len(annotator_choices) > 1
                    else annotator_choices[0] if annotator_choices else None
                ),
                interactive=True,
            ),
            inp["example_index_slider"]: gr.Slider(
                minimum=0, maximum=max_examples, value=0, interactive=True
            ),
        }

    def update_example_viewer_slider(data):
        """Update the example index slider when subset filter changes."""
        selected_dataset = data[inp["example_dataset_dropdown"]]
        annotator_row = data[inp["example_annotator_row_dropdown"]]
        annotator_col = data[inp["example_annotator_col_dropdown"]]
        subset_filter = data[inp["example_subset_dropdown"]]
        votes_dicts = data.get(state["votes_dicts"], {})

        if not selected_dataset or selected_dataset not in votes_dicts:
            return {inp["example_index_slider"]: _generate_non_functional_slider()}

        votes_dict = votes_dicts[selected_dataset]
        df = votes_dict.get("df")

        if df is None:
            return {inp["example_index_slider"]: _generate_non_functional_slider()}

        # Filter dataframe based on subset selection
        filtered_df = _filter_dataframe(
            df, votes_dict, annotator_row, annotator_col, subset_filter
        )
        max_examples = max(0, len(filtered_df) - 1)

        return {
            inp["example_index_slider"]: gr.Slider(
                minimum=0, maximum=max_examples, value=0, interactive=True
            )
        }

    def display_example(data):
        """Display the selected example details."""
        selected_dataset = data[inp["example_dataset_dropdown"]]
        annotator_row = data[inp["example_annotator_row_dropdown"]]
        annotator_col = data[inp["example_annotator_col_dropdown"]]
        subset_filter = data[inp["example_subset_dropdown"]]
        example_index = int(data[inp["example_index_slider"]])
        votes_dicts = data.get(state["votes_dicts"], {})

        if not selected_dataset or selected_dataset not in votes_dicts:
            return _empty_example_display(out)

        votes_dict = votes_dicts[selected_dataset]
        df = votes_dict.get("df")
        annotator_metadata = votes_dict.get("annotator_metadata", {})

        if df is None or len(df) == 0:
            return _empty_example_display(out)

        # Filter dataframe based on subset selection
        filtered_df = _filter_dataframe(
            df, votes_dict, annotator_row, annotator_col, subset_filter
        )

        if len(filtered_df) == 0 or example_index >= len(filtered_df):
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
        "update_example_viewer_annotators": update_example_viewer_annotators,
        "update_example_viewer_slider": update_example_viewer_slider,
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

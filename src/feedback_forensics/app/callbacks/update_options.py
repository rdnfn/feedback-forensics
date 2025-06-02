"""Callbacks to update config options in the app.

(e.g. split column, annotator columns, etc.)
"""

import pathlib
import gradio as gr
from loguru import logger

from feedback_forensics.data.loader import add_virtual_annotators, get_votes_dict
from feedback_forensics.data.dataset_utils import (
    get_annotators_by_type,
    get_available_models,
)
from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    DEFAULT_ANNOTATOR_VISIBLE_NAME,
    MODEL_IDENTITY_ANNOTATOR_TYPE,
    PRINCIPLE_ANNOTATOR_TYPE,
    PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS,
    PREFIX_MODEL_IDENTITY_ANNOTATORS,
)


def generate(inp: dict, state: dict, out: dict, utils_callbacks: dict) -> dict:
    """Generate callbacks for the ICAI app."""

    get_columns_in_dataset = utils_callbacks["get_columns_in_dataset"]
    get_avail_col_values = utils_callbacks["get_avail_col_values"]

    def _get_default_annotator_cols_config(data) -> str:
        """Get the default annotator cols config.

        This sets the annotator columns to the default, and the rows to all principle annotators
        """
        datasets = data[inp["active_datasets_dropdown"]]

        # Normalize datasets to always be a list for processing
        if not isinstance(datasets, list):
            datasets = [datasets] if datasets is not None else []

        # if not dataset is selected, abort
        # (only possible in multidataset selection mode)
        if len(datasets) == 0:
            # need to return something to avoid errors
            # thus returning a dict with no changes/effect
            return {
                inp["annotator_cols_dropdown"]: gr.Dropdown(),
            }

        # Load the full dataset (needed to extract annotator names)
        # which may take a few seconds. Caching ensures this cost is paid only once.
        dataset_config = data[state["avail_datasets"]][datasets[0]]
        results_dir = pathlib.Path(dataset_config.path)
        base_votes_dict = get_votes_dict(results_dir, cache=data[state["cache"]])
        votes_dict = add_virtual_annotators(
            base_votes_dict,
            cache=data[state["cache"]],
            dataset_cache_key=results_dir,
            reference_models=data[inp["reference_models_dropdown"]],
            target_models=[],
        )

        annotator_types = get_annotators_by_type(votes_dict)
        all_annotator_names = []
        model_annotator_names = annotator_types[MODEL_IDENTITY_ANNOTATOR_TYPE][
            "visible_names"
        ]
        model_annotator_names = [
            name.replace(PREFIX_MODEL_IDENTITY_ANNOTATORS, "")
            for name in model_annotator_names
        ]
        for variant, annotators in annotator_types.items():
            all_annotator_names.extend(annotators["visible_names"])

        regular_annotator_names = [
            name
            for name in all_annotator_names
            if PREFIX_MODEL_IDENTITY_ANNOTATORS not in name
            and PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS not in name
        ]

        return {
            inp["annotator_cols_dropdown"]: gr.Dropdown(
                choices=sorted(all_annotator_names),
                value=[DEFAULT_ANNOTATOR_VISIBLE_NAME],
                interactive=True,
            ),
            inp["annotator_rows_dropdown"]: gr.Dropdown(
                choices=sorted(all_annotator_names),
                value=annotator_types[PRINCIPLE_ANNOTATOR_TYPE]["visible_names"],
                interactive=True,
            ),
            inp["reference_models_dropdown"]: gr.Dropdown(
                choices=sorted(get_available_models(base_votes_dict["df"])),
                value=[],
                interactive=True,
            ),
            inp["models_to_compare_dropdown"]: gr.Dropdown(
                choices=sorted(model_annotator_names),
                value=[],
                interactive=True,
            ),
            inp["annotations_to_compare_dropdown"]: gr.Dropdown(
                choices=sorted(regular_annotator_names),
                value=[DEFAULT_ANNOTATOR_VISIBLE_NAME],
                interactive=True,
            ),
        }

    def set_advanced_settings_from_model_analysis_tab(data):
        """Set the advanced settings from the model analysis tab."""
        model_annotator_names = data[inp["models_to_compare_dropdown"]]
        model_annotator_names = [
            PREFIX_MODEL_IDENTITY_ANNOTATORS + name for name in model_annotator_names
        ]
        return {
            inp["annotator_cols_dropdown"]: gr.Dropdown(
                value=model_annotator_names,
            ),
            # clear annotations to compare dropdown
            inp["annotations_to_compare_dropdown"]: gr.Dropdown(
                value=[],
            ),
        }

    def set_advanced_settings_from_annotation_analysis_tab(data):
        """Set the advanced settings from the annotation analysis tab."""
        annotation_annotator_names = data[inp["annotations_to_compare_dropdown"]]
        return {
            inp["annotator_cols_dropdown"]: gr.Dropdown(
                value=annotation_annotator_names,
            ),
            # clear models to compare dropdown
            inp["models_to_compare_dropdown"]: gr.Dropdown(
                value=[],
            ),
        }

    def set_model_analysis_from_advanced_settings(data):
        """Set the model analysis settings from the advanced settings."""
        model_annotator_names = data[inp["annotator_cols_dropdown"]]
        model_annotator_names = [
            name
            for name in model_annotator_names
            if PREFIX_MODEL_IDENTITY_ANNOTATORS in name
        ]
        model_names = [
            name.replace(PREFIX_MODEL_IDENTITY_ANNOTATORS, "")
            for name in model_annotator_names
        ]
        return {
            inp["models_to_compare_dropdown"]: gr.Dropdown(
                value=model_names,
            ),
        }

    def set_annotation_analysis_from_advanced_settings(data):
        """Set the annotation analysis settings from the advanced settings."""
        annotation_annotator_names = data[inp["annotator_cols_dropdown"]]
        regular_annotator_names = [
            name
            for name in annotation_annotator_names
            if PREFIX_MODEL_IDENTITY_ANNOTATORS not in name
            and PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS not in name
        ]
        return {
            inp["annotations_to_compare_dropdown"]: gr.Dropdown(
                value=regular_annotator_names,
            ),
        }

    def update_config_on_dataset_change(data: dict):
        """Update config on dataset change.

        Primarily affects the config blocks that are only relevant
        for single dataset analysis. Also resets the annotator cols
        config if dataset is changed.
        """

        datasets = data[inp["active_datasets_dropdown"]]
        return_val = {}

        # Normalize datasets to always be a list for processing
        if not isinstance(datasets, list):
            datasets = [datasets] if datasets is not None else []

        if len(datasets) == 1:
            single_dataset_menus_active = True
        else:
            single_dataset_menus_active = False

        if not single_dataset_menus_active:
            return_val.update(
                {
                    # make split col dropdowns inactive
                    inp["split_col_dropdown"]: gr.Dropdown(
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    ),
                    inp["split_col_selected_vals_dropdown"]: gr.Dropdown(
                        choices=[],
                        value=None,
                        interactive=False,
                    ),
                }
            )
        else:
            split_col = data[inp["split_col_dropdown"]]
            avail_cols = get_columns_in_dataset(datasets[0], data)

            if split_col not in avail_cols:
                split_col = NONE_SELECTED_VALUE

            tuple_avail_cols = [(col, col) for col in avail_cols]

            return_val.update(
                {
                    inp["split_col_dropdown"]: gr.Dropdown(
                        choices=[
                            (
                                "(No grouping applied, click to select column)",
                                NONE_SELECTED_VALUE,
                            )
                        ]
                        + tuple_avail_cols,
                        value=split_col,
                        interactive=True,
                        # visible=is_in_advanced_settings,
                    ),
                    inp["split_col_selected_vals_dropdown"]: gr.Dropdown(
                        choices=[],
                        value=[],
                        interactive=False,
                        # visible=False,
                    ),
                }
            )

        # reset the annotator cols config if dataset is changed
        return_val.update(_get_default_annotator_cols_config(data))

        return return_val

    def update_col_split_value_dropdown(data: dict):
        """Update column split value dropdown."""
        split_col = data[inp["split_col_dropdown"]]

        if split_col != NONE_SELECTED_VALUE:
            avail_values = get_avail_col_values(split_col, data)
            return {
                inp["split_col_selected_vals_dropdown"]: gr.Dropdown(
                    choices=avail_values,
                    value=[val[1] for val in avail_values[: min(len(avail_values), 3)]],
                    multiselect=True,
                    interactive=True,
                    # visible=True,
                ),
                **_get_default_annotator_cols_config(data),
            }
        else:
            return {
                inp["split_col_selected_vals_dropdown"]: gr.Dropdown(
                    choices=[],
                    value=[],
                    interactive=False,
                    # visible=False,
                ),
                **_get_default_annotator_cols_config(data),
            }

    def update_analysis_type_from_radio(data):
        """Update the analysis type from the radio button."""
        analysis_type = data[inp["analysis_type_radio"]]

        # config blocks that may be affected by analysis type
        # (excludes the direct table configs, e.g. metric, sort by, sort order)
        all_config_blocks = [
            inp["models_to_compare_dropdown"],
            inp["annotations_to_compare_dropdown"],
            inp["reference_models_dropdown"],
            inp["annotator_cols_dropdown"],
            inp["annotator_rows_dropdown"],
            inp["split_col_dropdown"],
            inp["split_col_selected_vals_dropdown"],
            inp["multi_dataset_warning_md"],
            inp["enable_multiple_datasets_checkbox"],
            inp["enable_dataviewer_checkbox"],
        ]

        if analysis_type == "model_analysis":
            shown_blocks = [
                inp["models_to_compare_dropdown"],
                inp["reference_models_dropdown"],
            ]
        elif analysis_type == "annotation_analysis":
            shown_blocks = [inp["annotations_to_compare_dropdown"]]
        elif analysis_type == "advanced_settings":
            shown_blocks = all_config_blocks

        return {
            block: (
                gr.Dropdown(visible=block in shown_blocks)
                if not isinstance(block, gr.Checkbox)
                else gr.Checkbox(visible=block in shown_blocks)
            )
            for block in all_config_blocks
        }

    def update_dataset_dropdown_multiselect(data):
        """Update the dataset dropdown multiselect property based on the checkbox."""
        enable_multiple = data[inp["enable_multiple_datasets_checkbox"]]
        current_value = data[inp["active_datasets_dropdown"]]

        # If switching from multiple to single selection and multiple datasets are selected,
        # keep only the first one
        if (
            not enable_multiple
            and isinstance(current_value, list)
            and len(current_value) > 1
        ):
            current_value = [current_value[0]]

        # Ensure current_value is a list when multiselect is True, single value when False
        if enable_multiple and not isinstance(current_value, list):
            current_value = [current_value] if current_value else []
        elif not enable_multiple and isinstance(current_value, list):
            current_value = current_value[0] if current_value else None

        return {
            inp["active_datasets_dropdown"]: gr.Dropdown(
                multiselect=enable_multiple, value=current_value
            )
        }

    def toggle_results_view(data):
        """Update the visibility of results view components based on radio selection."""
        results_view = data[inp["results_view_radio"]]

        return {
            inp["numerical_results_col"]: gr.Column(
                visible=results_view == "numerical_results"
            ),
            inp["example_view_col"]: gr.Column(
                visible=results_view == "example_viewer"
            ),
        }

    def toggle_dataviewer_availability(data):
        """Toggle the availability of the dataviewer based on the checkbox."""

        enable_dataviewer = data[inp["enable_dataviewer_checkbox"]]

        if enable_dataviewer:
            results_view = data[inp["results_view_radio"]]
            results_view_changable = True
        else:
            results_view = "numerical_results"
            results_view_changable = False

        data[inp["results_view_radio"]] = results_view

        return {
            inp["results_view_radio"]: gr.Radio(
                value=results_view,
                visible=results_view_changable,
            ),
            **toggle_results_view(data),
        }

    return {
        "update_config_on_dataset_change": update_config_on_dataset_change,
        "update_col_split_value_dropdown": update_col_split_value_dropdown,
        "set_advanced_settings_from_model_analysis_tab": set_advanced_settings_from_model_analysis_tab,
        "set_model_analysis_from_advanced_settings": set_model_analysis_from_advanced_settings,
        "set_advanced_settings_from_annotation_analysis_tab": set_advanced_settings_from_annotation_analysis_tab,
        "set_annotation_analysis_from_advanced_settings": set_annotation_analysis_from_advanced_settings,
        "update_analysis_type_from_radio": update_analysis_type_from_radio,
        "update_dataset_dropdown_multiselect": update_dataset_dropdown_multiselect,
        "toggle_results_view": toggle_results_view,
        "toggle_dataviewer_availability": toggle_dataviewer_availability,
    }

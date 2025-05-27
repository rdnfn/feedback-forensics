"""Module to attach callbacks to components/blocks in the app.

This module for example enables the "analyze" button to trigger the loading of data.
"""

import gradio as gr
from loguru import logger


def attach(inp: dict, state: dict, out: dict, callbacks: dict, demo: gr.Blocks) -> None:
    """Attach callbacks to components/blocks in the app."""

    example_viewer_inputs = {
        inp["example_dataset_dropdown"],
        inp["example_annotator_1"],
        inp["example_annotator_2"],
        inp["example_index_slider"],
        inp["example_subset_dropdown"],
    }

    external_example_viewer_inputs = {
        state["votes_dicts"],
        inp["active_datasets_dropdown"],
    }

    example_viewer_outputs = {
        out["example_comparison_id"],
        out["example_prompt"],
        out["example_response_a_model"],
        out["example_response_b_model"],
        out["example_response_a"],
        out["example_response_b"],
        out["example_annotator_1_result"],
        out["example_annotator_2_result"],
        out["example_metadata"],
        out["example_message"],
        out["example_details_group"],
    }

    all_inputs = {
        inp["active_datasets_dropdown"],
        state["avail_datasets"],
        inp["analysis_type_radio"],
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["annotator_rows_dropdown"],
        inp["annotator_cols_dropdown"],
        inp["reference_models_dropdown"],
        inp["models_to_compare_dropdown"],
        inp["annotations_to_compare_dropdown"],
        inp["enable_multiple_datasets_checkbox"],
        state["app_url"],
        state["cache"],
        inp["metric_name_dropdown"],
        inp["sort_by_dropdown"],
        inp["sort_order_dropdown"],
        state["computed_annotator_metrics"],
        state["computed_overall_metrics"],
        state["default_annotator_cols"],
        state["default_annotator_rows"],
        state["votes_dicts"],
    } | example_viewer_inputs

    dataset_selection_outputs = [
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["multi_dataset_warning_md"],
        inp["annotator_rows_dropdown"],
        inp["annotator_cols_dropdown"],
        inp["models_to_compare_dropdown"],
        inp["annotations_to_compare_dropdown"],
        inp["reference_models_dropdown"],
        inp["load_btn"],
    ]

    load_data_outputs = (
        {
            inp["split_col_dropdown"],
            inp["split_col_selected_vals_dropdown"],
            inp["multi_dataset_warning_md"],
            inp["annotator_rows_dropdown"],
            inp["annotator_cols_dropdown"],
            inp["models_to_compare_dropdown"],
            inp["annotations_to_compare_dropdown"],
            inp["reference_models_dropdown"],
            out["share_link"],
            out["overall_metrics_table"],
            out["annotator_table"],
            state["cache"],
            inp["load_btn"],
            inp["sort_by_dropdown"],
            inp["sort_order_dropdown"],
            inp["metric_name_dropdown"],
            state["computed_annotator_metrics"],
            state["computed_overall_metrics"],
            state["default_annotator_cols"],
            state["default_annotator_rows"],
            state["votes_dicts"],
        }
        | example_viewer_inputs
        | example_viewer_outputs
    )

    config_blocks_inputs = {
        inp["multi_dataset_warning_md"],
        inp["models_to_compare_dropdown"],
        inp["annotations_to_compare_dropdown"],
        inp["reference_models_dropdown"],
        inp["annotator_cols_dropdown"],
        inp["annotator_rows_dropdown"],
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["enable_multiple_datasets_checkbox"],
    }

    annotation_table_outputs = [
        out["annotator_table"],
        inp["sort_by_dropdown"],
        out["share_link"],
    ]

    # reload data when load button is clicked
    inp["load_btn"].click(
        callbacks["load_data"],
        inputs=all_inputs,
        outputs=load_data_outputs,
    )

    # update single dataset menus when active dataset is changed
    # (e.g. hiding this menu if multiple datasets are selected)
    inp["active_datasets_dropdown"].input(
        callbacks["update_single_dataset_menus"],
        inputs=all_inputs,
        outputs=dataset_selection_outputs,
    )

    # update column split value dropdowns when split column is changed
    inp["split_col_dropdown"].input(
        callbacks["update_col_split_value_dropdown"],
        inputs=all_inputs,
        outputs=dataset_selection_outputs,
    )

    # Update annotator table when metric type, sort by, or sort order is changed
    inp["metric_name_dropdown"].change(
        callbacks["update_annotator_table"],
        inputs=all_inputs,
        outputs=annotation_table_outputs,
    )

    inp["sort_by_dropdown"].change(
        callbacks["update_annotator_table"],
        inputs=all_inputs,
        outputs=annotation_table_outputs,
    )

    inp["sort_order_dropdown"].change(
        callbacks["update_annotator_table"],
        inputs=all_inputs,
        outputs=annotation_table_outputs,
    )

    # update advanced settings from model analysis tab
    inp["models_to_compare_dropdown"].input(
        callbacks["set_advanced_settings_from_model_analysis_tab"],
        inputs={inp["models_to_compare_dropdown"]},
        outputs={
            inp["annotator_cols_dropdown"],
            inp["annotations_to_compare_dropdown"],
        },
        show_progress="hidden",
    )

    # update model analysis settings from advanced settings
    inp["annotator_cols_dropdown"].input(
        callbacks["set_model_analysis_from_advanced_settings"],
        inputs={inp["annotator_cols_dropdown"]},
        outputs={inp["models_to_compare_dropdown"]},
        show_progress="hidden",
    )

    # update advanced settings from annotation analysis tab
    inp["annotations_to_compare_dropdown"].input(
        callbacks["set_advanced_settings_from_annotation_analysis_tab"],
        inputs={inp["annotations_to_compare_dropdown"]},
        outputs={
            inp["annotator_cols_dropdown"],
            inp["models_to_compare_dropdown"],
        },
        show_progress="hidden",
    )

    # update annotation analysis settings from advanced settings
    inp["annotator_cols_dropdown"].input(
        callbacks["set_annotation_analysis_from_advanced_settings"],
        inputs={inp["annotator_cols_dropdown"]},
        outputs={inp["annotations_to_compare_dropdown"]},
        show_progress="hidden",
    )

    # update dataset dropdown multiselect when checkbox is toggled
    inp["enable_multiple_datasets_checkbox"].change(
        callbacks["update_dataset_dropdown_multiselect"],
        inputs={
            inp["enable_multiple_datasets_checkbox"],
            inp["active_datasets_dropdown"],
        },
        outputs={inp["active_datasets_dropdown"]},
        show_progress="hidden",
    )

    # update visible config blocks when analysis type is changed
    inp["analysis_type_radio"].change(
        callbacks["update_analysis_type_from_radio"],
        inputs={inp["analysis_type_radio"]},
        outputs=config_blocks_inputs,  # we changing their visibility
    )

    demo.load(
        callbacks["update_analysis_type_from_radio"],
        inputs={inp["analysis_type_radio"]},
        outputs=config_blocks_inputs,  # we changing their visibility
    )

    # update visible results view when results view radio is changed
    inp["results_view_radio"].change(
        callbacks["update_results_view"],
        inputs={inp["results_view_radio"]},
        outputs={
            inp["numerical_results_col"],
            inp["example_view_col"],
        },
        show_progress="hidden",
    )

    # set initial results view visibility on page load
    demo.load(
        callbacks["update_results_view"],
        inputs={inp["results_view_radio"]},
        outputs={
            inp["numerical_results_col"],
            inp["example_view_col"],
        },
        show_progress="hidden",
    )

    def test_fn(evt: gr.EventData, data):
        annotator_cols = data[inp["annotator_cols_dropdown"]]
        annotator_rows = data[inp["annotator_rows_dropdown"]]
        df_values: gr.DataFrame = data[out["annotator_table"]].values

        print(f"df_values: {df_values}")

        print(f"Event _data: {evt._data}")

        empty_return = {out["example_comparison_id"]: gr.Textbox(visible=True)}

        index = evt._data["index"]
        y_idx = index[0]
        x_idx = index[1]
        if index[1] == 0:
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
                        "found in annotator rows ({annotator_rows})"
                    )
                )
                return empty_return
            elif len(selected_annotator_row_potential_name) > 1:
                logger.warning(
                    (
                        f"Selected annotator row {selected_annotator_row_shown_name} "
                        "found multiple times in annotator rows ({annotator_rows})"
                        " Annotator names shown in table are not unique."
                    )
                )
                return empty_return
            else:
                selected_annotator_row = selected_annotator_row_potential_name[0]

        print(f"Selected annotator col: {selected_annotator_col}")
        print(f"Selected annotator row: {selected_annotator_row}")

        data[inp["example_annotator_1"]] = selected_annotator_row
        data[inp["example_annotator_2"]] = selected_annotator_col

        subset_val = "agree"
        data[inp["example_subset_dropdown"]] = subset_val

        gr.Info(
            f"Showing example datapoints where annotations by '{selected_annotator_row}' and '{selected_annotator_col}' agree."
        )

        return {
            inp["results_view_radio"]: "example_viewer",
            out["example_details_group"]: gr.Group(visible=True),
            inp["numerical_results_col"]: gr.Column(visible=False),
            inp["example_subset_dropdown"]: gr.Dropdown(
                value=subset_val, interactive=False
            ),
            **callbacks["update_example_viewer_options"](data),
        }

    example_viewer_all_components = example_viewer_inputs | example_viewer_outputs

    out["annotator_table"].select(
        test_fn,
        inputs=all_inputs | {out["annotator_table"]},
        outputs=example_viewer_all_components
        | {
            out["example_details_group"],
            inp["numerical_results_col"],
            inp["results_view_radio"],
        },
        scroll_to_output=True,
    )

    # Example viewer callbacks
    # Update example viewer options when data is loaded

    for component in example_viewer_inputs:
        component.input(
            callbacks["update_example_viewer_options"],
            inputs=example_viewer_inputs | external_example_viewer_inputs,
            outputs=example_viewer_all_components,
            show_progress="hidden",
        )

    inp["load_btn"].click(
        callbacks["update_example_viewer_options"],
        inputs=example_viewer_inputs | external_example_viewer_inputs,
        outputs=example_viewer_all_components,
    )

    # finally add callbacks that run on start of app
    demo.load(
        callbacks["load_from_query_params"],
        inputs=all_inputs,
        outputs=load_data_outputs | {inp["active_datasets_dropdown"], state["app_url"]},
        trigger_mode="always_last",
    )

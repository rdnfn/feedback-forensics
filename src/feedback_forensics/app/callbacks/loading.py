"""Callbacks to load data and populate output blocks in the app."""

import pathlib
import gradio as gr
import pandas as pd
from loguru import logger

from feedback_forensics.data.loader import add_virtual_annotators, get_votes_dict
import feedback_forensics.app.plotting
from feedback_forensics.data.dataset_utils import (
    get_annotators_by_type,
    get_available_models,
)
from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    APP_BASE_URL,
    PREFIX_MODEL_IDENTITY_ANNOTATORS,
    PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS,
)
from feedback_forensics.data.datasets import (
    get_available_datasets_names,
    get_default_dataset_names,
)

from feedback_forensics.app.url_parser import (
    get_config_from_query_params,
    get_url_with_query_params,
    get_list_member_from_url_string,
    transfer_url_str_to_nonurl_str,
    transfer_url_list_to_nonurl_list,
    parse_list_param,
)
from feedback_forensics.data.handler import (
    DatasetHandler,
    _get_annotator_df_col_names,
)

from feedback_forensics.app.metrics import DEFAULT_METRIC_NAME


def generate(
    inp: dict,
    state: dict,
    out: dict,
    utils_callbacks: dict,
    update_options_callbacks: dict,
    example_viewer_callbacks: dict,
) -> dict:
    """Generate callbacks for loading data and plots."""

    update_config_on_dataset_change = update_options_callbacks[
        "update_config_on_dataset_change"
    ]
    update_col_split_value_dropdown = update_options_callbacks[
        "update_col_split_value_dropdown"
    ]

    get_columns_in_dataset = utils_callbacks["get_columns_in_dataset"]
    get_avail_col_values = utils_callbacks["get_avail_col_values"]

    def load_data(
        data: dict,
    ) -> dict:
        """Load data with dictionary inputs instead of individual arguments."""
        datasets = data[inp["active_datasets_dropdown"]]

        # Normalize datasets to always be a list for processing
        if not isinstance(datasets, list):
            datasets = [datasets] if datasets is not None else []

        cache = data[state["cache"]]
        split_col = data[inp["split_col_dropdown"]]
        selected_vals = data[inp["split_col_selected_vals_dropdown"]]
        reference_models = data[inp["reference_models_dropdown"]]
        metric_name = data[inp["metric_name_dropdown"]]
        sort_by = data[inp["sort_by_dropdown"]]
        sort_ascending = data[inp["sort_order_dropdown"]] == "Ascending"

        if len(datasets) == 0:
            gr.Warning(
                "No datasets selected. Please select at least one dataset to run analysis on.",
            )
            return {
                out["overall_metrics_table"]: gr.Dataframe(
                    value=pd.DataFrame(), headers=["No data available"]
                ),
                out["annotator_table"]: gr.Dataframe(
                    value=pd.DataFrame(), headers=["No data available"]
                ),
                out["share_link"]: data.get(
                    state["app_url"], ""
                ),  # Return base URL or empty string
            }

        # loading data via handler
        gr.Info(f"Loading data for: {', '.join(datasets)}...", duration=3)
        dataset_handler = DatasetHandler(
            cache=cache,
            avail_datasets=data[state["avail_datasets"]],
            reference_models=reference_models,
        )
        dataset_handler.load_data_from_names(datasets)

        # set annotators rows and columns according to user input
        annotator_rows_visible_names = data[inp["annotator_rows_dropdown"]]
        dataset_handler.set_annotator_rows(annotator_rows_visible_names)
        annotator_cols_visible_names = data[inp["annotator_cols_dropdown"]]
        if len(datasets) > 1 and len(annotator_cols_visible_names) > 1:
            gr.Warning(
                (
                    "Only one annotator column (e.g. for model identity) is supported"
                    " when selecting multiple datasets. Only using first annotator column"
                    f" ({annotator_cols_visible_names[0]})."
                ),
            )
            avail_annotators_cross_datasets = (
                dataset_handler.get_available_annotator_visible_names()
            )
            annotator_cols_visible_names = annotator_cols_visible_names[:1]
            if annotator_cols_visible_names[0] not in avail_annotators_cross_datasets:
                gr.Warning(
                    (
                        f"Annotator column '{annotator_cols_visible_names[0]}' not"
                        " found in across all selected datasets. Please select a "
                        "different annotator column. Aborting analysis."
                    )
                )
                # return statement needs to have at least one output
                # thus we add this output without change
                return {
                    inp["split_col_dropdown"]: data[inp["split_col_dropdown"]],
                    out["overall_metrics_table"]: gr.Dataframe(
                        value=pd.DataFrame(), headers=["⛔️ Analysis stopped"]
                    ),
                    out["annotator_table"]: gr.Dataframe(
                        value=pd.DataFrame(), headers=["⛔️ Analysis stopped"]
                    ),
                }

        dataset_handler.set_annotator_cols(annotator_cols_visible_names)

        # checking if splitting by column is requested
        if split_col != NONE_SELECTED_VALUE and split_col is not None:
            if dataset_handler.num_cols > 1:
                raise gr.Error(
                    "Only one votes_df is supported when splitting by column"
                )

            # set values equivalent to no value to None
            if selected_vals == [] or set(selected_vals) == set(
                inp["split_col_selected_vals_dropdown"].choices
            ):
                selected_vals = None

            # split the first dataset (handler) by the selected column
            # this now is treated like multiple datasets (as in multiple columns)
            dataset_handler.split_by_col(col=split_col, selected_vals=selected_vals)

        # compute metrics
        overall_metrics = dataset_handler.get_overall_metrics()
        annotator_metrics = dataset_handler.get_annotator_metrics()

        # set up sorting of annotator metrics table
        sort_by_choices = ["Max diff"] + list(annotator_metrics.keys())
        if sort_by not in sort_by_choices and sort_by_choices:
            # might be url encoded version of sort_by
            sort_by = transfer_url_str_to_nonurl_str(sort_by, sort_by_choices)
            if sort_by is None:
                sort_by = sort_by_choices[0]

        # generate Gradio (not pandas) dataframes (shown as tables in the app)
        tables = feedback_forensics.app.plotting.generate_dataframes(
            annotator_metrics=annotator_metrics,
            overall_metrics=overall_metrics,
            metric_name=metric_name,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
        )

        data[state["votes_dicts"]] = dataset_handler.votes_dicts

        return_dict = {
            out["overall_metrics_table"]: tables["overall_metrics"],
            out["annotator_table"]: tables["annotator"],
            state["cache"]: cache,
            state["computed_overall_metrics"]: overall_metrics,
            state["computed_annotator_metrics"]: annotator_metrics,
            state[
                "default_annotator_cols"
            ]: dataset_handler.first_handler.default_annotator_cols,
            state[
                "default_annotator_rows"
            ]: dataset_handler.first_handler.default_annotator_rows,
            state["votes_dicts"]: dataset_handler.votes_dicts,
            inp["metric_name_dropdown"]: gr.Dropdown(
                value=metric_name,
                interactive=True,
            ),
            inp["sort_by_dropdown"]: gr.Dropdown(
                choices=sort_by_choices, value=sort_by
            ),
            inp["sort_order_dropdown"]: gr.Dropdown(
                value="Descending" if not sort_ascending else "Ascending"
            ),
            **example_viewer_callbacks["update_example_viewer_options"](data),
        }

        # generate share link based on updated app state data
        for key in [
            state["computed_annotator_metrics"],
            state["computed_overall_metrics"],
            state["default_annotator_cols"],
            state["default_annotator_rows"],
            state["votes_dicts"],
        ]:
            data[key] = return_dict[key]

        return_dict[out["share_link"]] = _get_url_share_link_from_app_state(data)

        return return_dict

    def load_from_query_params(data: dict, request: gr.Request):
        """Load data from query params."""
        config = get_config_from_query_params(request)

        # check if config is None (did not parse correctly)
        if config is None:
            return {
                inp["active_datasets_dropdown"]: gr.Dropdown(
                    choices=get_available_datasets_names(),
                    value=(
                        get_default_dataset_names()[0]
                        if get_default_dataset_names()
                        else None
                    ),
                    multiselect=False,
                )
            }

        # ensure that base_url is correctly set
        # (e.g. app.feedbackforensics.com or localhost:7860)
        if APP_BASE_URL is not None:
            app_url = APP_BASE_URL
        else:
            app_url = request.headers["origin"]

        return_dict = {
            state["app_url"]: app_url,
        }
        data[state["app_url"]] = app_url

        if "datasets" in config:

            # If multiple datasets are specified in URL, enable multiselect
            multiselect_enabled = len(config["datasets"]) > 1
            data[inp["active_datasets_dropdown"]] = (
                config["datasets"] if multiselect_enabled else config["datasets"][0]
            )

            return_dict[inp["active_datasets_dropdown"]] = gr.Dropdown(
                value=(
                    config["datasets"] if multiselect_enabled else config["datasets"][0]
                ),
                multiselect=multiselect_enabled,
            )

            # Also update the checkbox state if multiple datasets are loaded
            if multiselect_enabled:
                data[inp["enable_multiple_datasets_checkbox"]] = True
                return_dict[inp["enable_multiple_datasets_checkbox"]] = gr.Checkbox(
                    value=True,
                )

            # Only load the dataset when necessary (annotators or reference models are specified)
            need_to_load_dataset = (
                "annotator_rows" in config
                or "annotator_cols" in config
                or "reference_models" in config
            )
            base_votes_dict = None
            reference_models = []

            if need_to_load_dataset:
                # Load the dataset to get access to model data
                # May take seconds, but is necessary. Caching ensures we only pay this cost once.
                dataset_config = data[state["avail_datasets"]][config["datasets"][0]]
                results_dir = pathlib.Path(dataset_config.path)
                handler = DatasetHandler(
                    cache=data[state["cache"]],
                    avail_datasets=data[state["avail_datasets"]],
                    reference_models=reference_models,
                )
                handler.load_data_from_names([config["datasets"][0]])

                base_votes_dict = handler.first_handler.votes_dict
                base_votes_dict = get_votes_dict(
                    results_dir, cache=data[state["cache"]]
                )

                if "reference_models" in config:
                    available_models = get_available_models(base_votes_dict["df"])

                    # Use URL parser utility to translate URL-encoded model names to their original form
                    url_reference_models = config["reference_models"]
                    reference_models = parse_list_param(
                        url_list=url_reference_models,
                        avail_nonurl_list=available_models,
                        param_name="reference_models",
                    )

                    data[inp["reference_models_dropdown"]] = reference_models
                    return_dict[inp["reference_models_dropdown"]] = gr.Dropdown(
                        choices=sorted(available_models),
                        value=reference_models,
                        interactive=True,
                    )

                votes_dict = add_virtual_annotators(
                    base_votes_dict,
                    cache=data[state["cache"]],
                    dataset_cache_key=results_dir,
                    reference_models=reference_models,
                    target_models=[],
                )

                annotator_types = get_annotators_by_type(votes_dict)
                all_available_annotators = []
                for _, annotators in annotator_types.items():
                    all_available_annotators.extend(annotators["visible_names"])

                # If annotator rows are specified in the URL
                if "annotator_rows" in config:
                    url_annotator_rows = config["annotator_rows"]
                    annotator_rows = parse_list_param(
                        url_list=url_annotator_rows,
                        avail_nonurl_list=all_available_annotators,
                        param_name="annotator_rows",
                    )
                    data[inp["annotator_rows_dropdown"]] = annotator_rows
                    return_dict[inp["annotator_rows_dropdown"]] = gr.Dropdown(
                        choices=sorted(all_available_annotators),
                        value=annotator_rows,
                        interactive=True,
                    )

                # If annotator columns are specified in the URL
                if "annotator_cols" in config:
                    url_annotator_cols = config["annotator_cols"]
                    annotator_cols = parse_list_param(
                        url_list=url_annotator_cols,
                        avail_nonurl_list=all_available_annotators,
                        param_name="annotator_cols",
                    )
                    data[inp["annotator_cols_dropdown"]] = annotator_cols
                    return_dict[inp["annotator_cols_dropdown"]] = gr.Dropdown(
                        choices=sorted(all_available_annotators),
                        value=annotator_cols,
                        interactive=True,
                    )

                    # also update model analysis tab
                    selected_model_annotator_names = [
                        name.replace(PREFIX_MODEL_IDENTITY_ANNOTATORS, "")
                        for name in annotator_cols
                        if PREFIX_MODEL_IDENTITY_ANNOTATORS in name
                    ]
                    all_available_model_annotator_names = [
                        name.replace(PREFIX_MODEL_IDENTITY_ANNOTATORS, "")
                        for name in all_available_annotators
                        if PREFIX_MODEL_IDENTITY_ANNOTATORS in name
                    ]
                    return_dict[inp["models_to_compare_dropdown"]] = gr.Dropdown(
                        choices=sorted(all_available_model_annotator_names),
                        value=selected_model_annotator_names,
                    )

                    # also update annotation analysis tab
                    selected_annotation_annotator_names = [
                        name
                        for name in annotator_cols
                        if PREFIX_MODEL_IDENTITY_ANNOTATORS not in name
                        and PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS not in name
                    ]
                    all_available_annotation_annotator_names = [
                        name
                        for name in all_available_annotators
                        if PREFIX_MODEL_IDENTITY_ANNOTATORS not in name
                        and PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS not in name
                    ]

                    return_dict[inp["annotations_to_compare_dropdown"]] = gr.Dropdown(
                        choices=sorted(all_available_annotation_annotator_names),
                        value=selected_annotation_annotator_names,
                    )

        # Split dataset by column if specified in URL
        if "col" not in config:
            # update split col dropdowns even if no column is selected
            base_updated_config_dict = update_config_on_dataset_change(data)
            return_dict = {
                **base_updated_config_dict,
                **update_col_split_value_dropdown(data),
                **return_dict,
            }
        else:
            # parse out column and value params from url
            if "datasets" in config and len(config["datasets"]) > 1:
                gr.Warning(
                    f"URL problem: only one dataset is supported when splitting by column. Requested {len(config['datasets'])} datasets in URL ({config['datasets']}), and requested splitting by column {config['col']}.",
                    duration=15,
                )
                split_col = None
            else:
                url_split_col = config["col"]

                # adapt split col to match available columns in dataset
                avail_cols = get_columns_in_dataset(config["datasets"][0], data)
                split_col = get_list_member_from_url_string(
                    url_string=url_split_col, list_members=avail_cols
                )

                if split_col is None:
                    gr.Warning(
                        f"URL problem: column '{url_split_col}' not found in dataset '{config['datasets'][0]}' (available columns: {avail_cols}).",
                        duration=15,
                    )
                    data[inp["split_col_dropdown"]] = NONE_SELECTED_VALUE
                else:
                    data[inp["split_col_dropdown"]] = split_col

                base_updated_config_dict = update_config_on_dataset_change(data)
                return_dict = {
                    **base_updated_config_dict,
                    **update_col_split_value_dropdown(data),
                    **return_dict,
                }
                if (
                    "col_vals" in config
                    and split_col is not None
                    and split_col != NONE_SELECTED_VALUE
                ):
                    avail_values = get_avail_col_values(split_col, data)
                    init_selected_vals = config["col_vals"]
                    selected_vals = transfer_url_list_to_nonurl_list(
                        url_list=init_selected_vals,
                        nonurl_list=[val[1] for val in avail_values],
                    )
                    if len(selected_vals) != len(init_selected_vals):
                        gr.Warning(
                            f"URL problem: not all values for column {split_col} in URL ({init_selected_vals}) could be read succesfully. Requested values: {init_selected_vals}, retrieved values: {selected_vals}.",
                            duration=15,
                        )

                    data[inp["split_col_selected_vals_dropdown"]] = selected_vals
                    return_dict[inp["split_col_selected_vals_dropdown"]] = gr.Dropdown(
                        choices=avail_values,
                        value=selected_vals,
                        interactive=True,
                    )

        # Config of table (metric, sort_by, sort_order)
        if "metric" in config:
            data[inp["metric_name_dropdown"]] = config["metric"]
            return_dict[inp["metric_name_dropdown"]] = gr.Dropdown(
                value=config["metric"],
                interactive=True,
            )
        if "sort_by" in config:
            data[inp["sort_by_dropdown"]] = config["sort_by"]
            # We'll update choices after loading data
            return_dict[inp["sort_by_dropdown"]] = gr.Dropdown(
                value=config["sort_by"],
                interactive=True,
            )
        if "sort_order" in config:
            # Consistently capitalize the first letter of sort_order
            capitalized_sort_order = config["sort_order"].lower().capitalize()
            data[inp["sort_order_dropdown"]] = capitalized_sort_order
            return_dict[inp["sort_order_dropdown"]] = gr.Dropdown(
                value=capitalized_sort_order,
                interactive=True,
            )

        # Config of analysis mode
        if "analysis_mode" in config:
            # Validate that the analysis mode is one of the valid options
            valid_analysis_modes = [
                "model_analysis",
                "annotation_analysis",
                "advanced_settings",
            ]
            analysis_mode = config["analysis_mode"]
            if analysis_mode in valid_analysis_modes:
                data[inp["analysis_type_radio"]] = analysis_mode
                return_dict[inp["analysis_type_radio"]] = gr.Radio(
                    value=analysis_mode,
                    interactive=True,
                )
            else:
                gr.Warning(
                    f"URL problem: analysis mode '{analysis_mode}' is not valid. Valid options are: {valid_analysis_modes}. Using default 'annotation_analysis'.",
                    duration=15,
                )

        return_dict = {**return_dict, **load_data(data)}
        return return_dict

    def update_annotator_table(data):
        """Update the annotator table based on dropdown selections."""
        annotator_metrics = data[state["computed_annotator_metrics"]]
        overall_metrics = data[state["computed_overall_metrics"]]
        metric_name = data[inp["metric_name_dropdown"]]
        sort_by = data[inp["sort_by_dropdown"]]
        sort_ascending = data[inp["sort_order_dropdown"]] == "Ascending"

        # Generate the table with the new parameters
        tables = feedback_forensics.app.plotting.generate_dataframes(
            annotator_metrics=annotator_metrics,
            overall_metrics=overall_metrics,
            metric_name=metric_name,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
        )

        return {
            out["annotator_table"]: tables["annotator"],
            out["share_link"]: _get_url_share_link_from_app_state(data),
        }

    def _get_url_share_link_from_app_state(data):
        """Get the URL share link based on the current state of the app."""

        # Extract the current state of the app based on the data dictionary
        metric_name = data[inp["metric_name_dropdown"]]
        sort_by = data[inp["sort_by_dropdown"]]
        sort_ascending = data[inp["sort_order_dropdown"]] == "Ascending"
        default_sort_by = "Max diff"
        default_sort_ascending = False
        default_analysis_mode = "annotation_analysis"

        # Normalize datasets to always be a list and filter out None values
        datasets = data[inp["active_datasets_dropdown"]]
        if not isinstance(datasets, list):
            datasets = [datasets] if datasets is not None else []
        datasets = [d for d in datasets if d is not None]  # Filter out None values

        url_kwargs = {
            "datasets": datasets,
            "col": data[inp["split_col_dropdown"]],
            "col_vals": data[inp["split_col_selected_vals_dropdown"]],
            "base_url": data[state["app_url"]],
            "metric": None if metric_name == DEFAULT_METRIC_NAME else metric_name,
            "sort_by": None if sort_by == default_sort_by else sort_by,
            "sort_order": (
                None
                if sort_ascending == default_sort_ascending
                else "Ascending" if sort_ascending else "Descending"
            ),
            "reference_models": data[inp["reference_models_dropdown"]],
            "analysis_mode": (
                None
                if data[inp["analysis_type_radio"]] == default_analysis_mode
                else data[inp["analysis_type_radio"]]
            ),
        }

        # See if the selected annotator rows and columns
        # are different from the default annotator rows and columnså
        # Only add to URL if they are different
        annotator_rows = _get_annotator_df_col_names(
            data[inp["annotator_rows_dropdown"]],
            data[state["votes_dicts"]],
        )
        annotator_cols = _get_annotator_df_col_names(
            data[inp["annotator_cols_dropdown"]],
            data[state["votes_dicts"]],
        )
        default_annotator_rows = data[state["default_annotator_rows"]]
        default_annotator_cols = data[state["default_annotator_cols"]]
        if sorted(annotator_rows) != sorted(default_annotator_rows):
            url_kwargs["annotator_rows"] = data[inp["annotator_rows_dropdown"]]
        if sorted(annotator_cols) != sorted(default_annotator_cols):
            url_kwargs["annotator_cols"] = data[inp["annotator_cols_dropdown"]]

        return get_url_with_query_params(**url_kwargs)

    return {
        "load_data": load_data,
        "load_from_query_params": load_from_query_params,
        "update_annotator_table": update_annotator_table,
    }

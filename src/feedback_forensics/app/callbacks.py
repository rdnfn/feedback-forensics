"""Call backs to be used in the app."""

import pathlib
import copy
import gradio as gr
import pandas as pd

from loguru import logger

from feedback_forensics.app.data.loader import add_virtual_annotators, get_votes_dict
import feedback_forensics.app.plotting
from feedback_forensics.app.data.dataset_utils import (
    get_annotators_by_type,
    get_available_models,
)
from feedback_forensics.app.utils import (
    get_csv_columns,
    load_json_file,
)
from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    APP_BASE_URL,
    DEFAULT_ANNOTATOR_VISIBLE_NAME,
    MODEL_IDENTITY_ANNOTATOR_TYPE,
    PRINCIPLE_ANNOTATOR_TYPE,
    PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS,
)
from feedback_forensics.app.data.datasets import (
    get_available_datasets_names,
    get_default_dataset_names,
)

from feedback_forensics.app.url_parser import (
    get_config_from_query_params,
    get_url_with_query_params,
    get_list_member_from_url_string,
    transfer_url_list_to_nonurl_list,
)
from feedback_forensics.app.data.handler import (
    DatasetHandler,
    _get_annotator_df_col_names,
)

from feedback_forensics.app.metrics import DEFAULT_METRIC_NAME


def generate_callbacks(inp: dict, state: dict, out: dict) -> dict:
    """Generate callbacks for the ICAI app."""

    def load_data(
        data: dict,
    ) -> dict:
        """Load data with dictionary inputs instead of individual arguments."""
        datasets = data[inp["active_datasets_dropdown"]]
        cache = data[state["cache"]]
        split_col = data[inp["split_col_dropdown"]]
        selected_vals = data[inp["split_col_selected_vals_dropdown"]]
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
            }

        # loading data via handler
        gr.Info(f"Loading data for: {', '.join(datasets)}...", duration=3)
        dataset_handler = DatasetHandler(
            cache=cache,
            avail_datasets=data[state["avail_datasets"]],
        )
        dataset_handler.load_data_from_names(datasets)

        # set annotators rows and columns according to user input
        annotator_rows_visible_names = data[inp["annotator_rows_dropdown"]]
        dataset_handler.set_annotator_rows(annotator_rows_visible_names)
        annotator_cols_visible_names = data[inp["annotator_cols_dropdown"]]
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
            sort_by = sort_by_choices[0]

        # generate Gradio (not pandas) dataframes (shown as tables in the app)
        tables = feedback_forensics.app.plotting.generate_dataframes(
            annotator_metrics=annotator_metrics,
            overall_metrics=overall_metrics,
            metric_name=metric_name,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
        )

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

    def _get_columns_in_dataset(dataset_name, data) -> str:
        dataset_config = data[state["avail_datasets"]][dataset_name]
        results_dir = pathlib.Path(dataset_config.path)
        base_votes_dict = get_votes_dict(results_dir, cache=data[state["cache"]])
        avail_cols = base_votes_dict["available_metadata_keys"]

        if dataset_config.filterable_columns:
            avail_cols = [
                col for col in avail_cols if col in dataset_config.filterable_columns
            ]
        return avail_cols

    def _get_default_annotator_cols_config(data) -> str:
        """Get the default annotator cols config.

        This sets the annotator columns to the default, and the rows to all principle annotators
        """
        datasets = data[inp["active_datasets_dropdown"]]

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
        for variant, annotators in annotator_types.items():
            all_annotator_names.extend(annotators["visible_names"])
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
        }

    def update_single_dataset_menus(data: dict):
        """Update menus for single dataset analysis

        Includes splitting dataset by column and selecting annotators."""

        datasets = data[inp["active_datasets_dropdown"]]

        if len(datasets) == 1:
            menus_inactive = False
        else:
            menus_inactive = True

        if menus_inactive:
            return {
                inp["split_col_non_available_md"]: gr.Markdown(
                    visible=True,
                ),
                inp["split_col_dropdown"]: gr.Dropdown(
                    choices=[NONE_SELECTED_VALUE],
                    value=NONE_SELECTED_VALUE,
                    interactive=False,
                    visible=False,
                ),
                inp["split_col_selected_vals_dropdown"]: gr.Dropdown(
                    choices=[],
                    value=None,
                    interactive=False,
                    visible=False,
                ),
                inp["advanced_settings_accordion"]: gr.Accordion(
                    visible=False,
                ),
            }
        else:
            split_col = data[inp["split_col_dropdown"]]
            avail_cols = _get_columns_in_dataset(datasets[0], data)

            if split_col not in avail_cols:
                split_col = NONE_SELECTED_VALUE

            tuple_avail_cols = [(col, col) for col in avail_cols]

            return {
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
                    visible=True,
                ),
                inp["split_col_selected_vals_dropdown"]: gr.Dropdown(
                    choices=[],
                    value=None,
                    interactive=False,
                    visible=False,
                ),
                inp["split_col_non_available_md"]: gr.Markdown(
                    visible=False,
                ),
                inp["advanced_settings_accordion"]: gr.Accordion(
                    visible=True,
                ),
                **_get_default_annotator_cols_config(data),
            }

    def _get_avail_col_values(col_name, data):
        dataset = data[inp["active_datasets_dropdown"]][0]
        dataset_config = data[state["avail_datasets"]][dataset]
        results_dir = pathlib.Path(dataset_config.path)
        cache = data[state["cache"]]
        votes_dict = get_votes_dict(results_dir, cache=cache)
        votes_df = votes_dict["df"]
        value_counts = votes_df[col_name].value_counts()
        avail_values = [
            (count, f"{val} ({count})", str(val)) for val, count in value_counts.items()
        ]
        # sort by count descending
        avail_values = sorted(avail_values, key=lambda x: x[0], reverse=True)
        # remove count from avail_values
        avail_values = [(val[1], val[2]) for val in avail_values]
        return avail_values

    def update_col_split_value_dropdown(data: dict):
        """Update column split value dropdown."""
        split_col = data[inp["split_col_dropdown"]]

        if split_col != NONE_SELECTED_VALUE:
            avail_values = _get_avail_col_values(split_col, data)
            return {
                inp["split_col_selected_vals_dropdown"]: gr.Dropdown(
                    choices=avail_values,
                    value=[val[1] for val in avail_values[: min(len(avail_values), 3)]],
                    multiselect=True,
                    interactive=True,
                    visible=True,
                ),
                **_get_default_annotator_cols_config(data),
            }
        else:
            return {
                inp["split_col_selected_vals_dropdown"]: gr.Dropdown(
                    choices=[],
                    value=None,
                    interactive=False,
                    visible=False,
                ),
                **_get_default_annotator_cols_config(data),
            }

    def load_from_query_params(data: dict, request: gr.Request):
        """Load data from query params."""
        config = get_config_from_query_params(request)

        # check if config is None (did not parse correctly)
        if config is None:
            return {
                inp["active_datasets_dropdown"]: gr.Dropdown(
                    choices=get_available_datasets_names(),
                    value=get_default_dataset_names(),
                )
            }

        if APP_BASE_URL is not None:
            app_url = APP_BASE_URL
        else:
            app_url = request.headers["origin"]
        return_dict = {
            state["app_url"]: app_url,
        }
        data[state["app_url"]] = app_url
        annotator_return_dict = {}
        if "datasets" in config:
            data[inp["active_datasets_dropdown"]] = config["datasets"]
            return_dict[inp["active_datasets_dropdown"]] = gr.Dropdown(
                value=config["datasets"],
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
                base_votes_dict = get_votes_dict(
                    results_dir, cache=data[state["cache"]]
                )

                if "reference_models" in config:
                    available_models = get_available_models(base_votes_dict["df"])

                    # Use URL parser utility to translate URL-encoded model names to their original form
                    url_reference_models = config["reference_models"]
                    reference_models = transfer_url_list_to_nonurl_list(
                        url_list=url_reference_models,
                        nonurl_list=list(available_models),
                    )

                    logger.debug(
                        f"URL reference models: {url_reference_models} -> {reference_models}"
                    )

                    if len(reference_models) != len(url_reference_models):
                        gr.Warning(
                            f"URL problem: not all reference models in URL ({url_reference_models}) could be found in the dataset. "
                            f"Using only available models: {reference_models}.",
                            duration=15,
                        )

                    data[inp["reference_models_dropdown"]] = reference_models
                    annotator_return_dict[inp["reference_models_dropdown"]] = (
                        gr.Dropdown(
                            choices=sorted(available_models),
                            value=reference_models,
                            interactive=True,
                        )
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
                for variant, annotators in annotator_types.items():
                    all_available_annotators.extend(annotators["visible_names"])

                # If annotator rows are specified in the URL
                if "annotator_rows" in config:
                    logger.info(
                        f"Detected annotator rows in URL: {config['annotator_rows']}"
                    )
                    url_annotator_rows = config["annotator_rows"]
                    annotator_rows = transfer_url_list_to_nonurl_list(
                        url_list=url_annotator_rows,
                        nonurl_list=all_available_annotators,
                    )
                    if len(annotator_rows) != len(url_annotator_rows):
                        gr.Warning(
                            f"URL problem: not all annotator rows in URL ({url_annotator_rows}) could be read successfully. Requested rows: {url_annotator_rows}, retrieved rows: {annotator_rows}.",
                            duration=15,
                        )
                    data[inp["annotator_rows_dropdown"]] = annotator_rows
                    annotator_return_dict[inp["annotator_rows_dropdown"]] = gr.Dropdown(
                        choices=sorted(all_available_annotators),
                        value=annotator_rows,
                        interactive=True,
                    )

                # If annotator columns are specified in the URL
                if "annotator_cols" in config:
                    logger.info(
                        f"Detected annotator cols in URL: {config['annotator_cols']}"
                    )
                    url_annotator_cols = config["annotator_cols"]
                    annotator_cols = transfer_url_list_to_nonurl_list(
                        url_list=url_annotator_cols,
                        nonurl_list=all_available_annotators,
                    )
                    if len(annotator_cols) != len(url_annotator_cols):
                        gr.Warning(
                            f"URL problem: not all annotator columns in URL ({url_annotator_cols}) could be read successfully. Requested columns: {url_annotator_cols}, retrieved columns: {annotator_cols}.",
                            duration=15,
                        )
                    data[inp["annotator_cols_dropdown"]] = annotator_cols
                    annotator_return_dict[inp["annotator_cols_dropdown"]] = gr.Dropdown(
                        choices=sorted(all_available_annotators),
                        value=annotator_cols,
                        interactive=True,
                    )

        # Handle metric, sort_by, and sort_order parameters
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

        if "col" not in config:
            # update split col dropdowns even if no column is selected
            split_col_interface_dict = update_single_dataset_menus(data)
            return_dict = {
                **return_dict,
                **split_col_interface_dict,
            }
            return_dict = {
                **return_dict,
                **update_col_split_value_dropdown(data),
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
                avail_cols = _get_columns_in_dataset(config["datasets"][0], data)
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

                split_col_interface_dict = update_single_dataset_menus(data)
                return_dict = {
                    **return_dict,
                    **split_col_interface_dict,
                }
                return_dict = {
                    **return_dict,
                    **update_col_split_value_dropdown(data),
                }
                if (
                    "col_vals" in config
                    and split_col is not None
                    and split_col != NONE_SELECTED_VALUE
                ):
                    avail_values = _get_avail_col_values(split_col, data)
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
                        visible=True,
                    )
        return_dict = {**return_dict, **load_data(data)}
        return_dict = {**return_dict, **annotator_return_dict}
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

        url_kwargs = {
            "datasets": data[inp["active_datasets_dropdown"]],
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
        "update_single_dataset_menus": update_single_dataset_menus,
        "update_col_split_value_dropdown": update_col_split_value_dropdown,
        "update_annotator_table": update_annotator_table,
    }


def attach_callbacks(
    inp: dict, state: dict, out: dict, callbacks: dict, demo: gr.Blocks
) -> None:
    """Attach callbacks using dictionary inputs."""

    all_inputs = {
        inp["active_datasets_dropdown"],
        state["avail_datasets"],
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["annotator_rows_dropdown"],
        inp["annotator_cols_dropdown"],
        inp["reference_models_dropdown"],
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
    }

    dataset_selection_outputs = [
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["split_col_non_available_md"],
        inp["advanced_settings_accordion"],
        inp["annotator_rows_dropdown"],
        inp["annotator_cols_dropdown"],
        inp["reference_models_dropdown"],
        inp["load_btn"],
    ]

    load_data_outputs = [
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["split_col_non_available_md"],
        inp["advanced_settings_accordion"],
        inp["annotator_rows_dropdown"],
        inp["annotator_cols_dropdown"],
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
    ]

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

    # finally add callbacks that run on start of app
    demo.load(
        callbacks["load_from_query_params"],
        inputs=all_inputs,
        outputs=load_data_outputs
        + [inp["active_datasets_dropdown"]]
        + [state["app_url"]],
        trigger_mode="always_last",
    )

"""Call backs to be used in the app."""

import pathlib
import copy
import gradio as gr
import pandas as pd

from loguru import logger

from feedback_forensics.app.loader import get_votes_dict
import feedback_forensics.app.plotting
from feedback_forensics.app.utils import (
    get_csv_columns,
    load_json_file,
    get_value_from_json,
)
from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    APP_BASE_URL,
    DEFAULT_ANNOTATOR_NAME,
    PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS,
)
from feedback_forensics.app.datasets import (
    get_available_datasets_names,
    get_default_dataset_names,
)

from feedback_forensics.app.url_parser import (
    get_config_from_query_params,
    get_url_with_query_params,
    get_list_member_from_url_string,
    transfer_url_list_to_nonurl_list,
)

from feedback_forensics.app.metrics import DEFAULT_METRIC_NAME


def split_votes_dicts(
    votes_dicts: dict[str, dict],
    split_col: str,
    selected_vals: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split votes data by split_col.

    First assert that only one votes_df in vote_dfs, and split that votes_df into multiple, based on the unique values of split_col.

    Args:
        votes_dicts: Dictionary mapping dataset names to dicts with keys "df" and "annotator_metadata"
        split_col: Column to split on
        selected_vals: Optional list of values to filter split_col by. If None, use all values.

    Returns:
        Dictionary mapping split values to filtered DataFrames
    """
    assert len(votes_dicts) == 1, "Only one votes_df is supported for now"
    votes_dict = list(votes_dicts.values())[0]
    votes_df = votes_dict["df"]
    votes_df[split_col] = votes_df[split_col].astype(str)

    if selected_vals:
        # Filter to only selected values before grouping
        votes_df = votes_df[votes_df[split_col].isin(selected_vals)]

    grouped_df = votes_df.groupby(split_col)

    split_dicts = {}
    for name, group in grouped_df:
        split_dicts[name] = {
            "df": group,
            "annotator_metadata": votes_dict["annotator_metadata"],
            "reference_annotator_col": votes_dict["reference_annotator_col"],
            "shown_annotator_rows": votes_dict["shown_annotator_rows"],
        }

    return split_dicts


def generate_callbacks(inp: dict, state: dict, out: dict) -> dict:
    """Generate callbacks for the ICAI app."""

    def _get_annotator_df_col_names(
        annotator_visible_names: list[str], votes_dicts: dict[str, dict]
    ) -> list[str]:
        """Get the column names of the annotators in votes_df.

        Note that this fn can be used both for annotators shown in rows and columns of
        the final output plot. All annotators are columns in the original votes_df.

        This also gives warning if not all annotators are available across all datasets.
        """
        # get mappings from visible names to column names for each dataset
        visible_to_cols = {}
        for dataset_name, votes_dict in votes_dicts.items():
            metadata = votes_dict["annotator_metadata"]
            visible_to_col = {
                value["annotator_visible_name"]: col for col, value in metadata.items()
            }
            visible_to_cols[dataset_name] = visible_to_col

        updated_annotator_visible_names = copy.deepcopy(annotator_visible_names)

        for annotator_name in annotator_visible_names:
            for dataset_name, visible_to_col in visible_to_cols.items():
                if annotator_name not in visible_to_col:
                    gr.Warning(
                        f"Annotator '{annotator_name}' (visible name) not found in dataset '{dataset_name}'. Skipping this annotator."
                    )
                    # remove annotator from annotator_visible_names
                    updated_annotator_visible_names.remove(annotator_name)

        # get column names for the remaining annotators
        col_names = [
            visible_to_col[annotator_name]
            for annotator_name in updated_annotator_visible_names
        ]
        return col_names

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
        gr.Info(f"Loading data for {datasets}...", duration=3)

        votes_dicts = {}
        for dataset in datasets:
            dataset_config = data[state["avail_datasets"]][dataset]
            path = dataset_config.path
            # check results dir inside the path
            results_dir = pathlib.Path(path)
            votes_dict = get_votes_dict(results_dir, cache=cache)

            votes_dicts[dataset] = votes_dict

        default_annotator_rows = votes_dicts[datasets[0]]["shown_annotator_rows"]
        default_annotator_cols = [votes_dicts[datasets[0]]["reference_annotator_col"]]

        # parsing of potential url params
        if split_col != NONE_SELECTED_VALUE and split_col is not None:

            if len(votes_dicts) > 1:
                raise gr.Error(
                    "Only one votes_df is supported for now when splitting by column"
                )
            if (
                selected_vals is None
                or selected_vals == []
                or set(selected_vals)
                == set(inp["split_col_selected_vals_dropdown"].choices)
            ):
                votes_dicts = split_votes_dicts(votes_dicts, split_col)
            else:
                votes_dicts = split_votes_dicts(votes_dicts, split_col, selected_vals)

        annotator_cols_visible_names = data[inp["annotator_cols_dropdown"]]
        annotator_cols = _get_annotator_df_col_names(
            annotator_cols_visible_names, votes_dicts
        )
        annotator_rows_visible_names = data[inp["annotator_rows_dropdown"]]
        annotator_rows = _get_annotator_df_col_names(
            annotator_rows_visible_names, votes_dicts
        )

        # check if multiple annotator columns and datasets are selected
        if len(annotator_cols) > 1 and len(votes_dicts) > 1:
            gr.Warning(
                f"Only one votes_df is supported when selecting multiple annotator columns. "
                f"Currently {len(votes_dicts)} votes_dfs are loaded with the following annotators: "
                f"{annotator_cols_visible_names}. Only using the first annotator column ({annotator_cols[0]})."
            )
            annotator_cols = [annotator_cols[0]]
            annotator_cols_visible_names = [annotator_cols_visible_names[0]]

        # split votes_dicts into one per annotator column (only available for one dataset)
        if len(annotator_cols) >= 1:
            if len(votes_dicts) == 1:
                dataset_name = list(votes_dicts.keys())[0]
                dataset_names = [dataset_name] * len(annotator_cols)
                votes_dicts = [votes_dicts[dataset_name]] * len(annotator_cols)
            else:
                dataset_names = list(votes_dicts.keys())
                votes_dicts = list(votes_dicts.values())

            if len(annotator_cols) == 1:
                annotator_cols = annotator_cols * len(dataset_names)
                annotator_cols_visible_names = annotator_cols_visible_names * len(
                    dataset_names
                )

            votes_dicts = {
                f"{dataset_name}\n({annotator_name.replace('-', ' ')})": {
                    "df": votes_dict["df"],
                    "annotator_metadata": votes_dict["annotator_metadata"],
                    "reference_annotator_col": annotator_col,
                    "shown_annotator_rows": votes_dict["shown_annotator_rows"],
                }
                for annotator_col, annotator_name, dataset_name, votes_dict in zip(
                    annotator_cols,
                    annotator_cols_visible_names,
                    dataset_names,
                    votes_dicts,
                )
            }

        # update set of annotator rows (keys in annotator_metadata)
        if len(annotator_rows) >= 1:
            for votes_dict in votes_dicts.values():
                votes_dict["shown_annotator_rows"] = annotator_rows

        # compute metrics for each dataset
        overall_metrics = {}
        annotator_metrics = {}
        for dataset_name, votes_dict in votes_dicts.items():
            overall_metrics[dataset_name] = (
                feedback_forensics.app.metrics.get_overall_metrics(
                    votes_dict["df"],
                    ref_annotator_col=votes_dict["reference_annotator_col"],
                )
            )
            annotator_metrics[dataset_name] = (
                feedback_forensics.app.metrics.compute_metrics(votes_dict)
            )

        sort_by_choices = ["Max diff"] + list(votes_dicts.keys())
        if sort_by not in sort_by_choices and sort_by_choices:
            sort_by = sort_by_choices[0]

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
            state["default_annotator_cols"]: default_annotator_cols,
            state["default_annotator_rows"]: default_annotator_rows,
            state["votes_dicts"]: votes_dicts,
            inp["metric_name_dropdown"]: gr.Dropdown(
                value=metric_name,
                interactive=True,
            ),
            inp["sort_by_dropdown"]: gr.Dropdown(
                choices=sort_by_choices, value=sort_by
            ),
            state["computed_annotator_metrics"]: annotator_metrics,
            state["computed_overall_metrics"]: overall_metrics,
            state["default_annotator_cols"]: default_annotator_cols,
            state["default_annotator_rows"]: default_annotator_rows,
            state["votes_dicts"]: votes_dicts,
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

        # check if dataset is dir or json
        if dataset_config.path.is_dir():
            avail_cols = get_csv_columns(
                dataset_config.path / "results" / "000_train_data.csv",
            )
        elif dataset_config.path.is_file() and dataset_config.path.suffix == ".json":
            avail_cols = get_value_from_json(
                dataset_config.path,
                "metadata.available_metadata_keys_per_comparison",
            )
        else:
            raise ValueError(
                f"Dataset {dataset_name} is not a directory or a json file. Please check the dataset path."
            )

        if dataset_config.filterable_columns:
            avail_cols = [
                col for col in avail_cols if col in dataset_config.filterable_columns
            ]
        return avail_cols

    def _get_principle_annotator_names(dataset_name, data) -> str:
        """Get principle-following annotators from json without loading full dataset."""

        dataset_config = data[state["avail_datasets"]][dataset_name]
        if dataset_config.path.is_dir():
            principle_path = (
                dataset_config.path
                / "results"
                / "030_distilled_principles_per_cluster.json"
            )
            annotator_names = list(load_json_file(principle_path).values())
        elif dataset_config.path.is_file() and dataset_config.path.suffix == ".json":
            all_annotators = get_value_from_json(
                dataset_config.path,
                "annotators",
            )
            principle_annotators = [
                annotator
                for annotator in all_annotators.values()
                if annotator["type"] == "principle"
            ]
            annotator_names = [
                annotator["description"] for annotator in principle_annotators
            ]
        else:
            raise ValueError(
                f"Dataset {dataset_name} is not a directory or a json file. Please check the dataset path."
            )
        annotator_names = [
            PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS
            + name.replace("Select the response that", "").strip(" .")
            for name in annotator_names
        ]

        return sorted(annotator_names)

    def _get_datacol_annotator_names(dataset_name, data) -> str:
        """Get the annotator names from csv file without loading full dataset."""
        dataset_config = data[state["avail_datasets"]][dataset_name]
        if dataset_config.path.is_dir():
            datacol_path = dataset_config.path / "results" / "000_train_data.csv"
            datacol_annotator_names = get_csv_columns(datacol_path)
        elif dataset_config.path.is_file() and dataset_config.path.suffix == ".json":
            all_annotators = get_value_from_json(
                dataset_config.path,
                "annotators",
            )
            datacol_annotator_names = [
                annotator["name"] if "name" in annotator else annotator["description"]
                for annotator in all_annotators.values()
                if annotator["type"] != "principle"
            ]
        else:
            raise ValueError(
                f"Dataset {dataset_name} is not a directory or a json file. Please check the dataset path."
            )
        return sorted(datacol_annotator_names)

    def _get_default_annotator_cols_config(data) -> str:
        """Get the default annotator cols config.

        This sets the annotator columns to the default, and the rows to all principle annotators
        """
        datasets = data[inp["active_datasets_dropdown"]]
        avail_principle_annotator_names = _get_principle_annotator_names(
            datasets[0], data
        )
        avail_datacol_annotator_names = _get_datacol_annotator_names(datasets[0], data)
        avail_annotator_names = (
            avail_datacol_annotator_names + avail_principle_annotator_names
        )
        return {
            inp["annotator_cols_dropdown"]: gr.Dropdown(
                choices=avail_annotator_names,
                value=[DEFAULT_ANNOTATOR_NAME],
                interactive=True,
            ),
            inp["annotator_rows_dropdown"]: gr.Dropdown(
                choices=avail_annotator_names,
                value=avail_principle_annotator_names,
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
        datasets = data[inp["active_datasets_dropdown"]]

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

            # Get available annotators for the selected dataset
            if "annotator_rows" in config or "annotator_cols" in config:
                avail_principle_annotator_names = _get_principle_annotator_names(
                    config["datasets"][0], data
                )
                avail_datacol_annotator_names = _get_datacol_annotator_names(
                    config["datasets"][0], data
                )
                all_available_annotators = (
                    avail_principle_annotator_names + avail_datacol_annotator_names
                )

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
                        choices=all_available_annotators,
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
                        choices=all_available_annotators,
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
        annotator_metrics = data[state["computed_annotator_metrics"]]
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
        inp["load_btn"],
    ]

    load_data_outputs = [
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["split_col_non_available_md"],
        inp["advanced_settings_accordion"],
        inp["annotator_rows_dropdown"],
        inp["annotator_cols_dropdown"],
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

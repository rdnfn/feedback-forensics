"""Call backs to be used in the app."""

import pathlib

import gradio as gr
import pandas as pd

from feedback_forensics.app.loader import get_votes_dict
import feedback_forensics.app.plotting
from feedback_forensics.app.utils import get_csv_columns
from feedback_forensics.app.constants import NONE_SELECTED_VALUE, APP_BASE_URL
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
        }

    return split_dicts


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

        if len(datasets) == 0:
            gr.Warning(
                "No datasets selected. Please select at least one dataset to run analysis on.",
            )
            return {out["plot"]: gr.Plot()}
        gr.Info(f"Loading data for {datasets}...", duration=3)

        votes_dicts = {}
        for dataset in datasets:
            dataset_config = data[state["avail_datasets"]][dataset]
            path = dataset_config.path
            # check results dir inside the path
            results_dir = pathlib.Path(path) / "results"
            votes_dict = get_votes_dict(results_dir, cache=cache)

            votes_dicts[dataset] = votes_dict

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

        fig = feedback_forensics.app.plotting.generate_plot(
            votes_dicts=votes_dicts,
        )

        plot = gr.Plot(fig)

        return {
            out["plot"]: plot,
            state["cache"]: cache,
            out["share_link"]: get_url_with_query_params(
                datasets=datasets,
                col=data[inp["split_col_dropdown"]],
                col_vals=data[inp["split_col_selected_vals_dropdown"]],
                base_url=data[state["app_url"]],
            ),
        }

    def _get_columns_in_dataset(dataset_name, data) -> str:
        dataset_config = data[state["avail_datasets"]][dataset_name]
        avail_cols = get_csv_columns(
            dataset_config.path / "results" / "000_train_data.csv",
        )
        if dataset_config.filterable_columns:
            avail_cols = [
                col for col in avail_cols if col in dataset_config.filterable_columns
            ]
        return avail_cols

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
                inp["split_col_non_available_md"]: gr.Markdown(
                    visible=True,
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
            }

    def _get_avail_col_values(col_name, data):
        dataset = data[inp["active_datasets_dropdown"]][0]
        dataset_config = data[state["avail_datasets"]][dataset]
        results_dir = pathlib.Path(dataset_config.path) / "results"
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
            }
        else:
            return {
                inp["split_col_selected_vals_dropdown"]: gr.Dropdown(
                    choices=[],
                    value=None,
                    interactive=False,
                    visible=False,
                ),
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
        if "datasets" in config:
            data[inp["active_datasets_dropdown"]] = config["datasets"]
            return_dict[inp["active_datasets_dropdown"]] = gr.Dropdown(
                value=config["datasets"],
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
        return return_dict

    return {
        "load_data": load_data,
        "load_from_query_params": load_from_query_params,
        "update_single_dataset_menus": update_single_dataset_menus,
        "update_col_split_value_dropdown": update_col_split_value_dropdown,
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
        out["plot"],
        state["cache"],
        inp["load_btn"],
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

    # finally add callbacks that run on start of app
    demo.load(
        callbacks["load_from_query_params"],
        inputs=all_inputs,
        outputs=load_data_outputs
        + [inp["active_datasets_dropdown"]]
        + [state["app_url"]],
        trigger_mode="always_last",
    )

"""Call backs to be used in the app."""

import json
import pathlib

import gradio as gr
import pandas as pd
from loguru import logger

from feedback_forensics.app.loader import get_votes_df
import feedback_forensics.app.plotting
import feedback_forensics.app.plotting_v2
from feedback_forensics.app.utils import get_csv_columns
from feedback_forensics.app.constants import NONE_SELECTED_VALUE, APP_BASE_URL
from feedback_forensics.app.datasets import (
    get_config_from_name,
    get_dataset_from_name,
    BuiltinDataset,
    Config,
    get_available_datasets_names,
    get_default_dataset_names,
)

from feedback_forensics.app.url_parser import (
    get_config_from_query_params,
    get_url_with_query_params,
    get_list_member_from_url_string,
    transfer_url_list_to_nonurl_list,
)


def split_votes_dfs(
    votes_dfs: dict[str, pd.DataFrame],
    split_col: str,
    selected_vals: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split votes_dfs by split_col.

    First assert that only one votes_df in vote_dfs, and split that votes_df into multiple, based on the unique values of split_col.

    Args:
        votes_dfs: Dictionary mapping dataset names to DataFrames
        split_col: Column to split on
        selected_vals: Optional list of values to filter split_col by. If None, use all values.

    Returns:
        Dictionary mapping split values to filtered DataFrames
    """
    assert len(votes_dfs) == 1, "Only one votes_df is supported for now"
    votes_df = list(votes_dfs.values())[0]
    split_dfs = {}
    votes_df[split_col] = votes_df[split_col].astype(str)

    if selected_vals:
        # Filter to only selected values before grouping
        votes_df = votes_df[votes_df[split_col].isin(selected_vals)]

    grouped_df = votes_df.groupby(split_col)
    for name, group in grouped_df:
        split_dfs[name] = group

    return split_dfs


def generate_callbacks(inp: dict, state: dict, out: dict) -> dict:
    """Generate callbacks for the ICAI app."""

    def load_data(
        data: dict,
        *,
        reset_filters_if_new: bool = True,
        used_from_button: bool = False,
        filterable_columns: list[str] | None = None,
        dataset_name: str = None,
        dataset_description: str = None,
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

        votes_dfs = {}
        for dataset in datasets:
            dataset_config = data[state["avail_datasets"]][dataset]
            path = dataset_config.path
            # check results dir inside the path
            results_dir = pathlib.Path(path) / "results"
            votes_df: pd.DataFrame = get_votes_df(results_dir, cache=cache)

            votes_dfs[dataset] = votes_df

        # fig = feedback_forensics.app.plotting.generate_plot(
        #    votes_df,
        #    unfiltered_df=unfiltered_df,
        #    show_examples=show_individual_prefs,
        #    sort_examples_by_agreement=(
        #        True if pref_order == "By reconstruction success" else False
        #    ),
        #    shown_metric_names=metrics,
        #    plot_col_name=plot_col_name,
        # )

        # parsing of potential url params
        if split_col != NONE_SELECTED_VALUE and split_col is not None:

            if len(votes_dfs) > 1:
                raise gr.Error(
                    "Only one votes_df is supported for now when splitting by column"
                )
            if (
                selected_vals is None
                or selected_vals == []
                or set(selected_vals)
                == set(inp["split_col_selected_vals_dropdown"].choices)
            ):
                votes_dfs = split_votes_dfs(votes_dfs, split_col)
            else:
                votes_dfs = split_votes_dfs(votes_dfs, split_col, selected_vals)

        fig = feedback_forensics.app.plotting_v2.generate_plot(
            votes_df_dict=votes_dfs,
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

    def update_col_split_dropdowns(data: dict):
        """Update column and split value dropdowns."""

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
            }

    def _get_avail_col_values(col_name, data):
        dataset = data[inp["active_datasets_dropdown"]][0]
        dataset_config = data[state["avail_datasets"]][dataset]
        results_dir = pathlib.Path(dataset_config.path) / "results"
        cache = data[state["cache"]]
        votes_df: pd.DataFrame = get_votes_df(results_dir, cache=cache)
        votes_df = votes_df.groupby("comparison_id").first()
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

    def update_dataset_buttons(active_dataset: str) -> dict:
        """Update dataset button variants based on active dataset."""
        updates = {}
        for name, btn in inp["dataset_btns"].items():
            updates[btn] = gr.Button(
                variant="primary" if name == active_dataset else "secondary"
            )
        return updates

    def update_advanced_config_and_load_data(data: dict):
        """Update config with dictionary inputs instead of individual arguments."""
        prior_state_datapath = data[state["datapath"]]
        selected_adv_config = data[inp["simple_config_dropdown"]]
        cache = data[state["cache"]]

        # get dataset name from button clicked
        # other buttons are not in data dict
        dataset_name = None
        for button in inp["dataset_btns"].values():
            if button in data:
                dataset_name = data[button]

        if dataset_name is None:
            dataset_name = data[state["active_dataset"]]

        # load dataset specific setup
        dataset_config: BuiltinDataset = get_dataset_from_name(dataset_name)

        new_path = True if dataset_config.path != prior_state_datapath else False

        if not dataset_config.options:
            simple_config_avail = False
        else:
            simple_config_avail = True

        # load selected advanced config
        if new_path:
            if dataset_config.options:
                selected_adv_config = (
                    dataset_config.options[0].name
                    if dataset_config.options
                    else NONE_SELECTED_VALUE
                )
            else:
                selected_adv_config = NONE_SELECTED_VALUE

        adv_config: Config = get_config_from_name(
            selected_adv_config, dataset_config.options
        )

        # Update button variants
        button_updates = update_dataset_buttons(dataset_name)

        return {
            **button_updates,
            inp["simple_config_dropdown_placeholder"]: gr.Text(
                visible=not simple_config_avail
            ),
            inp["simple_config_dropdown"]: gr.Dropdown(
                choices=(
                    [config.name for config in dataset_config.options]
                    + [NONE_SELECTED_VALUE]
                    if dataset_config.options
                    else [NONE_SELECTED_VALUE]
                ),
                value=selected_adv_config,
                interactive=True,
                visible=simple_config_avail,
            ),
            state["active_dataset"]: dataset_name,  # Update active dataset state
            inp["datapath"]: dataset_config.path,
            state["datapath"]: dataset_config.path,
            state["dataset_name"]: dataset_name,
            **load_data(
                {
                    inp["datapath"]: dataset_config.path,
                    state["datapath"]: prior_state_datapath,
                    inp[
                        "show_individual_prefs_dropdown"
                    ]: adv_config.show_individual_prefs,
                    inp["pref_order_dropdown"]: adv_config.pref_order,
                    inp["plot_col_name_dropdown"]: adv_config.plot_col_name,
                    inp["plot_col_value_dropdown"]: adv_config.plot_col_values,
                    inp["filter_col_dropdown"]: adv_config.filter_col,
                    inp["filter_value_dropdown"]: adv_config.filter_value,
                    inp["filter_col_dropdown_2"]: adv_config.filter_col_2,
                    inp["filter_value_dropdown_2"]: adv_config.filter_value_2,
                    inp["metrics_dropdown"]: adv_config.metrics,
                    state["cache"]: cache,
                },
                reset_filters_if_new=False,
                used_from_button=True,
                filterable_columns=dataset_config.filterable_columns,
                dataset_name=dataset_config.name,
                dataset_description=dataset_config.description,
            ),
            inp["filter_value_dropdown"]: gr.Dropdown(
                choices=[adv_config.filter_value],
                value=adv_config.filter_value,
                interactive=True,
            ),
            inp["filter_value_dropdown_2"]: gr.Dropdown(
                choices=[adv_config.filter_value_2],
                value=adv_config.filter_value_2,
                interactive=True,
            ),
            inp["show_individual_prefs_dropdown"]: gr.Dropdown(
                value=adv_config.show_individual_prefs,
                interactive=True,
            ),
            inp["pref_order_dropdown"]: gr.Dropdown(
                value=adv_config.pref_order,
                interactive=True,
            ),
            inp["metrics_dropdown"]: gr.Dropdown(
                value=adv_config.metrics,
                interactive=True,
            ),
        }

    def set_filter_val_dropdown(data: dict):
        """Set filter values with dictionary inputs."""
        votes_df = data.pop(state["unfiltered_df"])
        column = data.popitem()[1]

        if NONE_SELECTED_VALUE in votes_df.columns:
            raise gr.Error(
                f"Column '{NONE_SELECTED_VALUE}' is in the "
                "dataframe. This is currently not "
                "supported."
            )
        if column == NONE_SELECTED_VALUE:
            return gr.Dropdown(
                choices=[NONE_SELECTED_VALUE],
                value=NONE_SELECTED_VALUE,
                interactive=True,
            )
        else:
            avail_values = votes_df[column].unique().tolist()
            return gr.Dropdown(
                choices=[NONE_SELECTED_VALUE] + avail_values,
                value=NONE_SELECTED_VALUE,
                interactive=True,
            )

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
            split_col_interface_dict = update_col_split_dropdowns(data)
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

                split_col_interface_dict = update_col_split_dropdowns(data)
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
        "set_filter_val_dropdown": set_filter_val_dropdown,
        "update_advanced_config_and_load_data": update_advanced_config_and_load_data,
        "update_col_split_dropdowns": update_col_split_dropdowns,
        "update_col_split_value_dropdown": update_col_split_value_dropdown,
    }


def create_dataset_info(
    unfiltered_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    dataset_description: str | None = None,
) -> str:
    """Create dataset info markdown string.

    Args:
        df: DataFrame containing the dataset
        dataset_name: Name of the dataset
        dataset_description: Description of the dataset

    Returns:
        str: Markdown formatted dataset info
    """
    if unfiltered_df.empty:
        return "*No dataset loaded*"

    if dataset_name is None:
        dataset_name = "N/A"
    if dataset_description is None:
        dataset_description = "N/A"
    if dataset_path is None:
        dataset_path = "N/A"

    metrics = {}

    for name, df in [("Unfiltered", unfiltered_df), ("Filtered", filtered_df)]:
        metrics[name] = {}
        metrics[name]["num_comparisons"] = df["comparison_id"].nunique()
        metrics[name]["num_principles"] = df["principle"].nunique()
        metrics[name]["num_total_votes"] = len(df)

    info = f"""
**Name**: {dataset_name}

**Path**: {dataset_path}

**Description**: {dataset_description}

**Metrics:**
- *Total pairwise comparisons*: {metrics["Unfiltered"]["num_comparisons"]:,} (shown: {metrics["Filtered"]["num_comparisons"]:,})
- *Total tested principles*: {metrics["Unfiltered"]["num_principles"]:,} (shown: {metrics["Filtered"]["num_principles"]:,})
- *Total votes (comparisons x principles)*: {metrics["Unfiltered"]["num_total_votes"]:,} (shown: {metrics["Filtered"]["num_total_votes"]:,})
    """

    return info


def attach_callbacks(
    inp: dict, state: dict, out: dict, callbacks: dict, demo: gr.Blocks
) -> None:
    """Attach callbacks using dictionary inputs."""

    all_inputs = {
        inp["active_datasets_dropdown"],
        state["avail_datasets"],
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["datapath"],
        state["datapath"],
        state["dataset_name"],
        state["active_dataset"],
        state["app_url"],
        inp["show_individual_prefs_dropdown"],
        inp["pref_order_dropdown"],
        inp["plot_col_name_dropdown"],
        inp["plot_col_value_dropdown"],
        inp["filter_col_dropdown"],
        inp["filter_value_dropdown"],
        inp["filter_col_dropdown_2"],
        inp["filter_value_dropdown_2"],
        inp["metrics_dropdown"],
        inp["simple_config_dropdown"],
        state["cache"],
    }

    dataset_selection_outputs = [
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["split_col_non_available_md"],
        inp["load_btn"],
    ]

    load_data_outputs = [
        inp["split_col_dropdown"],
        inp["split_col_selected_vals_dropdown"],
        inp["split_col_non_available_md"],
        inp["plot_col_name_dropdown"],
        inp["plot_col_value_dropdown"],
        inp["filter_col_dropdown"],
        inp["filter_col_dropdown_2"],
        out["share_link"],
        out["plot"],
        state["df"],
        state["unfiltered_df"],
        state["datapath"],
        state["active_dataset"],
        state["dataset_name"],
        state["cache"],
        inp["datapath"],
        inp["dataset_info"],
        inp["load_btn"],
    ] + list(inp["dataset_btns"].values())

    # reload data when load button is clicked or view config is changed
    inp["load_btn"].click(
        callbacks["load_data"],
        inputs=all_inputs,
        outputs=load_data_outputs,
    )

    for config_value_dropdown in [
        inp["pref_order_dropdown"],
        inp["show_individual_prefs_dropdown"],
        inp["plot_col_value_dropdown"],
        inp["filter_value_dropdown"],
        inp["filter_value_dropdown_2"],
        inp["metrics_dropdown"],
    ]:
        config_value_dropdown.input(
            callbacks["load_data"],
            inputs=all_inputs,
            outputs=load_data_outputs,
        )

    update_load_data_outputs = (
        load_data_outputs
        + [
            inp["simple_config_dropdown"],
            inp["simple_config_dropdown_placeholder"],
            inp["plot_col_value_dropdown"],
            inp["filter_value_dropdown"],
            inp["filter_value_dropdown_2"],
            inp["show_individual_prefs_dropdown"],
            inp["pref_order_dropdown"],
            inp["metrics_dropdown"],
            state["active_dataset"],  # Add active dataset state
        ]
        + list(inp["dataset_btns"].values())
    )  # Add all dataset buttons as outputs

    inp["active_datasets_dropdown"].input(
        callbacks["update_col_split_dropdowns"],
        inputs=all_inputs,
        outputs=dataset_selection_outputs,
    )
    inp["split_col_dropdown"].input(
        callbacks["update_col_split_value_dropdown"],
        inputs=all_inputs,
        outputs=dataset_selection_outputs,
    )

    # TODO: remove old dataset selection panel (including from callbacks etc.)
    for dataset_button in inp["dataset_btns"].values():
        dataset_button.click(
            callbacks["update_advanced_config_and_load_data"],
            inputs=all_inputs.union({dataset_button}),
            outputs=update_load_data_outputs,
        )

    inp["simple_config_dropdown"].input(
        callbacks["update_advanced_config_and_load_data"],
        inputs=all_inputs,
        outputs=update_load_data_outputs,
    )

    # update filter value dropdowns when
    # corresponding filter column dropdown is changed
    for dropdown, output in [
        (inp["plot_col_name_dropdown"], inp["plot_col_value_dropdown"]),
        (inp["filter_col_dropdown"], inp["filter_value_dropdown"]),
        (inp["filter_col_dropdown_2"], inp["filter_value_dropdown_2"]),
    ]:
        dropdown.input(
            callbacks["set_filter_val_dropdown"],
            inputs={state["unfiltered_df"], dropdown},
            outputs=[output],
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

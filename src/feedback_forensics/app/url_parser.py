import gradio as gr
from loguru import logger
import re

from feedback_forensics.data.datasets import (
    get_available_datasets,
    get_stringname_from_urlname,
    get_urlname_from_stringname,
)
from feedback_forensics.app.constants import NONE_SELECTED_VALUE


def get_config_from_query_params(request: gr.Request) -> dict:
    params = dict(request.query_params)
    error_code = None
    config = {}
    if "data" in params:
        dataset_url_names = load_str_list(params["data"])
        datasets = []
        available_datasets = get_available_datasets()
        for dataset_url_name in dataset_url_names:
            dataset_name = get_stringname_from_urlname(
                dataset_url_name, available_datasets
            )
            if dataset_name is None:
                logger.warning(f"Dataset {dataset_url_name} not found")
                gr.Warning(
                    f"URL Problem: Dataset requested in URL ({dataset_url_name}) not found. Please check the URL and try again.",
                    duration=15,
                )
                return None
            datasets.append(dataset_name)
        config["datasets"] = datasets
    if "col" in params:
        config["col"] = params["col"]
    if "col_vals" in params:
        config["col_vals"] = load_str_list(params["col_vals"])
    if "ann_rows" in params:
        config["annotator_rows"] = load_str_list(params["ann_rows"])
    if "ann_cols" in params:
        config["annotator_cols"] = load_str_list(params["ann_cols"])
    if "ref_models" in params:
        config["reference_models"] = load_str_list(params["ref_models"])
    if "metric" in params:
        config["metric"] = params["metric"]
    if "sort_by" in params:
        config["sort_by"] = params["sort_by"]
    if "sort_order" in params:
        config["sort_order"] = params["sort_order"]
    if "analysis_mode" in params:
        config["analysis_mode"] = params["analysis_mode"]
    return config


def load_str_list(str_list: str) -> list[str]:
    return str_list.split(",")


def get_url_with_query_params(
    datasets: list[str],
    col: str | None,
    col_vals: list[str],
    base_url: str,
    annotator_rows: list[str] = None,
    annotator_cols: list[str] = None,
    reference_models: list[str] = None,
    metric: str = None,
    sort_by: str = None,
    sort_order: str = None,
    analysis_mode: str = None,
) -> str:
    available_datasets = get_available_datasets()

    # Handle empty datasets list or filter out None values
    if not datasets:
        # Return base URL if no datasets are specified
        return base_url

    datasets_url_names = [
        get_urlname_from_stringname(dataset, available_datasets) for dataset in datasets
    ]
    # Filter out None values that might occur if dataset name is not found
    datasets_url_names = [name for name in datasets_url_names if name is not None]

    if not datasets_url_names:
        # Return base URL if no valid dataset URL names are found
        return base_url

    url = f"{base_url}?data={','.join(datasets_url_names)}"
    if col is not None and col != NONE_SELECTED_VALUE and col != "":
        url += f"&col={col}"
        if col_vals is not None and col_vals != [NONE_SELECTED_VALUE]:
            url_ready_col_vals = [make_str_url_ready(val) for val in col_vals]
            url += f"&col_vals={','.join(url_ready_col_vals)}"

    if annotator_rows is not None and len(annotator_rows) > 0:
        url_ready_annotator_rows = [make_str_url_ready(val) for val in annotator_rows]
        url += f"&ann_rows={','.join(url_ready_annotator_rows)}"

    if annotator_cols is not None and len(annotator_cols) > 0:
        url_ready_annotator_cols = [make_str_url_ready(val) for val in annotator_cols]
        url += f"&ann_cols={','.join(url_ready_annotator_cols)}"

    if reference_models is not None and len(reference_models) > 0:
        url_ready_ref_models = [make_str_url_ready(val) for val in reference_models]
        url += f"&ref_models={','.join(url_ready_ref_models)}"

    if metric is not None:
        url += f"&metric={make_str_url_ready(metric)}"

    if sort_by is not None:
        url += f"&sort_by={make_str_url_ready(sort_by)}"

    if sort_order is not None:
        url += f"&sort_order={sort_order}"

    if analysis_mode is not None:
        url += f"&analysis_mode={analysis_mode}"

    return url


def make_str_url_ready(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "", s.replace(" ", "_")).lower().strip(" _-")


def get_list_member_from_url_string(
    url_string: str, list_members: list[str]
) -> str | None:
    urlified_list_members_dict = {
        make_str_url_ready(text): text for text in list_members
    }
    return urlified_list_members_dict.get(url_string, None)


def transfer_url_str_to_nonurl_str(url_str: str, nonurl_list: list[str]) -> str:
    urlified_generic_list_dict = {
        make_str_url_ready(text): text for text in nonurl_list
    }
    return urlified_generic_list_dict.get(url_str, None)


def transfer_url_list_to_nonurl_list(
    url_list: list[str], nonurl_list: list[str]
) -> list[str]:
    urlified_generic_list_dict = {
        make_str_url_ready(text): text for text in nonurl_list
    }
    return [
        urlified_generic_list_dict[url]
        for url in url_list
        if url in urlified_generic_list_dict
    ]


def parse_list_param(
    url_list: list[str], avail_nonurl_list: list[str], param_name: str
) -> list[str]:
    """Parse a list parameter from url to nonurl list.

    Args:
        url_list: List of strings from URL parameter.
        avail_nonurl_list: List of available non-URL strings to match against.
            These will eventually be used in the code.
        param_name: Name of the parameter being parsed (for error messages).

    Returns:
        List of strings that match between url_list and avail_nonurl_list.
    """

    nonurl_list = transfer_url_list_to_nonurl_list(
        url_list=url_list,
        nonurl_list=avail_nonurl_list,
    )
    logger.debug(f"URL list param {param_name} parsed: {url_list} -> {nonurl_list}")

    if len(nonurl_list) != len(url_list):
        gr.Warning(
            f"URL problem: not all values for '{param_name}' in URL ({url_list}) could be read successfully. "
            f"Requested {param_name}: {url_list}, "
            f"retrieved {param_name}: {nonurl_list}.",
            duration=15,
        )
    return nonurl_list

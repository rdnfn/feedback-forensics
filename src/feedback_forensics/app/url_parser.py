import gradio as gr
from loguru import logger
import re

from feedback_forensics.app.datasets import (
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
    if "metric" in params:
        config["metric"] = params["metric"]
    if "sort_by" in params:
        config["sort_by"] = params["sort_by"]
    if "sort_order" in params:
        config["sort_order"] = params["sort_order"]
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
    metric: str = None,
    sort_by: str = None,
    sort_order: str = None,
) -> str:
    available_datasets = get_available_datasets()
    datasets_url_names = [
        get_urlname_from_stringname(dataset, available_datasets) for dataset in datasets
    ]
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

    if metric is not None:
        url += f"&metric={make_str_url_ready(metric)}"

    if sort_by is not None:
        url += f"&sort_by={make_str_url_ready(sort_by)}"

    if sort_order is not None:
        url += f"&sort_order={sort_order}"

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

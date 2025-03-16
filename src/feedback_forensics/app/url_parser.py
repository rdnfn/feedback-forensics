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
    return config


def load_str_list(str_list: str) -> list[str]:
    return str_list.split(",")


def get_url_with_query_params(
    datasets: list[str], col: str | None, col_vals: list[str], base_url: str
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

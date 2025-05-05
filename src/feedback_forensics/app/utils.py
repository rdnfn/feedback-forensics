# general utils for app

import json
from importlib import resources
from pathlib import Path
import gradio as gr
import pandas as pd
from loguru import logger
import time
from itertools import islice


def get_image_path(image_name: str) -> Path:
    """Get the path to an image in the assets directory."""
    with resources.path("feedback_forensics.assets", image_name) as path:
        return path


def get_gradio_image_path(image_name: str) -> str:
    """Get the path to an image in the assets directory, via gradio files api."""
    with resources.path("feedback_forensics.assets", image_name) as path:
        img = gr.Image(
            get_image_path(image_name),
            visible=False,
        )
        # access the image path via gradio files api
        return "/gradio_api/file=" + list(img.temp_files)[0]


def get_csv_columns(file_path: str | Path) -> list[str]:
    """Get the column names from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        List of column names
    """
    return sorted(pd.read_csv(file_path, nrows=0).columns.tolist())


def load_json_file(file_path: str | Path) -> dict:
    """Load a JSON file.

    Args:
        file_path: Path to the JSON file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content


def iter_to_trunc_str(iter, max_items):
    """Convert an iterable to a string with maximum item count."""
    lst = list(islice(iter, max_items + 1))
    result = ", ".join(lst[:max_items])
    if len(lst) > max_items:
        result += ", ..."
    return result

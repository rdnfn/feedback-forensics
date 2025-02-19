# general utils for app

from importlib import resources
from pathlib import Path
import gradio as gr
import pandas as pd


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

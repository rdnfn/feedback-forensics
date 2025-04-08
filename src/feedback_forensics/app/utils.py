# general utils for app

import json
from importlib import resources
from pathlib import Path
import gradio as gr
import pandas as pd
import ijson
from io import BytesIO


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


def get_value_from_json(file_path: str | Path, key_path: str) -> any:
    """Get a value from a JSON file by following a dot-notation key path without loading the entire file.

    This function uses ijson to parse the JSON file incrementally, which is memory efficient
    for large JSON files. It follows the dot-notation key path to find the desired value.
    Note: Array access is not supported.

    Args:
        file_path: Path to the JSON file
        key_path: Dot-notation key path to the desired value (e.g., "user.settings.theme")

    Returns:
        The value found at the specified key path, or None if not found
    """
    # Convert file_path to string if it's a Path object
    file_path_str = str(file_path)

    try:
        with open(file_path_str, "rb") as file:
            # Use ijson to parse the file and get the value at the specified key path
            parser = ijson.parse(file)

            for prefix, event, value in parser:
                # Check if we're in the target key path
                if prefix == key_path:
                    return value

                # Handle nested objects
                if prefix.startswith(key_path + "."):
                    if event == "start_map":
                        continue
                    elif event == "end_map":
                        continue
                    else:
                        return value

            return None
    except Exception as e:
        print(f"Error accessing JSON file: {e}")
        return None

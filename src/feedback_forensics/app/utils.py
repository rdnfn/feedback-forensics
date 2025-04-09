# general utils for app

import json
from importlib import resources
from pathlib import Path
import gradio as gr
import pandas as pd
import ijson
from loguru import logger
import time
import re


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

    Args:
        file_path: Path to the JSON file
        key_path: Dot-notation key path to the desired value (e.g., "user.settings.theme")
                  Also supports array indexing with bracket notation (e.g., "users[0].name")

    Returns:
        The value found at the specified key path, or None if not found.
        For arrays, returns a list of all values in the array.
    """
    # Convert file_path to string if it's a Path object
    file_path_str = str(file_path)

    # Check if the key_path contains array indexing with bracket notation
    array_index = None
    base_key_path = key_path
    match = re.search(r"(.*)\[(\d+)\]$", key_path)
    if match:
        base_key_path = match.group(1)
        array_index = int(match.group(2))

    try:
        with open(file_path_str, "rb") as file:
            # Try to use a direct approach for nested objects first
            # This will handle the case where we're looking for a specific object that might contain arrays
            try:
                # Simple first approach using ijson.items - works for many common cases
                # Especially for returning complete objects at a given path
                parts = base_key_path.split(".")

                if len(parts) > 0:
                    # Try to get the object directly using ijson.items
                    for item in ijson.items(file, base_key_path):
                        # If we're looking for a specific array index
                        if (
                            isinstance(item, list)
                            and array_index is not None
                            and 0 <= array_index < len(item)
                        ):
                            return item[array_index]
                        return item

                # Reset file position for the fallback approach
                file.seek(0)
            except Exception:
                # Fall back to manual parsing if the direct approach fails
                file.seek(0)

            # Fallback: Use ijson.parse to find the value at the specified key path
            parser = ijson.parse(file)

            # Track if we're inside an array
            in_array = False
            array_values = []
            current_prefix = None

            # For handling objects within arrays
            current_object = {}
            current_key = None
            in_object = False

            # For handling nested objects
            nested_objects = {}
            current_nested_path = None
            collecting_nested = False

            for prefix, event, value in parser:
                # If we found the exact key path
                if prefix == base_key_path:
                    if event == "start_array":
                        in_array = True
                        array_values = []
                        current_prefix = prefix
                    elif event == "end_array":
                        in_array = False
                        # If we're looking for a specific array index
                        if array_index is not None and 0 <= array_index < len(
                            array_values
                        ):
                            return array_values[array_index]
                        return array_values
                    elif event == "start_map":
                        in_object = True
                        current_object = {}
                        collecting_nested = True
                        current_nested_path = prefix
                        nested_objects[current_nested_path] = {}
                    elif event == "end_map":
                        in_object = False
                        if collecting_nested and current_nested_path == base_key_path:
                            collecting_nested = False
                            return nested_objects[current_nested_path]
                        elif in_array:
                            array_values.append(current_object)
                        else:
                            return current_object
                    elif event == "map_key":
                        current_key = value
                    elif event not in [
                        "start_map",
                        "end_map",
                        "start_array",
                        "end_array",
                    ]:
                        if in_object:
                            current_object[current_key] = value
                        else:
                            return value

                # Handle nested objects and arrays
                elif prefix.startswith(base_key_path + "."):
                    parts = prefix.split(".")

                    # If we're collecting a nested object
                    if (
                        collecting_nested
                        and current_nested_path
                        and prefix.startswith(current_nested_path)
                    ):
                        if (
                            len(parts) == len(current_nested_path.split(".")) + 1
                            and event == "map_key"
                        ):
                            nested_key = parts[-1]
                            nested_objects[current_nested_path][value] = {}
                        elif len(parts) == len(current_nested_path.split(".")) + 1:
                            nested_key = parts[-1]
                            if event == "start_array":
                                nested_objects[current_nested_path][nested_key] = []
                            elif event == "start_map":
                                nested_objects[current_nested_path][nested_key] = {}
                            elif event not in ["end_array", "end_map"]:
                                nested_objects[current_nested_path][nested_key] = value

                    # If we're in an array, collect values
                    if in_array and current_prefix == base_key_path:
                        if event == "start_map":
                            in_object = True
                            current_object = {}
                        elif event == "end_map":
                            in_object = False
                            array_values.append(current_object)
                        elif event == "map_key":
                            current_key = value
                        elif event not in [
                            "start_map",
                            "end_map",
                            "start_array",
                            "end_array",
                        ]:
                            if in_object:
                                current_object[current_key] = value
                            else:
                                array_values.append(value)
                    # Skip map start/end events for nested objects
                    elif event in ["start_map", "end_map", "start_array", "end_array"]:
                        continue
                    else:
                        return value

            logger.warning(
                f"Key path {key_path} not found in JSON file {file_path_str}"
            )
            return None
    except Exception as e:
        logger.error(f"Error accessing JSON file: {e}")
        return None

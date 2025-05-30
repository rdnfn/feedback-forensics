"""Core data operations for loading and saving annotated pairs datasets."""

from pathlib import Path
from typing import Dict, Union, Any

from inverse_cai.data.annotated_pairs_format import (
    load_annotated_pairs_from_file,
    save_annotated_pairs_to_file,
)


def load_ap(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load annotated pairs dataset from file.

    Args:
        file_path: Path to annotated pairs JSON file

    Returns:
        Annotated pairs data structure

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid annotated pairs format
    """
    return load_annotated_pairs_from_file(Path(file_path))


def save_ap(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save annotated pairs dataset to file.

    Args:
        data: Annotated pairs data structure
        file_path: Path where to save the JSON file

    Raises:
        ValueError: If data is not valid annotated pairs format
    """
    save_annotated_pairs_to_file(data, Path(file_path))

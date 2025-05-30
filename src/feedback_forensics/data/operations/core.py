"""Core data operations for loading and saving AnnotatedPairs datasets."""

from pathlib import Path
from typing import Dict, Union, Any

from inverse_cai.data.annotated_pairs_format import (
    load_annotated_pairs_from_file,
    save_annotated_pairs_to_file,
)


def load_ap(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load AnnotatedPairs dataset from file.

    Args:
        file_path: Path to AnnotatedPairs JSON file

    Returns:
        AnnotatedPairs data structure

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid AnnotatedPairs format
    """
    return load_annotated_pairs_from_file(Path(file_path))


def save_ap(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save AnnotatedPairs dataset to file.

    Args:
        data: AnnotatedPairs data structure
        file_path: Path where to save the JSON file

    Raises:
        ValueError: If data is not valid AnnotatedPairs format
    """
    save_annotated_pairs_to_file(data, Path(file_path))

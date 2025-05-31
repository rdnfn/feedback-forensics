"""Core data operations for loading and saving AnnotatedPairs datasets."""

from pathlib import Path
from typing import Dict, Union, Any

from inverse_cai.data.annotated_pairs_format import (
    load_annotated_pairs_from_file,
    save_annotated_pairs_to_file,
    create_annotated_pairs,
)
from inverse_cai.data.loader.standard import load


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


def csv_to_ap(csv_path: Union[str, Path], dataset_name: str) -> Dict[str, Any]:
    """Convert CSV to AnnotatedPairs format.

    Args:
        csv_path: Path to CSV file with columns text_a, text_b, preferred_text
        dataset_name: Name for the dataset

    Returns:
        AnnotatedPairs data structure

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV is missing required columns
    """
    df = load(str(csv_path))
    return create_annotated_pairs(df, dataset_name)

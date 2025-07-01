"""Module for creating information texts in app."""

from feedback_forensics.data.datasets import BuiltinDataset


def get_datasets_description(datasets: list[BuiltinDataset]) -> str:
    """Get the info text for the datasets."""

    md_string = ""

    for dataset in datasets:
        md_string += f"- **{dataset.name}**\n"
        md_string += f"  - {dataset.description}\n"
        md_string += f"  - Source: {dataset.source}\n"
        md_string += "\n"

    return md_string

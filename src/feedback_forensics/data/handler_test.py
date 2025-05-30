"""
Tests for the data handler module.
"""

import pytest
import pandas as pd
from pathlib import Path

from feedback_forensics.data.handler import (
    DatasetHandler,
    _get_annotator_df_col_names,
)


class TestDatasetHandler:
    """Test class for the DatasetHandler module."""

    def test_init_empty(self):
        """Test initialization with empty parameters."""
        handler = DatasetHandler()

        assert handler.cache is None
        assert handler.avail_datasets is None
        assert handler.col_handlers == {}

    def test_init_with_params(self):
        """Test initialization with provided parameters."""
        cache = {}
        avail_datasets = {"test": object()}

        handler = DatasetHandler(cache=cache, avail_datasets=avail_datasets)

        assert handler.cache is cache
        assert handler.avail_datasets is avail_datasets

    def test_add_data_from_name(self):
        """Test error when avail_datasets is not provided."""
        handler = DatasetHandler()

        with pytest.raises(AssertionError):
            handler.add_data_from_name("test_dataset")

    def test_load_data_from_paths(self, setup_test_data):
        """Test loading data from file paths."""
        test_dir = setup_test_data
        handler = DatasetHandler(cache={})

        # Instead of loading multiple paths, we'll add a single path with a specific name
        path_name = "test_data"
        handler.add_data_from_path(test_dir, name=path_name)

        # Verify loading worked by checking if a handler was created with our name
        col_handler = handler.get_col_handler(path_name)
        assert col_handler is not None
        assert "comparison_id" in col_handler.df.columns
        assert "text_a" in col_handler.df.columns
        assert "text_b" in col_handler.df.columns

    def test_add_data_from_path(self, setup_annotated_pairs_json):
        """Test loading data from AnnotatedPairs JSON."""
        json_file = setup_annotated_pairs_json
        handler = DatasetHandler()

        # Use a custom name for the added data
        custom_name = "test_annotated_pairs"
        handler.add_data_from_path(json_file, custom_name)

        # Verify data loaded correctly using the custom name
        col_handler = handler.get_col_handler(custom_name)
        assert col_handler is not None

        # Verify columns are present and they match the fixture data
        df = col_handler.df
        annotator_metadata = col_handler.annotator_metadata

        assert "d36860d4" in df.columns  # default annotator
        assert "2f45a6d0" in df.columns  # principle annotator
        assert "435cef52" in df.columns  # principle annotator
        assert len(annotator_metadata) == 3


def test_get_annotator_df_col_names():
    """Test getting column names of annotators from visible names."""
    annotator_visible_names = ["Annotator 1", "Annotator 2", "Annotator 3"]
    votes_dicts = {
        "dataset1": {
            "annotator_metadata": {
                "col1": {"annotator_visible_name": "Annotator 1"},
                "col2": {"annotator_visible_name": "Annotator 2"},
                "col3": {"annotator_visible_name": "Annotator 3"},
            }
        },
        "dataset2": {
            "annotator_metadata": {
                "col1": {"annotator_visible_name": "Annotator 1"},
                "col2": {"annotator_visible_name": "Annotator 2"},
                # Annotator 3 missing from dataset2
            }
        },
    }

    result = _get_annotator_df_col_names(annotator_visible_names, votes_dicts)

    # Should only include annotators present in all datasets
    assert len(result) == 2
    assert "col1" in result
    assert "col2" in result
    assert "col3" not in result

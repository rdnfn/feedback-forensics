"""
Tests for the data handler module.
"""

import pytest
import pandas as pd
from pathlib import Path

from feedback_forensics.app.data.handler import (
    DatasetHandler,
    _get_annotator_df_col_names,
)


class TestDatasetHandler:
    """Test class for the DatasetHandler module."""

    def test_init(self):
        """Test initialization of DatasetHandler."""
        # Test with empty parameters
        handler = DatasetHandler()
        assert handler.cache is None
        assert handler.avail_datasets is None
        assert handler.votes_dict is None
        assert handler.data_path is None

        # Test with provided parameters
        cache = {}
        avail_datasets = {"test": object()}
        handler = DatasetHandler(cache=cache, avail_datasets=avail_datasets)
        assert handler.cache is cache
        assert handler.avail_datasets is avail_datasets

    def test_get_data_path(self):
        """Test retrieving data path from available datasets."""
        # Setup mock avail_datasets
        mock_dataset = type("obj", (object,), {"path": "test/path"})
        avail_datasets = {"test_dataset": mock_dataset}

        handler = DatasetHandler(avail_datasets=avail_datasets)
        path = handler._get_data_path("test_dataset")
        assert path == "test/path"

        # Test with missing dataset
        with pytest.raises(ValueError):
            handler._get_data_path("nonexistent_dataset")

    def test_load_data_from_name(self):
        """Test loading data from dataset name."""
        # Setup mock dataset config
        mock_dataset = type("obj", (object,), {"path": "test/path"})
        avail_datasets = {"test_dataset": mock_dataset}

        # Test with missing avail_datasets
        handler = DatasetHandler()
        with pytest.raises(AssertionError):
            handler.load_data_from_name("test_dataset")

    def test_load_data_from_path(self, setup_test_data):
        """Test loading data from a file path."""
        # Use the test data setup fixture
        test_dir = setup_test_data

        # Create a handler with a cache
        cache = {}
        handler = DatasetHandler(cache=cache)

        # Load data
        handler.load_data_from_path(test_dir)

        # Verify data was loaded
        assert handler.votes_dict is not None
        assert "df" in handler.votes_dict

        # Verify the DataFrame has expected columns
        df = handler.votes_dict["df"]
        assert "comparison_id" in df.columns
        assert "text_a" in df.columns
        assert "text_b" in df.columns

    def test_load_data_from_path_annotated_pairs(self, setup_annotated_pairs_json):
        """Test loading data from annotated pairs JSON."""
        # Use the annotated pairs JSON fixture
        json_file = setup_annotated_pairs_json

        # Create a handler
        handler = DatasetHandler()

        # Load data
        handler.load_data_from_path(json_file)

        # Verify data was loaded
        assert handler.votes_dict is not None
        assert "df" in handler.votes_dict
        assert "annotator_metadata" in handler.votes_dict

        # Verify the DataFrame has expected columns and data
        df = handler.votes_dict["df"]
        assert "d36860d4" in df.columns  # default annotator
        assert "2f45a6d0" in df.columns  # principle annotator
        assert "435cef52" in df.columns  # principle annotator


class TestHelperFunctions:
    """Test class for helper functions in the handler module."""

    def test_get_annotator_df_col_names(self):
        """Test getting column names of annotators from visible names."""
        # Setup test data
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

        # Test function
        result = _get_annotator_df_col_names(annotator_visible_names, votes_dicts)

        # Annotator 3 should be removed as it's not in all datasets
        assert len(result) == 2
        assert "col1" in result
        assert "col2" in result
        assert "col3" not in result

"""
Tests for the loader module.
"""

import os
import json
import tempfile
import shutil
import pandas as pd
import pytest
from pathlib import Path

from feedback_forensics.app.loader import (
    load_json_file,
    convert_vote_to_string,
    get_votes_df,
    create_votes_df,
)


class TestLoader:
    """Test class for the loader module."""

    @pytest.fixture
    def setup_test_data(self):
        """Set up temporary test directory with mock data files for testing."""
        # Create a temporary directory
        test_dir = Path(tempfile.mkdtemp())

        # Create test data
        # 1. Create principles JSON file
        principles_data = {
            "1": "Principle 1 text",
            "2": "Principle 2 text",
            "3": "Principle 3 text",
        }
        principles_file = test_dir / "030_distilled_principles_per_cluster.json"
        with open(principles_file, "w") as f:
            json.dump(principles_data, f)

        # 2. Create comparison data CSV
        comparisons_data = pd.DataFrame(
            {
                "text_a": ["text A1", "text A2", "text A3"],
                "text_b": ["text B1", "text B2", "text B3"],
                "source": ["source1", "source2", "source3"],
            }
        )
        comparisons_data.index.name = "index"
        comparisons_file = test_dir / "000_train_data.csv"
        comparisons_data.to_csv(comparisons_file)

        # 3. Create votes data CSV
        votes_data = pd.DataFrame(
            {
                "votes": [
                    '{"1": True, "2": False, "3": None}',
                    '{"1": False, "2": True, "3": True}',
                    '{"1": None, "2": None, "3": False}',
                ]
            }
        )
        votes_data.index.name = "index"
        votes_file = test_dir / "040_votes_per_comparison.csv"
        votes_data.to_csv(votes_file)

        yield test_dir

        # Cleanup after tests
        shutil.rmtree(test_dir)

    def test_load_json_file(self, setup_test_data):
        """Test loading JSON file."""
        test_dir = setup_test_data
        principles_file = test_dir / "030_distilled_principles_per_cluster.json"

        # Test loading
        data = load_json_file(principles_file)

        # Verify content
        assert isinstance(data, dict)
        assert len(data) == 3
        assert data["1"] == "Principle 1 text"
        assert data["2"] == "Principle 2 text"
        assert data["3"] == "Principle 3 text"

    def test_convert_vote_to_string(self):
        """Test vote conversion to string."""
        assert convert_vote_to_string(True) == "Agree"
        assert convert_vote_to_string(False) == "Disagree"
        assert convert_vote_to_string(None) == "Not applicable"
        assert convert_vote_to_string("invalid") == "Invalid"

        # Test invalid input
        with pytest.raises(ValueError):
            convert_vote_to_string("unknown_value")

    def test_create_votes_df(self, setup_test_data):
        """Test creating votes dataframe."""
        test_dir = setup_test_data

        # Get the dataframe
        votes_df = create_votes_df(test_dir)

        # Basic validation
        assert isinstance(votes_df, pd.DataFrame)
        assert len(votes_df) == 3  # 3 comparisons

        # Check column creation
        assert "annotation_principle_1" in votes_df.columns
        assert "annotation_principle_2" in votes_df.columns
        assert "annotation_principle_3" in votes_df.columns

        # Check vote conversion
        assert votes_df.loc[0, "annotation_principle_1"] == "Agree"
        assert votes_df.loc[0, "annotation_principle_2"] == "Disagree"
        assert votes_df.loc[0, "annotation_principle_3"] == "Not applicable"

        assert votes_df.loc[1, "annotation_principle_1"] == "Disagree"
        assert votes_df.loc[1, "annotation_principle_2"] == "Agree"
        assert votes_df.loc[1, "annotation_principle_3"] == "Agree"

        # Check weight column
        assert "weight" in votes_df.columns
        assert all(votes_df["weight"] == 1)

        # Ensure temporary column is dropped
        assert "votes_dicts" not in votes_df.columns

    def test_get_votes_df_cache(self, setup_test_data):
        """Test votes df with caching."""
        test_dir = setup_test_data
        cache = {}

        # First call should create and cache
        df1 = get_votes_df(test_dir, cache)
        assert test_dir in cache.get("votes_df", {})

        # Second call should return cached df
        df2 = get_votes_df(test_dir, cache)
        assert df1 is df2  # Should be the same object

    def test_get_votes_df_missing_dir(self):
        """Test getting votes df with missing directory."""
        with pytest.raises(FileNotFoundError):
            get_votes_df(Path("/nonexistent/path"), {})

    def test_get_votes_df_empty_dir(self, setup_test_data):
        """Test getting votes df with empty directory."""
        test_dir = setup_test_data
        empty_dir = Path(tempfile.mkdtemp())

        try:
            with pytest.raises(FileNotFoundError):
                get_votes_df(empty_dir, {})
        finally:
            shutil.rmtree(empty_dir)

"""
Tests for the loader module.
"""

import json
import pandas as pd
import pytest
from pathlib import Path

from feedback_forensics.data.loader import (
    load_json_file,
    convert_vote_to_string,
    get_votes_dict,
    create_votes_dict_from_icai_log_files,
    get_votes_dict_from_annotated_pairs_json,
    add_virtual_annotators,
)
from feedback_forensics.app.constants import DEFAULT_ANNOTATOR_HASH, hash_string


class TestLoader:
    """Test class for the loader module."""

    def test_load_json_file(self, setup_test_data):
        """Test loading JSON file."""
        test_dir = setup_test_data
        principles_file = (
            test_dir / "results" / "030_distilled_principles_per_cluster.json"
        )

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

        # Get the votes dictionary
        votes_dict = create_votes_dict_from_icai_log_files(test_dir / "results")

        # Basic validation
        votes_df = votes_dict["df"]
        assert isinstance(votes_df, pd.DataFrame)
        assert len(votes_df) == 3  # 3 comparisons

        # Check column creation
        assert hash_string("Principle 1 text") in votes_df.columns
        assert hash_string("Principle 2 text") in votes_df.columns
        assert hash_string("Principle 3 text") in votes_df.columns

        # Check vote conversion - we're now checking text_a, text_b or Not applicable
        assert votes_df.loc[0, hash_string("Principle 1 text")] in [
            "text_a",
            "text_b",
            "Not applicable",
        ]
        assert votes_df.loc[0, hash_string("Principle 2 text")] in [
            "text_a",
            "text_b",
            "Not applicable",
        ]
        assert votes_df.loc[0, hash_string("Principle 3 text")] == "Not applicable"

        # Check weight column
        assert "weight" in votes_df.columns
        assert all(votes_df["weight"] == 1)

        # Ensure temporary column is dropped
        assert "votes_dicts" not in votes_df.columns

    def test_get_votes_df_cache(self, setup_test_data):
        """Test votes dict with caching."""
        test_dir = setup_test_data
        cache = {}

        # First call should create and cache
        dict1 = get_votes_dict(test_dir, cache)
        assert test_dir in cache.get("votes_dict", {})

        # Second call should return cached df
        dict2 = get_votes_dict(test_dir, cache)
        assert dict1 is dict2  # Should be the same object

    def test_get_votes_df_missing_dir(self):
        """Test getting votes dict with missing directory."""
        with pytest.raises(FileNotFoundError):
            get_votes_dict(Path("/nonexistent/path"), {})

    def test_get_votes_df_empty_dir(self, tmp_path):
        """Test getting votes dict with empty directory."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            get_votes_dict(empty_dir, {})

    def test_get_votes_dict_from_annotated_pairs_json(self, setup_annotated_pairs_json):
        """Test getting votes dict from AnnotatedPairs JSON."""
        json_file = setup_annotated_pairs_json

        # Get the votes dictionary
        votes_dict = get_votes_dict_from_annotated_pairs_json(json_file)

        # Basic validation
        assert isinstance(votes_dict, dict)
        assert "df" in votes_dict
        assert "shown_annotator_rows" in votes_dict
        assert "annotator_metadata" in votes_dict
        assert "reference_annotator_col" in votes_dict

        # Check DataFrame
        df = votes_dict["df"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 comparisons

        # Check columns
        assert "comparison_id" in df.columns
        assert "text_a" in df.columns
        assert "text_b" in df.columns
        assert "prompt" in df.columns
        assert "d36860d4" in df.columns  # default annotator
        assert "2f45a6d0" in df.columns  # principle annotator
        assert "435cef52" in df.columns  # principle annotator
        assert "weight" in df.columns
        assert "source" in df.columns
        assert "category" in df.columns

        # Check annotator metadata
        annotator_metadata = votes_dict["annotator_metadata"]
        assert "d36860d4" in annotator_metadata
        assert annotator_metadata["d36860d4"]["variant"] == "default_annotator"
        assert "2f45a6d0" in annotator_metadata
        assert annotator_metadata["2f45a6d0"]["variant"] == "icai_principle"
        assert "435cef52" in annotator_metadata
        assert annotator_metadata["435cef52"]["variant"] == "icai_principle"

        # Check shown annotator rows
        shown_annotator_rows = votes_dict["shown_annotator_rows"]
        assert "2f45a6d0" in shown_annotator_rows
        assert "435cef52" in shown_annotator_rows

        # Check reference annotator column
        assert votes_dict["reference_annotator_col"] == DEFAULT_ANNOTATOR_HASH

        # Check principle annotations
        # For the first comparison, all annotators prefer text_a
        assert df.loc[0, "d36860d4"] == "text_a"
        assert df.loc[0, "2f45a6d0"] == "text_a"
        assert df.loc[0, "435cef52"] == "text_a"

        # For the second comparison, default annotator and first principle prefer text_b, second principle prefers text_a
        assert df.loc[1, "d36860d4"] == "text_b"
        assert df.loc[1, "2f45a6d0"] == "text_b"
        assert df.loc[1, "435cef52"] == "text_a"

    def test_get_votes_dict_from_annotated_pairs_json_missing_pref(
        self, setup_annotated_pairs_json
    ):
        """Test getting votes dict from AnnotatedPairs JSON with missing preferences."""
        json_file = setup_annotated_pairs_json

        # Modify the JSON to have a missing preference
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Remove a preference
        json_data["comparisons"][0]["annotations"]["2f45a6d0"] = {}

        # Write back to file
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        # Get the votes dictionary
        votes_dict = get_votes_dict_from_annotated_pairs_json(json_file)

        # Check that the function still works
        assert isinstance(votes_dict, dict)
        assert "df" in votes_dict

        # Check that the missing preference is handled
        df = votes_dict["df"]
        assert "2f45a6d0" in df.columns

    def test_get_votes_dict_from_annotated_pairs_json_with_metadata(
        self, setup_annotated_pairs_json
    ):
        """Test getting votes dict from AnnotatedPairs JSON with metadata."""
        json_file = setup_annotated_pairs_json

        # Get the votes dictionary
        votes_dict = get_votes_dict_from_annotated_pairs_json(json_file)

        # Check DataFrame
        df = votes_dict["df"]

        # Check metadata columns
        assert "source" in df.columns
        assert "category" in df.columns

        # Check metadata values
        assert df.loc[0, "source"] == "test_source"
        assert df.loc[0, "category"] == "fiction"
        assert df.loc[1, "source"] == "test_source"
        assert df.loc[1, "category"] == "fiction"

    def test_get_votes_dict_from_annotated_pairs_json_v2(
        self, setup_annotated_pairs_json_v2
    ):
        """Test getting votes dict from AnnotatedPairs JSON with format v2.0."""
        json_file = setup_annotated_pairs_json_v2

        # Get the votes dictionary
        votes_dict = get_votes_dict_from_annotated_pairs_json(json_file)

        # Basic validation
        assert isinstance(votes_dict, dict)
        assert "df" in votes_dict
        assert "shown_annotator_rows" in votes_dict
        assert "annotator_metadata" in votes_dict
        assert "reference_annotator_col" in votes_dict

        # Check DataFrame
        df = votes_dict["df"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 comparisons

        # Check columns
        assert "comparison_id" in df.columns
        assert "text_a" in df.columns
        assert "text_b" in df.columns
        assert "model_a" in df.columns
        assert "model_b" in df.columns
        assert "prompt" in df.columns
        assert "d36860d4" in df.columns  # default annotator
        assert "2f45a6d0" in df.columns  # principle annotator
        assert "435cef52" in df.columns  # principle annotator
        assert "weight" in df.columns
        assert "source" in df.columns
        assert "category" in df.columns

        # Check annotator metadata
        annotator_metadata = votes_dict["annotator_metadata"]
        assert "d36860d4" in annotator_metadata
        assert annotator_metadata["d36860d4"]["variant"] == "default_annotator"
        assert "2f45a6d0" in annotator_metadata
        assert annotator_metadata["2f45a6d0"]["variant"] == "icai_principle"
        assert "435cef52" in annotator_metadata
        assert annotator_metadata["435cef52"]["variant"] == "icai_principle"

        # Check shown annotator rows
        shown_annotator_rows = votes_dict["shown_annotator_rows"]
        assert "2f45a6d0" in shown_annotator_rows
        assert "435cef52" in shown_annotator_rows

        # Check all response fields are correctly transformed to key_a/key_b format
        assert df.loc[0, "model_a"] == "Model X"
        assert df.loc[0, "model_b"] == "Model Y"
        assert df.loc[0, "text_a"].startswith("In the heart of a bustling city")
        assert df.loc[0, "text_b"].startswith("Across the town")
        assert df.loc[0, "timestamp_a"] == "2025-04-01T12:00:00Z"
        assert df.loc[0, "timestamp_b"] == "2025-04-01T12:05:00Z"
        assert df.loc[0, "rating_a"] == 5
        assert df.loc[0, "rating_b"] == 4

        assert df.loc[1, "model_a"] == "Model X"
        assert df.loc[1, "model_b"] == "Model Y"
        assert df.loc[1, "timestamp_a"] == "2025-04-01T13:00:00Z"
        assert df.loc[1, "timestamp_b"] == "2025-04-01T13:05:00Z"
        assert df.loc[1, "rating_a"] == 4
        assert df.loc[1, "rating_b"] == 5

        # Check annotations - a/b should be converted to text_a/text_b
        assert df.loc[0, "d36860d4"] == "text_a"
        assert df.loc[0, "2f45a6d0"] == "text_a"
        assert df.loc[0, "435cef52"] == "text_a"
        assert df.loc[1, "d36860d4"] == "text_b"
        assert df.loc[1, "2f45a6d0"] == "text_b"
        assert df.loc[1, "435cef52"] == "text_a"

    def test_model_annotators(self, setup_annotated_pairs_json_v2):
        """Test adding model annotators to votes dict."""
        json_file = setup_annotated_pairs_json_v2
        cache = {}

        base_votes_dict = get_votes_dict(json_file, cache)

        # Add virtual annotators
        reference_models = []
        target_models = []
        votes_dict_with_annotators = add_virtual_annotators(
            base_votes_dict, cache, json_file, reference_models, target_models
        )

        # Check for model annotators in metadata
        annotator_metadata = votes_dict_with_annotators["annotator_metadata"]

        # Find model annotator keys
        model_annotator_keys = [
            key
            for key, meta in annotator_metadata.items()
            if meta.get("variant") == "model_identity"
        ]

        assert len(model_annotator_keys) >= 2

        # Check model annotator columns in dataframe
        df = votes_dict_with_annotators["df"]
        for key in model_annotator_keys:
            assert key in df.columns

        # Check that model annotators have correct metadata format
        for key in model_annotator_keys:
            meta = annotator_metadata[key]
            assert "model_id" in meta
            assert "annotator_visible_name" in meta
            assert meta["annotator_visible_name"].startswith("Model: ")

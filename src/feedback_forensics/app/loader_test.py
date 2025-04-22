"""
Tests for the loader module.
"""

import json
import pandas as pd
import pytest
from pathlib import Path

from feedback_forensics.app.loader import (
    load_json_file,
    convert_vote_to_string,
    get_votes_dict,
    create_votes_dict_from_icai_log_files,
    get_votes_dict_from_annotated_pairs_json,
)
from feedback_forensics.app.constants import DEFAULT_ANNOTATOR_HASH, hash_string


class TestLoader:
    """Test class for the loader module."""

    @pytest.fixture
    def setup_test_data(self, tmp_path):
        """Set up temporary test directory with mock data files for testing."""
        # Create test data
        # 1. Create principles JSON file
        principles_data = {
            "1": "Principle 1 text",
            "2": "Principle 2 text",
            "3": "Principle 3 text",
        }
        results_path = tmp_path / "results"
        results_path.mkdir(parents=True, exist_ok=True)

        principles_file = results_path / "030_distilled_principles_per_cluster.json"
        with open(principles_file, "w", encoding="utf-8") as f:
            json.dump(principles_data, f)

        # 2. Create comparison data CSV
        comparisons_data = pd.DataFrame(
            {
                "text_a": ["text A1", "text A2", "text A3"],
                "text_b": ["text B1", "text B2", "text B3"],
                "source": ["source1", "source2", "source3"],
                "preferred_text": ["text_a", "text_b", "text_a"],
            }
        )
        comparisons_data.index.name = "index"
        comparisons_file = results_path / "000_train_data.csv"
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
        votes_file = results_path / "040_votes_per_comparison.csv"
        votes_data.to_csv(votes_file)

        return tmp_path

    @pytest.fixture
    def setup_annotated_pairs_json(self, tmp_path):
        """Set up a temporary JSON file with annotated pairs data for testing."""
        # Create test JSON data
        json_data = {
            "metadata": {
                "version": "1.0",
                "description": "Annotated pairs dataset with annotations from ICAI",
                "created_at": "2025-04-02T16:02:37Z",
                "dataset_name": "ICAI Dataset - 2025-04-02_16-02-05",
                "default_annotator": "d36860d4",
            },
            "annotators": {
                "d36860d4": {
                    "name": "Human",
                    "description": "Human annotator from original dataset",
                    "type": "human",
                },
                "2f45a6d0": {
                    "description": "Select the response that evokes a sense of mystery.",
                    "type": "principle",
                },
                "435cef52": {
                    "description": "Select the response that features a more adventurous setting.",
                    "type": "principle",
                },
            },
            "comparisons": [
                {
                    "id": "2fbb184f",
                    "prompt": "Write a story about a pet.",
                    "text_a": "In the heart of a bustling city, a sleek black cat named Shadow prowled the moonlit rooftops, her eyes gleaming with curiosity and mischief. She discovered a hidden garden atop an old apartment building, where she danced under the stars, chasing fireflies that glowed like tiny lanterns. As dawn painted the sky in hues of orange and pink, Shadow found her way back home, carrying the secret of the garden in her heart.",
                    "text_b": "Across the town, in a cozy neighborhood, a golden retriever named Buddy embarked on his daily adventure, tail wagging with uncontainable excitement. He found a lost toy under the bushes in the park, its colors faded and fabric worn, but to Buddy, it was a treasure untold. Returning home with his newfound prize, Buddy's joyful barks filled the air, reminding everyone in the house that happiness can be found in the simplest of things.",
                    "annotations": {
                        "d36860d4": {"pref": "text_a"},
                        "2f45a6d0": {"pref": "text_a"},
                        "435cef52": {"pref": "text_a"},
                    },
                    "metadata": {"source": "test_source", "category": "fiction"},
                },
                {
                    "id": "3a7c9e2d",
                    "prompt": "Write a story about a pet.",
                    "text_a": "In a quiet suburban backyard, a small rabbit named Hoppy nibbled on fresh carrots, his nose twitching with delight. The garden was his kingdom, filled with tall grass to hide in and flowers to admire. As the sun set, Hoppy would return to his cozy hutch, dreaming of tomorrow's adventures in his little paradise.",
                    "text_b": "Deep in the forest, a wise old owl named Oliver perched high in an ancient oak tree, watching over the woodland creatures below. His keen eyes spotted a family of mice scurrying home, and he hooted softly, a gentle reminder that he was their silent guardian. As night fell, Oliver spread his wings and soared through the moonlit sky, a majestic shadow against the stars.",
                    "annotations": {
                        "d36860d4": {"pref": "text_b"},
                        "2f45a6d0": {"pref": "text_b"},
                        "435cef52": {"pref": "text_a"},
                    },
                    "metadata": {"source": "test_source", "category": "fiction"},
                },
            ],
        }

        # Write JSON to file
        json_file = tmp_path / "annotated_pairs.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        return json_file

    @pytest.fixture
    def setup_annotated_pairs_json_v2(self, tmp_path):
        """Set up a temporary JSON file with annotated pairs data for testing format v2.0."""
        json_data = {
            "metadata": {
                "version": "2.0",
                "description": "Annotated pairs dataset with annotations from ICAI",
                "created_at": "2025-04-02T16:02:37Z",
                "dataset_name": "ICAI Dataset - 2025-04-02_16-02-05",
                "default_annotator": "d36860d4",
            },
            "annotators": {
                "d36860d4": {
                    "name": "Human",
                    "description": "Human annotator from original dataset",
                    "type": "human",
                },
                "2f45a6d0": {
                    "description": "Select the response that evokes a sense of mystery.",
                    "type": "principle",
                },
                "435cef52": {
                    "description": "Select the response that features a more adventurous setting.",
                    "type": "principle",
                },
            },
            "comparisons": [
                {
                    "id": "2fbb184f",
                    "prompt": "Write a story about a pet.",
                    "response_a": {
                        "text": "In the heart of a bustling city, a sleek black cat named Shadow prowled the moonlit rooftops, her eyes gleaming with curiosity and mischief. She discovered a hidden garden atop an old apartment building, where she danced under the stars, chasing fireflies that glowed like tiny lanterns. As dawn painted the sky in hues of orange and pink, Shadow found her way back home, carrying the secret of the garden in her heart.",
                        "model": "Model X",
                        "timestamp": "2025-04-01T12:00:00Z",
                        "rating": 5,
                    },
                    "response_b": {
                        "text": "Across the town, in a cozy neighborhood, a golden retriever named Buddy embarked on his daily adventure, tail wagging with uncontainable excitement. He found a lost toy under the bushes in the park, its colors faded and fabric worn, but to Buddy, it was a treasure untold. Returning home with his newfound prize, Buddy's joyful barks filled the air, reminding everyone in the house that happiness can be found in the simplest of things.",
                        "model": "Model Y",
                        "timestamp": "2025-04-01T12:05:00Z",
                        "rating": 4,
                    },
                    "annotations": {
                        "d36860d4": {"pref": "a"},
                        "2f45a6d0": {"pref": "a"},
                        "435cef52": {"pref": "a"},
                    },
                    "metadata": {"source": "test_source", "category": "fiction"},
                },
                {
                    "id": "3a7c9e2d",
                    "prompt": "Write a story about a pet.",
                    "response_a": {
                        "text": "In a quiet suburban backyard, a small rabbit named Hoppy nibbled on fresh carrots, his nose twitching with delight. The garden was his kingdom, filled with tall grass to hide in and flowers to admire. As the sun set, Hoppy would return to his cozy hutch, dreaming of tomorrow's adventures in his little paradise.",
                        "model": "Model X",
                        "timestamp": "2025-04-01T13:00:00Z",
                        "rating": 4,
                    },
                    "response_b": {
                        "text": "Deep in the forest, a wise old owl named Oliver perched high in an ancient oak tree, watching over the woodland creatures below. His keen eyes spotted a family of mice scurrying home, and he hooted softly, a gentle reminder that he was their silent guardian. As night fell, Oliver spread his wings and soared through the moonlit sky, a majestic shadow against the stars.",
                        "model": "Model Y",
                        "timestamp": "2025-04-01T13:05:00Z",
                        "rating": 5,
                    },
                    "annotations": {
                        "d36860d4": {"pref": "b"},
                        "2f45a6d0": {"pref": "b"},
                        "435cef52": {"pref": "a"},
                    },
                    "metadata": {"source": "test_source", "category": "fiction"},
                },
            ],
        }

        json_file = tmp_path / "annotated_pairs_v2.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        return json_file

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
        """Test getting votes dict from annotated pairs JSON."""
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
        """Test getting votes dict from annotated pairs JSON with missing preferences."""
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
        """Test getting votes dict from annotated pairs JSON with metadata."""
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
        """Test getting votes dict from annotated pairs JSON with format v2.0."""
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

"""Tests for core data operations."""

import json
import pytest
from pathlib import Path

from feedback_forensics.data.operations.core import csv_to_ap


def test_csv_to_ap(tmp_path):
    """Test CSV to AnnotatedPairs conversion."""
    csv_content = """text_a,text_b,preferred_text
"Hello world","Hi there","text_a"
"Good morning","Good day","text_b"
"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    result = csv_to_ap(csv_file, "Test Dataset")

    assert "metadata" in result
    assert "annotators" in result
    assert "comparisons" in result

    assert result["metadata"]["dataset_name"] == "Test Dataset"
    assert result["metadata"]["version"] == "2.0"
    assert "created_at" in result["metadata"]

    assert len(result["comparisons"]) == 2

    default_annotator_id = result["metadata"]["default_annotator"]

    comparison1 = result["comparisons"][0]
    assert comparison1["response_a"]["text"] == "Hello world"
    assert comparison1["response_b"]["text"] == "Hi there"
    assert comparison1["annotations"][default_annotator_id]["pref"] == "a"

    comparison2 = result["comparisons"][1]
    assert comparison2["response_a"]["text"] == "Good morning"
    assert comparison2["response_b"]["text"] == "Good day"
    assert comparison2["annotations"][default_annotator_id]["pref"] == "b"


def test_csv_to_ap_missing_columns(tmp_path):
    """Test CSV with missing required columns."""
    csv_content = """text_a,preferred_text
"Hello","text_a"
"""
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text(csv_content)

    with pytest.raises(ValueError, match="missing required columns"):
        csv_to_ap(csv_file, "Test")

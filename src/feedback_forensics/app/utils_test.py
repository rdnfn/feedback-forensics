import json
import os
import tempfile
from pathlib import Path

import pytest

from feedback_forensics.app.utils import get_value_from_json


@pytest.fixture
def sample_json_file():
    """Create a temporary JSON file with test data."""
    data = {
        "user": {
            "name": "John Doe",
            "settings": {
                "theme": "dark",
                "notifications": True,
                "preferences": {"language": "en", "timezone": "UTC"},
            },
        },
        "metadata": {"version": "1.0", "last_updated": "2023-01-01"},
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_file:
        json.dump(data, temp_file)
        temp_file_path = temp_file.name

    yield temp_file_path

    # Clean up the temporary file
    os.unlink(temp_file_path)


def test_get_value_from_json_simple_key(sample_json_file):
    """Test getting a simple key from the JSON file."""
    result = get_value_from_json(sample_json_file, "user.name")
    assert result == "John Doe"


def test_get_value_from_json_nested_key(sample_json_file):
    """Test getting a nested key from the JSON file."""
    result = get_value_from_json(sample_json_file, "user.settings.theme")
    assert result == "dark"


def test_get_value_from_json_deeply_nested_key(sample_json_file):
    """Test getting a deeply nested key from the JSON file."""
    result = get_value_from_json(sample_json_file, "user.settings.preferences.language")
    assert result == "en"


def test_get_value_from_json_nonexistent_key(sample_json_file):
    """Test getting a nonexistent key from the JSON file."""
    result = get_value_from_json(sample_json_file, "nonexistent.key")
    assert result is None


def test_get_value_from_json_nonexistent_nested_key(sample_json_file):
    """Test getting a nonexistent nested key from the JSON file."""
    result = get_value_from_json(sample_json_file, "user.settings.nonexistent")
    assert result is None


def test_get_value_from_json_with_path_object(sample_json_file):
    """Test getting a value using a Path object for the file path."""
    path = Path(sample_json_file)
    result = get_value_from_json(path, "metadata.version")
    assert result == "1.0"


def test_get_value_from_json_with_invalid_file():
    """Test getting a value from an invalid JSON file."""
    result = get_value_from_json("nonexistent_file.json", "some.key")
    assert result is None


def test_get_value_from_json_with_invalid_json():
    """Test getting a value from a file with invalid JSON."""
    # Create a temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_file:
        temp_file.write("This is not valid JSON")
        temp_file_path = temp_file.name

    try:
        result = get_value_from_json(temp_file_path, "some.key")
        assert result is None
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

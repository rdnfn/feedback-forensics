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
            "tags": ["developer", "admin", "user"],
            "access_history": [
                {"timestamp": "2023-01-01", "action": "login"},
                {"timestamp": "2023-01-02", "action": "update"},
                {"timestamp": "2023-01-03", "action": "logout"},
            ],
        },
        "metadata": {"version": "1.0", "last_updated": "2023-01-01"},
        "available_metadata_keys_per_comparison": [
            "index",
            "category",
            "comparison_id",
            "completion_a",
            "completion_b",
        ],
        "numbers": [1, 2, 3, 4, 5],
        "mixed_array": [1, "two", 3.0, True, None],
        "nested": {
            "arrays": {
                "simple": [1, 2, 3],
                "complex": [
                    {"id": 1, "value": "first"},
                    {"id": 2, "value": "second"},
                    {"id": 3, "value": "third"},
                ],
            }
        },
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


def test_get_value_from_json_simple_array(sample_json_file):
    """Test getting a simple array of primitive values."""
    result = get_value_from_json(sample_json_file, "numbers")
    assert result == [1, 2, 3, 4, 5]


def test_get_value_from_json_string_array(sample_json_file):
    """Test getting an array of strings."""
    result = get_value_from_json(sample_json_file, "user.tags")
    assert result == ["developer", "admin", "user"]


def test_get_value_from_json_mixed_array(sample_json_file):
    """Test getting an array with mixed value types."""
    result = get_value_from_json(sample_json_file, "mixed_array")
    assert result == [1, "two", 3.0, True, None]


def test_get_value_from_json_nested_simple_array(sample_json_file):
    """Test getting a nested simple array."""
    result = get_value_from_json(sample_json_file, "nested.arrays.simple")
    assert result == [1, 2, 3]


def test_get_value_from_json_array_of_objects(sample_json_file):
    """Test getting an array of objects."""
    result = get_value_from_json(sample_json_file, "user.access_history")
    expected = [
        {"timestamp": "2023-01-01", "action": "login"},
        {"timestamp": "2023-01-02", "action": "update"},
        {"timestamp": "2023-01-03", "action": "logout"},
    ]
    assert result == expected


def test_get_value_from_json_nested_array_of_objects(sample_json_file):
    """Test getting a nested array of objects."""
    result = get_value_from_json(sample_json_file, "nested.arrays.complex")
    expected = [
        {"id": 1, "value": "first"},
        {"id": 2, "value": "second"},
        {"id": 3, "value": "third"},
    ]
    assert result == expected


def test_nested_dict_with_array_of_objects(sample_json_file):
    """Test getting a nested dict with an array of objects."""
    result = get_value_from_json(sample_json_file, "nested.arrays")
    expected = [
        {"id": 1, "value": "first"},
        {"id": 2, "value": "second"},
        {"id": 3, "value": "third"},
    ]

    assert isinstance(result, dict)
    assert result["complex"] == expected

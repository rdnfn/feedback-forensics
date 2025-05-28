"""Tests for URL parser functionality."""

import pytest
from unittest.mock import Mock, patch
import gradio as gr

from feedback_forensics.app.url_parser import (
    get_config_from_query_params,
    get_url_with_query_params,
    make_str_url_ready,
)


def test_get_config_from_query_params_with_analysis_mode():
    """Test that analysis_mode is correctly parsed from query parameters."""
    # Mock request object without dataset validation
    request = Mock()
    request.query_params = {
        "analysis_mode": "annotation_analysis",
        "metric": "strength",  # Don't include data to avoid validation
    }

    config = get_config_from_query_params(request)

    assert config is not None
    assert "analysis_mode" in config
    assert config["analysis_mode"] == "annotation_analysis"


def test_get_config_from_query_params_without_analysis_mode():
    """Test that config works correctly when analysis_mode is not provided."""
    # Mock request object
    request = Mock()
    request.query_params = {"metric": "strength"}

    config = get_config_from_query_params(request)

    assert config is not None
    assert "analysis_mode" not in config
    assert "metric" in config
    assert config["metric"] == "strength"


@patch("feedback_forensics.app.url_parser.get_available_datasets")
@patch("feedback_forensics.app.url_parser.get_urlname_from_stringname")
def test_get_url_with_query_params_with_analysis_mode(
    mock_get_urlname, mock_get_datasets
):
    """Test that analysis_mode is correctly included in generated URLs."""
    # Mock the dataset functions to avoid validation issues
    mock_get_datasets.return_value = {}
    mock_get_urlname.return_value = "test_dataset"

    url = get_url_with_query_params(
        datasets=["test_dataset"],
        col=None,
        col_vals=[],
        base_url="http://localhost:7860",
        analysis_mode="advanced_settings",
    )

    assert "analysis_mode=advanced_settings" in url


@patch("feedback_forensics.app.url_parser.get_available_datasets")
@patch("feedback_forensics.app.url_parser.get_urlname_from_stringname")
def test_get_url_with_query_params_without_analysis_mode(
    mock_get_urlname, mock_get_datasets
):
    """Test that URL generation works correctly when analysis_mode is None."""
    # Mock the dataset functions to avoid validation issues
    mock_get_datasets.return_value = {}
    mock_get_urlname.return_value = "test_dataset"

    url = get_url_with_query_params(
        datasets=["test_dataset"],
        col=None,
        col_vals=[],
        base_url="http://localhost:7860",
        analysis_mode=None,
    )

    assert "analysis_mode" not in url
    assert "data=test_dataset" in url


def test_make_str_url_ready_with_analysis_mode_values():
    """Test that analysis mode values are correctly URL-encoded."""
    # The make_str_url_ready function removes special characters but keeps underscores
    assert make_str_url_ready("model_analysis") == "model_analysis"
    assert make_str_url_ready("annotation_analysis") == "annotation_analysis"
    assert make_str_url_ready("advanced_settings") == "advanced_settings"


@patch("feedback_forensics.app.url_parser.get_available_datasets")
@patch("feedback_forensics.app.url_parser.get_urlname_from_stringname")
def test_get_url_with_query_params_all_analysis_modes(
    mock_get_urlname, mock_get_datasets
):
    """Test URL generation with all valid analysis modes."""
    # Mock the dataset functions to avoid validation issues
    mock_get_datasets.return_value = {}
    mock_get_urlname.return_value = "test_dataset"

    base_url = "http://localhost:7860"
    datasets = ["test_dataset"]

    for mode in ["model_analysis", "annotation_analysis", "advanced_settings"]:
        url = get_url_with_query_params(
            datasets=datasets,
            col=None,
            col_vals=[],
            base_url=base_url,
            analysis_mode=mode,
        )

        assert f"analysis_mode={mode}" in url


def test_analysis_mode_only_config():
    """Test parsing config with only analysis_mode parameter."""
    request = Mock()
    request.query_params = {"analysis_mode": "advanced_settings"}

    config = get_config_from_query_params(request)

    assert config is not None
    assert config == {"analysis_mode": "advanced_settings"}

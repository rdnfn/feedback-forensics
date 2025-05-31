"""Tests for merge operations module."""

from unittest.mock import patch

import pytest

from feedback_forensics.data.operations.merge import (
    merge_ap,
    _categorize_comparisons,
    _merge_single_comparison,
    _merge_metadata,
    _merge_value,
    _merge_dict,
)


@pytest.fixture
def sample_annotated_pairs_1():
    """First sample AnnotatedPairs dataset."""
    return {
        "metadata": {
            "version": "2.0",
            "dataset_name": "Dataset1",
            "default_annotator": "1262121b",
            "created_at": "2025-01-01T00:00:00Z",
        },
        "annotators": {
            "1262121b": {
                "name": "Human Annotator 1",
                "description": "Human annotator from dataset 1",
                "type": "human",
            },
            "bf731c7f": {
                "description": "Select the response that is more concise",
                "type": "principle",
            },
        },
        "comparisons": [
            {
                "id": "comp1",
                "prompt": "Explain AI",
                "response_a": {
                    "text": "AI is artificial intelligence",
                    "model": "model-1",
                },
                "response_b": {
                    "text": "Artificial intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence",
                    "model": "model-2",
                },
                "annotations": {"1262121b": {"pref": "a"}, "bf731c7f": {"pref": "a"}},
                "metadata": {"index": "0"},
            },
            {
                "id": "comp2",
                "prompt": "What is ML?",
                "response_a": {"text": "Machine learning", "model": "model-1"},
                "response_b": {"text": "ML is a subset of AI", "model": "model-2"},
                "annotations": {"1262121b": {"pref": "b"}},
            },
        ],
    }


@pytest.fixture
def sample_annotated_pairs_2():
    """Second sample AnnotatedPairs dataset."""
    return {
        "metadata": {
            "version": "2.0",
            "dataset_name": "Dataset2",
            "default_annotator": "a1522ded",
            "created_at": "2025-01-02T00:00:00Z",
        },
        "annotators": {
            "a1522ded": {
                "name": "Human Annotator 2",
                "description": "Human annotator from dataset 2",
                "type": "human",
            },
            "9a1afb94": {
                "description": "Select the response that is more detailed",
                "type": "principle",
            },
        },
        "comparisons": [
            {
                "id": "comp1",
                "prompt": "Explain AI",
                "response_a": {
                    "text": "AI is artificial intelligence",
                    "model": "model-1",
                },
                "response_b": {
                    "text": "Artificial intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence",
                    "model": "model-2",
                },
                "annotations": {"a1522ded": {"pref": "b"}, "9a1afb94": {"pref": "b"}},
            },
            {
                "id": "comp3",
                "prompt": "Define deep learning",
                "response_a": {
                    "text": "Deep learning is a type of ML",
                    "model": "model-1",
                },
                "response_b": {
                    "text": "Deep learning uses neural networks with multiple layers",
                    "model": "model-2",
                },
                "annotations": {"a1522ded": {"pref": "b"}},
            },
        ],
    }


class TestCategorizeComparisons:
    """Tests for _categorize_comparisons function."""

    def test_find_matches_with_ids(
        self, sample_annotated_pairs_1, sample_annotated_pairs_2
    ):
        """Test finding matches when comparisons have IDs."""
        comps1 = sample_annotated_pairs_1["comparisons"]
        comps2 = sample_annotated_pairs_2["comparisons"]

        paired, first_only, second_only = _categorize_comparisons(comps1, comps2)

        assert len(paired) == 1
        assert paired[0] == (0, 0)
        assert len(first_only) == 1
        assert first_only[0] == 1
        assert len(second_only) == 1
        assert second_only[0] == 1

    def test_find_matches_without_ids(self):
        """Test finding matches when comparisons don't have IDs (should generate them)."""
        comps1 = [
            {
                "prompt": "Test prompt",
                "response_a": {"text": "Response A"},
                "response_b": {"text": "Response B"},
            }
        ]
        comps2 = [
            {
                "prompt": "Test prompt",
                "response_a": {"text": "Response A"},
                "response_b": {"text": "Response B"},
            }
        ]

        paired, first_only, second_only = _categorize_comparisons(comps1, comps2)

        assert len(paired) == 1
        assert len(first_only) == 0
        assert len(second_only) == 0

        assert "id" in comps1[0]
        assert "id" in comps2[0]
        assert comps1[0]["id"] == comps2[0]["id"]


class TestMergeSingleComparison:
    """Tests for _merge_single_comparison function."""

    def test_merge_comparison_basic(self):
        """Test basic comparison merging."""
        comp1 = {
            "id": "test1",
            "prompt": "Test prompt",
            "response_a": {"text": "A1", "model": "model1"},
            "response_b": {"text": "B1", "model": "model2"},
            "annotations": {"ann1": {"pref": "a"}},
            "metadata": {"source": "first"},
        }
        comp2 = {
            "id": "test1",
            "response_a": {"text": "A1", "model": "model1"},
            "response_b": {"text": "B1", "model": "model2"},
            "annotations": {"ann2": {"pref": "b"}},
            "metadata": {"source": "second"},
        }

        merged = _merge_single_comparison(comp1, comp2)

        assert merged["id"] == "test1"
        assert merged["prompt"] == "Test prompt"
        assert merged["annotations"] == {"ann2": {"pref": "b"}, "ann1": {"pref": "a"}}
        assert merged["metadata"] == {"source": "first"}

    def test_merge_comparison_missing_fields(self):
        """Test merging when some fields are missing."""
        comp1 = {
            "id": "test1",
            "response_a": {"text": "A1"},
            "annotations": {"ann1": {"pref": "a"}},
        }
        comp2 = {
            "prompt": "Test prompt",
            "response_b": {"text": "B1"},
            "annotations": {"ann2": {"pref": "b"}},
        }

        merged = _merge_single_comparison(comp1, comp2)

        assert merged["id"] == "test1"
        assert merged["prompt"] == "Test prompt"
        assert merged["response_a"] == {"text": "A1"}
        assert merged["response_b"] == {"text": "B1"}
        assert merged["annotations"] == {"ann2": {"pref": "b"}, "ann1": {"pref": "a"}}

    def test_merge_prompt_conflict(self):
        """Test that conflicting prompts raise an error."""
        comp1 = {"id": "test1", "prompt": "What is AI?"}
        comp2 = {"id": "test1", "prompt": "What is ML?"}

        with pytest.raises(
            ValueError,
            match="Prompt conflict in comparison test1.*",
        ):
            _merge_single_comparison(comp1, comp2)

    def test_merge_response_text_conflict(self):
        """Test that conflicting response text raises an error."""
        comp1 = {
            "id": "test1",
            "response_a": {"text": "Text 1", "model": "model1"},
        }
        comp2 = {
            "id": "test1",
            "response_a": {"text": "Text 2", "model": "model1"},
        }

        with pytest.raises(
            ValueError,
            match='Conflicting value for "text" key in comparison test1 response_a.*',
        ):
            _merge_single_comparison(comp1, comp2)

    def test_merge_response_model_conflict(self):
        """Test that conflicting response models raise an error."""
        comp1 = {
            "id": "test1",
            "response_a": {"text": "Same text", "model": "model1"},
        }
        comp2 = {
            "id": "test1",
            "response_a": {"text": "Same text", "model": "model2"},
        }

        with pytest.raises(
            ValueError,
            match='Conflicting value for "model" key in comparison test1 response_a.*',
        ):
            _merge_single_comparison(comp1, comp2)

    @patch("feedback_forensics.data.operations.merge.logger")
    def test_merge_annotation_conflict(self, mock_logger):
        """Test that annotation conflicts log warnings but use first dataset."""
        comp1 = {
            "id": "test1",
            "annotations": {"shared_annotator": {"pref": "a"}},
        }
        comp2 = {
            "id": "test1",
            "annotations": {"shared_annotator": {"pref": "b"}},
        }

        merged = _merge_single_comparison(comp1, comp2)

        assert merged["annotations"]["shared_annotator"] == {"pref": "a"}
        mock_logger.warning.assert_called_with(
            "Conflicting value for \"shared_annotator\" key in comparison \"test1\" annotations: \"{'pref': 'a'}\" vs \"{'pref': 'b'}\", using first dataset value"
        )

    @patch("feedback_forensics.data.operations.merge.logger")
    def test_merge_metadata_conflict(self, mock_logger):
        """Test that metadata conflicts log warnings but use first dataset."""
        comp1 = {
            "id": "test1",
            "metadata": {"source": "dataset1"},
        }
        comp2 = {
            "id": "test1",
            "metadata": {"source": "dataset2"},
        }

        merged = _merge_single_comparison(comp1, comp2)

        assert merged["metadata"]["source"] == "dataset1"
        mock_logger.warning.assert_called_with(
            'Conflicting value for "source" key in comparison dictionary "test1": "dataset1" vs "dataset2", using first dataset value'
        )


class TestMergeMetadata:
    """Tests for _merge_metadata function."""

    def test_merge_metadata(self):
        """Test metadata merging."""
        meta1 = {
            "dataset_name": "Dataset A",
            "default_annotator": "1262121b",
            "custom_field": "value1",
        }
        meta2 = {
            "dataset_name": "Dataset B",
            "default_annotator": "a1522ded",
            "other_field": "value2",
        }

        merged = _merge_metadata(meta1, meta2)

        assert merged["version"] == "2.0"
        assert merged["dataset_name"] == "Merged: Dataset A + Dataset B"
        assert "description" not in merged
        assert merged["default_annotator"] == "1262121b"
        assert merged["custom_field"] == "value1"
        assert "created_at" in merged

    def test_merge_metadata_with_description(self):
        """Test metadata merging when datasets have descriptions."""
        meta1 = {
            "dataset_name": "Dataset A",
            "description": "First dataset description",
        }
        meta2 = {
            "dataset_name": "Dataset B",
        }

        merged = _merge_metadata(meta1, meta2)

        assert merged["description"] == "First dataset description"

        # Test merging different descriptions
        meta2["description"] = "Second dataset description"
        merged = _merge_metadata(meta1, meta2)
        assert (
            merged["description"]
            == "Merged: First dataset description + Second dataset description"
        )


class TestMergeAP:
    """Tests for merge_ap function (data-only merging)."""

    def test_merge_ap_basic(self, sample_annotated_pairs_1, sample_annotated_pairs_2):
        """Test basic merging of two datasets."""
        merged = merge_ap(sample_annotated_pairs_1, sample_annotated_pairs_2)

        assert "metadata" in merged
        assert "annotators" in merged
        assert "comparisons" in merged

        assert len(merged["annotators"]) == 4
        assert len(merged["comparisons"]) == 3

        merged_comp = next(c for c in merged["comparisons"] if c["id"] == "comp1")
        assert len(merged_comp["annotations"]) == 4

    def test_merge_ap_empty_datasets(self):
        """Test merging empty datasets."""
        empty1 = {
            "metadata": {"version": "2.0", "dataset_name": "Empty1"},
            "annotators": {},
            "comparisons": [],
        }
        empty2 = {
            "metadata": {"version": "2.0", "dataset_name": "Empty2"},
            "annotators": {},
            "comparisons": [],
        }

        merged = merge_ap(empty1, empty2)

        assert len(merged["annotators"]) == 0
        assert len(merged["comparisons"]) == 0
        assert "Empty1" in merged["metadata"]["dataset_name"]
        assert "Empty2" in merged["metadata"]["dataset_name"]

    def test_merge_ap_precedence(self):
        """Test that first dataset takes precedence in conflicts."""
        data1 = {
            "metadata": {"dataset_name": "First", "custom": "from_first"},
            "annotators": {"shared": {"type": "human"}},
            "comparisons": [],
        }
        data2 = {
            "metadata": {"dataset_name": "Second", "custom": "from_second"},
            "annotators": {"shared": {"type": "principle"}},
            "comparisons": [],
        }

        merged = merge_ap(data1, data2)

        assert merged["annotators"]["shared"]["type"] == "human"
        assert merged["metadata"]["custom"] == "from_first"

    def test_merge_ap_strict_validation_fails(self):
        """Test that merge fails with strict validation when core data conflicts."""
        data1 = {
            "metadata": {"dataset_name": "First"},
            "annotators": {},
            "comparisons": [
                {
                    "id": "comp1",
                    "prompt": "What is AI?",
                    "response_a": {"text": "AI is artificial intelligence"},
                    "response_b": {"text": "AI is machine learning"},
                    "annotations": {},
                }
            ],
        }
        data2 = {
            "metadata": {"dataset_name": "Second"},
            "annotators": {},
            "comparisons": [
                {
                    "id": "comp1",
                    "prompt": "What is ML?",
                    "response_a": {"text": "AI is artificial intelligence"},
                    "response_b": {"text": "AI is machine learning"},
                    "annotations": {},
                }
            ],
        }

        with pytest.raises(
            ValueError,
            match="Prompt conflict in comparison comp1.*",
        ):
            merge_ap(data1, data2)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_datasets(self):
        """Test merging empty datasets."""
        empty1 = {
            "metadata": {"version": "2.0", "dataset_name": "Empty1"},
            "annotators": {},
            "comparisons": [],
        }
        empty2 = {
            "metadata": {"version": "2.0", "dataset_name": "Empty2"},
            "annotators": {},
            "comparisons": [],
        }

        result = merge_ap(empty1, empty2)

        assert len(result["annotators"]) == 0
        assert len(result["comparisons"]) == 0
        assert "Empty1" in result["metadata"]["dataset_name"]
        assert "Empty2" in result["metadata"]["dataset_name"]

    def test_missing_metadata_fields(self):
        """Test handling of missing metadata fields."""
        meta1 = {}
        meta2 = {"dataset_name": "Dataset2"}

        merged = _merge_metadata(meta1, meta2)

        assert merged["version"] == "2.0"
        assert "Dataset1" in merged["dataset_name"]
        assert "Dataset2" in merged["dataset_name"]


class TestGenericMergeFunctions:
    """Tests for generic _merge_value and _merge_dict functions."""

    def test_merge_value_strict_vs_non_strict(self):
        """Test _merge_value behavior with strict parameter."""
        with pytest.raises(ValueError, match="test context: 'value1' vs 'value2'.*"):
            _merge_value("value1", "value2", "test context", strict=True)

        with patch("feedback_forensics.data.operations.merge.logger") as mock_logger:
            result = _merge_value("value1", "value2", "test context", strict=False)
            assert result == "value1"
            mock_logger.warning.assert_called_with(
                "test context: 'value1' vs 'value2', using first dataset value"
            )

    def test_merge_dict_strict_vs_non_strict(self):
        """Test _merge_dict behavior with strict parameter."""
        dict1 = {"key": "value1"}
        dict2 = {"key": "value2"}

        with pytest.raises(
            ValueError,
            match='Conflicting value for "key" key in test: "value1" vs "value2".*',
        ):
            _merge_dict(dict1, dict2, "test", strict=True)

        with patch("feedback_forensics.data.operations.merge.logger") as mock_logger:
            result = _merge_dict(dict1, dict2, "test", strict=False)
            assert result == {"key": "value1"}
            mock_logger.warning.assert_called_with(
                'Conflicting value for "key" key in test: "value1" vs "value2", using first dataset value'
            )

"""
Shared fixtures for test modules in the data package.
"""

import json
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def setup_test_data(tmp_path):
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
def setup_annotated_pairs_json(tmp_path):
    """Set up a temporary JSON file with AnnotatedPairs data for testing."""
    # Create test JSON data
    json_data = {
        "metadata": {
            "version": "1.0",
            "description": "AnnotatedPairs dataset with annotations from ICAI",
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
def setup_annotated_pairs_json_v2(tmp_path):
    """Set up a temporary JSON file with AnnotatedPairs data for testing format v2.0."""
    json_data = {
        "metadata": {
            "version": "2.0",
            "description": "AnnotatedPairs dataset with annotations from ICAI",
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

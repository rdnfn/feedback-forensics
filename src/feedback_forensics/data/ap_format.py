"""AnnotatedPairs (AP) format: creation, hashing, loading, saving and merging.

Ported from `inverse_cai.data.annotated_pairs_format` (inverse-cai v0.4.8) so that
standard annotation does not depend on the ICAI package. Hashing functions MUST stay
byte-for-byte identical to the ICAI implementation: comparison ids and annotator ids
of existing published datasets depend on them (pinned by golden values in
`ap_format_test.py`).
"""

import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd
from loguru import logger

# Constants
DEFAULT_ANNOTATOR_DESCRIPTION = (
    "Default annotator from original dataset (from column `preferred_text`)"
)
FORMAT_VERSION = "2.0"
DEFAULT_ANNOTATOR_TYPE = "unknown"
DEFAULT_PREFERENCE_KEY = "pref"
DEFAULT_PREFERENCE_COLUMN = "preferred_text"
PRINCIPLE_ANNOTATOR_TYPE = "principle"
DEFAULT_DATASET_DESCRIPTION = (
    "Annotated pairs dataset with annotations from Feedback Forensics"
)

PREFERENCE_ALIAS_MAPPING = {
    "a": "a",
    "b": "b",
    "text_a": "a",
    "text_b": "b",
}

# Special (non-a/b) vote values and the `no_pref_reason` they map to.
# "position_inconsistent" is only produced by position-debiased annotation
# (vote differs depending on which response is shown first).
SPECIAL_VOTE_REASONS = ("invalid", "both", "neither", "position_inconsistent")


def hash_string(s: str) -> str:
    """Create a shortened hash of a string."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def hash_comparison(
    response_a: Dict[str, Any], response_b: Dict[str, Any], prompt: Optional[str]
) -> str:
    """Create a hash ID for a comparison based on its content.

    Args:
        response_a: A dictionary with at least a 'text' key
        response_b: A dictionary with at least a 'text' key
        prompt: An optional prompt string
    """
    # Serialize each response dictionary to JSON to include all fields
    response_a_str = json.dumps(response_a, sort_keys=True)
    response_b_str = json.dumps(response_b, sort_keys=True)

    combined = f"{response_a_str}|{response_b_str}"
    if prompt is not None:
        combined = f"{prompt}|{combined}"
    return hash_string(combined)


def votes_to_annotations(
    votes: Mapping[int, Union[bool, str, None]],
    principle_index_to_text: Mapping[int, str],
    active_principles: Sequence[str],
    reference_preference: str,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Convert principle votes to annotations in the standardized format.

    Args:
        votes: Dictionary mapping principle IDs to their votes:
            - True: Principle agrees with reference preference
            - False: Principle disagrees with reference preference
            - None: Principle not applicable to this comparison
            - one of SPECIAL_VOTE_REASONS (case-insensitive), e.g. "invalid"
        principle_index_to_text: Dictionary mapping principle IDs to their text
        active_principles: List of active principles to include
        reference_preference: The reference preference ("a" or "b")

    Returns:
        Dictionary mapping principle IDs (hashed) to dictionaries with preference information:
        - For True/False votes: {"pref": "a" or "b"}
        - For None votes: {"pref": None, "no_pref_reason": "not_applicable"}
        - For special votes: {"pref": None, "no_pref_reason": <reason>}
    """
    annotations = {}

    for principle_idx, vote in votes.items():
        principle_text = principle_index_to_text[principle_idx]

        # Only include principles that are active
        if principle_text in active_principles:
            principle_id = hash_string(principle_text)

            if vote is None:
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: None,
                    "no_pref_reason": "not_applicable",
                }
            elif vote is True:
                # Principle agrees with reference preference
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: reference_preference
                }
            elif vote is False:
                # Principle disagrees with reference preference
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: (
                        "b" if reference_preference == "a" else "a"
                    )
                }
            elif isinstance(vote, str) and vote.lower() in SPECIAL_VOTE_REASONS:
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: None,
                    "no_pref_reason": vote.lower(),
                }
            else:
                raise ValueError(
                    f"Unexpected vote value: {vote} (type: {type(vote)}) for principle {principle_text}.\n\nTaken from votes: {votes} (type: {type(votes)})"
                )

    return annotations


def add_annotators(
    output: Dict,
    principles: List[str] | None = None,
    additional_columns: List[str] | None = None,
) -> None:
    """Add all annotators to the output structure (in-place).

    Adds annotator information to output["annotators"] and sets the default annotator.

    Args:
        output: The output dataset dictionary to modify in-place
        principles: principles to add as annotators
        additional_columns: additional columns from the data to include as annotators
    """
    default_annotator_id = hash_string(DEFAULT_ANNOTATOR_DESCRIPTION)
    output["annotators"][default_annotator_id] = {
        "name": DEFAULT_PREFERENCE_COLUMN,
        "description": DEFAULT_ANNOTATOR_DESCRIPTION,
        "type": DEFAULT_ANNOTATOR_TYPE,
    }
    output["metadata"]["default_annotator"] = default_annotator_id

    if principles is not None:
        for principle in principles:
            annotator_id = hash_string(principle)
            output["annotators"][annotator_id] = {
                "description": principle,
                "type": PRINCIPLE_ANNOTATOR_TYPE,
            }

    if additional_columns:
        for col in additional_columns:
            column_annotator_id = hash_string(f"column_{col}")
            # Set type to "human" if "human" is in the column name
            annotator_type = "human" if "human" in col.lower() else "unknown"
            output["annotators"][column_annotator_id] = {
                "name": col,
                "description": f"Column from original dataset: {col}",
                "type": annotator_type,
            }


def detect_annotator_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that appear to be annotator columns in the DataFrame.

    A column is considered an annotator column if it is not a standard column,
    has at most 10 unique values and contains "text_a" or "text_b".
    """
    standard_columns = {
        "text_a",
        "text_b",
        "model_a",
        "model_b",
        DEFAULT_PREFERENCE_COLUMN,
    }
    potential_annotators = []

    for col in df.columns:
        if col in standard_columns:
            continue

        unique_values = df[col].dropna().unique()
        if len(unique_values) <= 10:
            values_set = set(str(v).lower() for v in unique_values if pd.notna(v))
            if "text_a" in values_set or "text_b" in values_set:
                potential_annotators.append(col)

    return potential_annotators


def create_annotated_pairs(
    df: pd.DataFrame,
    dataset_name: str,
    principle_index_to_text: Mapping[int, str] | None = None,
    comparison_votes: (
        Mapping[int, Dict[int, Union[bool, str, None]]] | pd.Series | None
    ) = None,
    additional_columns: List[str] | None = None,
    auto_detect_annotators: bool = True,
    ignored_columns: tuple = ("principles", "prompt_principles"),
    description: str = DEFAULT_DATASET_DESCRIPTION,
) -> Dict:
    """Create an AnnotatedPairs dataset from a dataframe and (optional) principle votes.

    Args:
        df: DataFrame with preference data pairs. Must have "text_a", "text_b" and
            DEFAULT_PREFERENCE_COLUMN columns, and an optional "input" (prompt) column.
        dataset_name: Name for the dataset
        principle_index_to_text: Mapping of principle ids (used in votes) to principle text
        comparison_votes: Mapping of df index -> {principle id -> vote}
        additional_columns: Additional columns to include as annotators
        auto_detect_annotators: Whether to automatically detect annotator columns
        ignored_columns: Columns to not add as metadata
        description: Dataset description stored in the metadata

    Returns:
        The annotated pairs format as a dictionary
    """
    output = {
        "metadata": {
            "version": FORMAT_VERSION,
            "description": description,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_name": dataset_name,
        },
        "annotators": {},
        "comparisons": [],
    }

    df = df.copy()
    df = df.astype(str)  # ensure all columns are hashable

    detected_columns = []
    if auto_detect_annotators:
        detected_columns = detect_annotator_columns(df)
        if detected_columns:
            logger.info(f"Automatically detected annotator columns: {detected_columns}")

    all_additional_columns = list(set((additional_columns or []) + detected_columns))

    if (principle_index_to_text is None) != (comparison_votes is None):
        raise ValueError(
            "Got values for principles or comparison_votes, but not both. "
            "Please provide both or neither."
        )
    active_principles = (
        []
        if principle_index_to_text is None
        else list(principle_index_to_text.values())
    )

    add_annotators(
        output,
        principles=active_principles,
        additional_columns=all_additional_columns,
    )

    # Metadata columns: not standard, not annotators, not ignored
    standard_columns = {
        "text_a",
        "text_b",
        "model_a",
        "model_b",
        "input",
        DEFAULT_PREFERENCE_COLUMN,
    }
    metadata_columns = [
        col
        for col in df.columns
        if (
            col not in standard_columns
            and col not in all_additional_columns
            and col not in ignored_columns
        )
    ]

    if metadata_columns:
        output["metadata"]["available_metadata_keys_per_comparison"] = metadata_columns
        logger.info(f"Available metadata columns: {metadata_columns}")

    default_annotator_id = output["metadata"]["default_annotator"]

    for idx, row in df.iterrows():
        # Use original texts if prompts were merged into the texts
        text_a = row["_og_text_a"] if "_og_text_a" in row else row["text_a"]
        text_b = row["_og_text_b"] if "_og_text_b" in row else row["text_b"]

        response_a = {"text": text_a}
        response_b = {"text": text_b}

        if "model_a" in row and pd.notna(row["model_a"]):
            response_a["model"] = row["model_a"]
        if "model_b" in row and pd.notna(row["model_b"]):
            response_b["model"] = row["model_b"]

        comparison_id = hash_comparison(response_a, response_b, row.get("input"))

        annotations = {}
        reference_preference = row[DEFAULT_PREFERENCE_COLUMN]
        assert (
            reference_preference in PREFERENCE_ALIAS_MAPPING.keys()
        ), f"Invalid reference preference: {reference_preference}"
        reference_preference = PREFERENCE_ALIAS_MAPPING[reference_preference]

        annotations[default_annotator_id] = {
            DEFAULT_PREFERENCE_KEY: reference_preference
        }

        if comparison_votes is not None:
            if idx in comparison_votes:
                principle_annotations = votes_to_annotations(
                    votes=comparison_votes[idx],
                    principle_index_to_text=principle_index_to_text,
                    active_principles=active_principles,
                    reference_preference=reference_preference,
                )
                annotations.update(principle_annotations)
            else:
                logger.warning(
                    f"Missing votes for comparison with index {idx}, skipping principle annotations"
                )

        for col in all_additional_columns:
            if col in row and pd.notna(row[col]):
                column_annotator_id = hash_string(f"column_{col}")
                annotations[column_annotator_id] = {
                    DEFAULT_PREFERENCE_KEY: str(row[col])
                }

        comparison = {
            "id": comparison_id,
            "prompt": row.get("input"),
            "response_a": response_a,
            "response_b": response_b,
            "annotations": annotations,
        }

        if metadata_columns:
            comparison["metadata"] = {}
            for col in metadata_columns:
                if col in row and pd.notna(row[col]):
                    comparison["metadata"][col] = str(row[col])

        output["comparisons"].append(comparison)

    return output


def _sanitize(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def save_annotated_pairs_to_file(
    annotated_pairs: Dict, output_file: str | Path
) -> None:
    """Save the annotated pairs to a JSON file."""
    output_file = Path(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            _sanitize(annotated_pairs), f, ensure_ascii=False, indent=2, allow_nan=False
        )
    logger.info(f"Created annotated pairs format dataset: {output_file}")
    logger.info(f"- Dataset contains {len(annotated_pairs['comparisons'])} comparisons")
    logger.info(f"- Dataset contains {len(annotated_pairs['annotators'])} annotators")


def load_annotated_pairs_from_file(annotated_pairs_file: str | Path) -> Dict:
    """Load an annotated pairs dataset from a JSON file."""
    with open(annotated_pairs_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Loading of standard preference data (CSV or AP JSON) into a dataframe.
# Ported from `inverse_cai.data.loader.standard`.
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["text_a", "text_b"]


def _switch_pref_labels_in_df(df: pd.DataFrame) -> pd.DataFrame:
    swap = {"text_a": "text_b", "text_b": "text_a"}
    df["preferred_text"] = df["preferred_text"].apply(lambda v: swap.get(v, v))
    return df


def load_preference_data(
    path: str | Path, switch_labels: bool = False, merge_prompts: bool = True
) -> pd.DataFrame:
    """Load standard preference data from a CSV or AnnotatedPairs JSON file.

    If a separate `prompt` column is present (and `merge_prompts`), the prompt is
    merged into the `text_a`/`text_b` columns as a chat conversation, and the
    original texts are kept in `_og_text_a`/`_og_text_b`.
    """
    path = Path(path)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        df = _load_df_from_ap_json(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset {path} is missing required columns: {missing_columns}"
        )

    if "prompt" in df.columns and merge_prompts:
        df = _add_prompt_to_texts(df)

    if switch_labels:
        df = _switch_pref_labels_in_df(df)

    return df


def _load_df_from_ap_json(path: str | Path) -> pd.DataFrame:
    """Load an AnnotatedPairs JSON file as a simple preference dataframe."""
    ap_data = load_annotated_pairs_from_file(path)

    default_annotator = ap_data["metadata"]["default_annotator"]

    ap_version = ap_data["metadata"]["version"]
    if ap_version in ["1.0"]:
        raise ValueError(
            f"Loading from AnnotatedPairs (AP) {ap_version} is currently not supported, only from AP v2.0 or higher."
        )

    rows = []
    for comparison in ap_data["comparisons"]:
        pref_text = comparison["annotations"].get(default_annotator, {}).get("pref")
        pref_text = pref_text.replace("a", "text_a").replace("b", "text_b")
        rows.append(
            {
                "id": comparison["id"],
                "prompt": comparison["prompt"],
                "text_a": comparison["response_a"]["text"],
                "text_b": comparison["response_b"]["text"],
                "model_a": comparison["response_a"].get("model", None),
                "model_b": comparison["response_b"].get("model", None),
                "preferred_text": pref_text,
            }
        )

    return pd.DataFrame(rows)


def _add_prompt_to_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the prompt column into the text columns as a chat conversation."""

    # sanity check (first 100 rows) whether prompt already included in responses
    prompt_in_response = df.iloc[:100].apply(
        lambda row: str(row["prompt"]) in str(row["text_a"])
        and str(row["prompt"]) in str(row["text_b"]),
        axis=1,
    )
    if prompt_in_response.all():
        logger.warning(
            "Prompt is always in the response. This is unexpected. "
            "This is likely because the prompt is already included in the text columns."
            "Skipping merging the prompt and text columns."
        )
        return df

    logger.info("Combining prompts and texts since separate prompt column provided.")

    df["_og_text_a"] = df["text_a"]
    df["_og_text_b"] = df["text_b"]

    for text_name in ["text_a", "text_b"]:
        df[text_name] = df[["prompt", text_name]].apply(
            lambda x: str(
                [
                    {"role": "user", "content": x["prompt"]},
                    {"role": "assistant", "content": x[text_name]},
                ]
            ),
            axis=1,
        )

    return df

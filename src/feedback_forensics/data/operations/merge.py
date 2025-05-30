"""Merge operation for AnnotatedPairs datasets."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from loguru import logger
import datetime

from inverse_cai.data.annotated_pairs_format import hash_comparison


def merge_ap(
    first_data: Dict[str, Any],
    second_data: Dict[str, Any],
    dataset_name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge two AnnotatedPairs datasets.

    Args:
        first_data: First AnnotatedPairs dataset (takes precedence in conflicts)
        second_data: Second AnnotatedPairs dataset
        dataset_name: Override dataset name for merged result
        description: Override description for merged result

    Returns:
        Merged AnnotatedPairs data structure

    Raises:
        ValueError: If inputs are invalid
    """
    logger.info(f"Merging AnnotatedPairs datasets")
    logger.info(
        f"First dataset: {len(first_data['comparisons'])} comparisons, {len(first_data['annotators'])} annotators"
    )
    logger.info(
        f"Second dataset: {len(second_data['comparisons'])} comparisons, {len(second_data['annotators'])} annotators"
    )

    paired, first_only, second_only = _categorize_comparisons(
        first_data["comparisons"], second_data["comparisons"]
    )

    logger.info(
        f"Found {len(paired)} matching comparisons, {len(first_only)} unique to first, {len(second_only)} unique to second"
    )

    merged_comparisons = []

    for first_idx, second_idx in paired:
        merged_comp = _merge_single_comparison(
            first_data["comparisons"][first_idx], second_data["comparisons"][second_idx]
        )
        merged_comparisons.append(merged_comp)

    for idx in first_only:
        merged_comparisons.append(first_data["comparisons"][idx])

    for idx in second_only:
        merged_comparisons.append(second_data["comparisons"][idx])

    merged_metadata = _merge_metadata(first_data["metadata"], second_data["metadata"])

    if dataset_name is not None:
        merged_metadata["dataset_name"] = dataset_name
    if description is not None:
        merged_metadata["description"] = description

    merged_data = {
        "metadata": merged_metadata,
        "annotators": _merge_annotators(
            first_data["annotators"], second_data["annotators"]
        ),
        "comparisons": merged_comparisons,
    }

    logger.info(
        f"Merged result: {len(merged_data['comparisons'])} comparisons, {len(merged_data['annotators'])} annotators"
    )

    return merged_data


def _categorize_comparisons(
    comps1: List[Dict], comps2: List[Dict]
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Categorize comparisons between datasets by ID overlap.

    Returns:
        Tuple of:
        - paired: List of (first_idx, second_idx) tuples for matches
        - first_only: List of indices only in first dataset
        - second_only: List of indices only in second dataset
    """
    first_id_map = {}
    second_id_map = {}

    for i, comp in enumerate(comps1):
        comp_id = comp.get("id")
        if comp_id is None:
            comp_id = hash_comparison(
                comp.get("response_a", {}),
                comp.get("response_b", {}),
                comp.get("prompt"),
            )
            comp["id"] = comp_id
        first_id_map[comp_id] = i

    for i, comp in enumerate(comps2):
        comp_id = comp.get("id")
        if comp_id is None:
            comp_id = hash_comparison(
                comp.get("response_a", {}),
                comp.get("response_b", {}),
                comp.get("prompt"),
            )
            comp["id"] = comp_id
        second_id_map[comp_id] = i

    paired = []
    first_ids = set(first_id_map.keys())
    second_ids = set(second_id_map.keys())

    for comp_id in first_ids & second_ids:
        paired.append((first_id_map[comp_id], second_id_map[comp_id]))

    first_only = [first_id_map[comp_id] for comp_id in first_ids - second_ids]
    second_only = [second_id_map[comp_id] for comp_id in second_ids - first_ids]

    return (paired, first_only, second_only)


def _merge_single_comparison(comp1: Dict, comp2: Dict) -> Dict:
    """Merge two comparisons, first takes precedence in conflicts."""
    merged = {
        "id": comp1.get("id") or comp2.get("id"),
        "prompt": _merge_prompt(comp1, comp2),
        "response_a": _merge_response(comp1, comp2, "response_a"),
        "response_b": _merge_response(comp1, comp2, "response_b"),
        "annotations": _merge_annotations(comp1, comp2),
        "metadata": _merge_comparison_metadata(comp1, comp2),
    }

    # Drop None values returned by the individual merge calls
    return {k: v for k, v in merged.items() if v is not None}


def _merge_value(value1: Any, value2: Any, context: str, strict: bool) -> Any:
    """Generic value merging with conflict checking."""
    if value1 is None:
        return value2
    if value2 is None:
        return value1
    if value1 == value2:
        return value1

    base_msg = f"{context}: '{value1}' vs '{value2}'"
    if strict:
        raise ValueError(f"{base_msg}. No mismatch allowed in this field.")
    else:
        logger.warning(f"{base_msg}, using first dataset value")

    return value1


def _merge_dict(dict1: Dict, dict2: Dict, context: str, strict: bool) -> Dict:
    """Generic dictionary merging with optional strict conflict checking."""
    for key in dict1.keys() & dict2.keys():
        if dict1[key] != dict2[key]:
            base_msg = f'Conflicting value for "{key}" key in {context}: "{dict1[key]}" vs "{dict2[key]}"'
            if strict:
                raise ValueError(f"{base_msg}. No mismatch allowed in this field.")
            else:
                logger.warning(f"{base_msg}, using first dataset value")

    return {**dict2, **dict1}


def _merge_prompt(comp1: Dict, comp2: Dict) -> Optional[str]:
    """Merge prompt strings from two comparisons, raising on conflicts."""
    comp_id = comp1.get("id") or comp2.get("id")
    return _merge_value(
        comp1.get("prompt"),
        comp2.get("prompt"),
        f"Prompt conflict in comparison {comp_id}",
        strict=True,
    )


def _merge_response(comp1: Dict, comp2: Dict, field_name: str) -> Optional[Dict]:
    """Merge response objects from two comparisons, raising on conflicts."""
    resp1 = comp1.get(field_name)
    resp2 = comp2.get(field_name)

    if resp1 is None:
        return resp2
    if resp2 is None:
        return resp1

    comp_id = comp1.get("id") or comp2.get("id")
    return _merge_dict(resp1, resp2, f"comparison {comp_id} {field_name}", strict=True)


def _merge_annotations(comp1: Dict, comp2: Dict) -> Dict:
    """Merge annotation dictionaries from two comparisons, warning on conflicts."""
    ann1 = comp1.get("annotations", {})
    ann2 = comp2.get("annotations", {})
    comp_id = comp1.get("id") or comp2.get("id")

    return _merge_dict(ann1, ann2, f'comparison "{comp_id}" annotations', strict=False)


def _merge_comparison_metadata(comp1: Dict, comp2: Dict) -> Dict:
    """Merge comparison-level metadata from two comparisons, detecting conflicts."""
    meta1 = comp1.get("metadata", {})
    meta2 = comp2.get("metadata", {})
    comp_id = comp1.get("id") or comp2.get("id")

    return _merge_dict(meta1, meta2, f'comparison dictionary "{comp_id}"', strict=False)


def _merge_annotators(ann1: Dict, ann2: Dict) -> Dict:
    """Merge annotator definitions, first takes precedence."""
    return _merge_dict(ann1, ann2, "annotator definitions dictionary", strict=False)


def _merge_name_or_description(
    value1: Optional[str], value2: Optional[str]
) -> Optional[str]:
    """Merge name or description fields, reusing if identical or creating combined name."""
    if value1 is None:
        return value2
    if value2 is None:
        return value1
    if value1 == value2:
        return value1
    return f"Merged: {value1} + {value2}"


def _merge_default_annotator(
    annotator1: Optional[str], annotator2: Optional[str]
) -> Optional[str]:
    """Merge default annotator fields with first precedence and warning on disagreement."""
    return _merge_value(
        annotator1, annotator2, "Default annotator conflict", strict=False
    )


def _merge_metadata(meta1: Dict, meta2: Dict) -> Dict:
    """Create merged metadata."""
    version1 = meta1.get("version", "2.0")
    version2 = meta2.get("version", "2.0")
    assert (
        version1 == version2
    ), f"Dataset versions must match: {version1} vs {version2}"

    merged_metadata = {
        "version": version1,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_name": _merge_name_or_description(
            meta1.get("dataset_name") or "Dataset1",
            meta2.get("dataset_name") or "Dataset2",
        ),
        "description": _merge_name_or_description(
            meta1.get("description"), meta2.get("description")
        ),
        "default_annotator": _merge_default_annotator(
            meta1.get("default_annotator"), meta2.get("default_annotator")
        ),
    }

    for key, value in meta1.items():
        if key not in merged_metadata:
            merged_metadata[key] = value

    # Drop None values returned by the individual merge calls
    return {k: v for k, v in merged_metadata.items() if v is not None}

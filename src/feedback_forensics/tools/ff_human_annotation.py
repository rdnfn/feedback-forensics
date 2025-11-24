import argparse
import pathlib
from typing import Any, Dict, List, Tuple
import copy
import random
import json
from datetime import datetime, timezone

import gradio as gr
from loguru import logger

from feedback_forensics.data.operations.core import load_ap, save_ap
from inverse_cai.data.annotated_pairs_format import hash_string
from feedback_forensics.app.constants import DEFAULT_ANNOTATOR_HASH


PERSONALITY_TRAITS_DEFAULT: List[str] = (
    [  # top and bottom 5 traits from LMArena in og experiments
        "is more verbose",
        "has more structured formatting",
        "makes more confident statements",
        "is more factually correct",
        "more strictly follows the requested output format",
        "is more concise",
        "has a more avoidant tone",
        "refuses to answer the question",
        "ends with a follow-up question",
        "is more polite",
        "ISSUE_LANGUAGE",
    ]
)


def _ensure_trait_annotators_exist(
    ap: Dict[str, Any], traits: List[str], rater: str
) -> Dict[str, str]:
    """Ensure that an annotator entry exists for every trait.

    Returns mapping from trait -> annotator_id (stable, hashed).

    Note: 'rater' refers to the human doing the rating/annotation work. We use 'rater'
    rather than 'annotator' to avoid confusion with the trait-level annotators created
    from this rater (one rater produces multiple trait-level annotators).
    """
    annotators: Dict[str, Any] = ap.setdefault("annotators", {})
    trait_to_annotator_id: Dict[str, str] = {}

    for trait in traits:
        description = f"Human ({rater}): {trait}"
        annotator_id = hash_string(description)
        trait_to_annotator_id[trait] = annotator_id

        if annotator_id not in annotators:
            annotators[annotator_id] = {
                "description": description,
                "type": "principle",
            }

    return trait_to_annotator_id


def _get_text_from_response(response: Any) -> str:
    """Extract displayable text for a single response from AnnotatedPairs.

    Supports both v1.0 (string fields) and v2.0 (dict response fields).
    """
    # v1.0 already provides a string
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        # Common keys we may find in v2.0
        for key in ("text", "output", "message", "content"):
            if key in response and isinstance(response[key], str):
                return response[key]
        # Fallback to a compact string representation
        return str(response)

    return str(response)


def _read_pair_texts(comparison: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (prompt, text_a, text_b) for a comparison in v1.0 or v2.0."""
    prompt = comparison.get("metadata", {}).get("prompt", "")

    if "response_a" in comparison and "response_b" in comparison:
        # v2.0 style
        text_a = _get_text_from_response(comparison["response_a"])
        text_b = _get_text_from_response(comparison["response_b"])
    else:
        # v1.0 style
        text_a = comparison.get("text_a", "")
        text_b = comparison.get("text_b", "")

    return prompt, text_a, text_b


def _annotation_from_value(value: str) -> str:
    """Map radio value to AnnotatedPairs 'pref' string.

    Radio values: 'text_a', 'text_b', 'not relevant'
    """
    value_str = str(value).strip().lower()
    if value_str in ("text_a", "a"):
        return "a"
    if value_str in ("text_b", "b"):
        return "b"
    return "irrelevant"


def _value_from_annotation(pref: str) -> str:
    """Inverse mapping to populate radio controls from existing annotations."""
    pref_l = (pref or "").lower()
    return {"a": "text_a", "b": "text_b"}.get(pref_l, "not relevant")


def _save(ap: Dict[str, Any], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ap(ap, output_path)
    logger.info(f"Saved annotations to: {output_path}")


def _log_event(event_log_path, event_type, **data) -> None:
    event = {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    with open(event_log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _compute_stats(
    event_log_path: pathlib.Path, total_comparisons: int, rater: str
) -> str:
    """Compute annotation statistics from event log, filtered by rater.

    Returns markdown string with progress and timing stats.
    """
    try:
        events: List[Dict[str, Any]] = []
        with open(event_log_path) as f:
            for line in f:
                if len(line.strip()) > 0:
                    events.append(json.loads(line))
    except FileNotFoundError:
        return "**Progress**: No events yet."

    last_load_time = None
    last_save_time = None
    time_diffs = []

    # Compute delta between load and save.
    for event in events:
        event_type = event["event"]
        event_rater = event["rater"]

        if event_rater != rater:
            continue

        timestamp = datetime.fromisoformat(event["timestamp"])

        if event_type == "comparison_loaded":
            if last_load_time is not None and last_save_time is not None:
                time_diffs.append((last_save_time - last_load_time).total_seconds())

            last_load_time = timestamp
            last_save_time = None

        elif event_type == "comparison_saved":
            last_save_time = timestamp

    annotated_count = len(time_diffs)
    progress_pct = int(100 * annotated_count / total_comparisons)

    if len(time_diffs) > 0:
        avg_time_secs = sum(time_diffs) / len(time_diffs)
        time_str = f"{avg_time_secs:.1f}s"
        remaining_str = f"{int((total_comparisons - annotated_count) * avg_time_secs)}s"
    else:
        time_str = "N/A"
        remaining_str = "N/A"

    return f"""**Progress**

- **Annotated**: {annotated_count} / {total_comparisons} ({progress_pct}%)
- **Avg time/comparison**: {time_str}
- **Est. time remaining**: {remaining_str}"""


def _is_annotated(
    comp_id: str,
    output_comparisons: Dict[str, Any],
    trait_to_annotator_id: Dict[str, str],
) -> bool:
    """Check if a comparison has any trait annotations."""
    if comp_id not in output_comparisons:
        return False
    annotations = output_comparisons[comp_id].get("annotations", {})
    for annotator_id in trait_to_annotator_id.values():
        if annotator_id in annotations:
            return True
    return False


def _find_unannotated(
    start_idx: int,
    input_comparisons_ordered: List[Dict[str, Any]],
    output_comparisons: Dict[str, Any],
    trait_to_annotator_id: Dict[str, str],
) -> int:
    """
    Find next unannotated comparison starting after start_idx.
    Returns a valid index clamped to the input_comparisons_ordered list bounds.
    """
    idx = start_idx
    for _ in range(len(input_comparisons_ordered)):
        idx = (idx + 1) % len(input_comparisons_ordered)
        comp_id = input_comparisons_ordered[idx]["id"]
        if not _is_annotated(comp_id, output_comparisons, trait_to_annotator_id):
            return idx
    return start_idx


def build_interface(
    input_path: pathlib.Path,
    output_path: pathlib.Path | None,
    rater: str,
    traits: List[str] | None = None,
    use_standard_traits: bool = False,
) -> gr.Blocks:
    ap: Dict[str, Any] = load_ap(input_path)

    # Get traits from principle annotators generated by FF/ICAI
    if traits is None and use_standard_traits:
        traits = PERSONALITY_TRAITS_DEFAULT
    elif traits is None:
        traits = [
            annotator["description"]
            for annotator in ap["annotators"].values()
            if annotator["type"] == "principle"
        ]

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_human_ap.json")
    if pathlib.Path(output_path).exists():
        logger.info(
            f"Loading existing human-annotated AnnotatedPairs from {output_path}"
        )
        new_ap = load_ap(output_path)
    else:
        logger.info(
            f"Creating new AnnotatedPairs with human annotations at {output_path}"
        )
        new_ap: Dict[str, Any] = {
            "metadata": {
                "version": "2.0",
                "description": "AnnotatedPairs with human annotations for personality traits.",
                "dataset_name": "ff-model-personality",
                "default_annotator": DEFAULT_ANNOTATOR_HASH,
                "available_metadata_keys_per_comparison": ap.get(
                    "available_metadata_keys_per_comparison", []
                ),
            },
            "annotators": {},
            "comparisons": [],
        }
    comparisons: List[Dict[str, Any]] = ap["comparisons"]

    # randomize order of comparisons
    random.seed(42)
    random.shuffle(comparisons)

    # add default annotator
    if DEFAULT_ANNOTATOR_HASH not in new_ap["annotators"]:
        new_ap["annotators"][DEFAULT_ANNOTATOR_HASH] = ap["annotators"].get(
            DEFAULT_ANNOTATOR_HASH, {}
        )

    new_comparisons: Dict[str, Any] = {
        comp["id"]: comp for comp in new_ap["comparisons"]
    }
    trait_to_annotator_id = _ensure_trait_annotators_exist(new_ap, traits, rater)

    event_log_path = pathlib.Path(str(output_path) + ".events.jsonl")
    _log_event(
        event_log_path,
        "session_started",
        input_file=str(input_path),
        output_file=str(output_path),
        total_comparisons=len(comparisons),
        rater=rater,
    )

    # Save immediately to ensure annotators are present in the output file
    _save(new_ap, output_path)

    # Reset scroll to top in the textboxes when the content is mutated
    scroll_reset_js = """
    () => {
        const observer = new MutationObserver(() => {
            document.querySelectorAll('#text_a_box textarea, #text_b_box textarea').forEach(el => {
                el.scrollTop = 0;
            });
        });

        const config = { childList: true, subtree: true, characterData: true };
        ['text_a_box', 'text_b_box'].forEach(id => {
            const el = document.getElementById(id);
            if (el) observer.observe(el, config);
        });
    }
    """

    with gr.Blocks(
        title="Feedback Forensics: Human Trait Annotation", js=scroll_reset_js
    ) as demo:
        gr.Markdown(
            """
        ### Human annotation for personality traits

        Navigate pairs, set one label per trait. Each change is autosaved to the output file.
        """
        )

        with gr.Row():
            gr.Textbox(
                label="Input AnnotatedPairs (loaded)",
                value=str(input_path),
                interactive=False,
            )
            gr.Textbox(
                label="Output AnnotatedPairs (autosave)",
                value=str(output_path),
                interactive=False,
            )

        with gr.Row():
            # One row with two cols: Index and progress
            with gr.Column(scale=2):
                initial_idx = _find_unannotated(
                    -1,
                    comparisons,
                    new_comparisons,
                    trait_to_annotator_id,
                )
                idx_display = gr.Number(
                    label=f"Index (out of {len(comparisons)})",
                    value=initial_idx,
                    precision=0,
                    interactive=True,
                    container=False,
                )
            with gr.Column(scale=1):
                stats_display = gr.Markdown(
                    value=_compute_stats(event_log_path, len(comparisons), rater)
                )

        with gr.Row():
            btn_save_next = gr.Button("Save and Next")

        with gr.Group():
            prompt_md = gr.Textbox(label="Prompt", lines=4)
            with gr.Row():
                text_a_box = gr.Textbox(label="Text A", lines=10, elem_id="text_a_box")
                text_b_box = gr.Textbox(label="Text B", lines=10, elem_id="text_b_box")

        # Dynamic controls per trait
        trait_controls: Dict[str, gr.components.Component] = {}
        with gr.Group():
            gr.Markdown("#### Trait annotations")
            for trait in traits:
                ctrl = gr.Radio(
                    choices=["text_a", "not relevant", "text_b"],
                    value="not relevant",
                    label=f"{trait}",
                )
                trait_controls[trait] = ctrl

        def load_index(i: int) -> List[Any]:
            comp = comparisons[i]
            comp_id = comp["id"]
            prompt, text_a, text_b = _read_pair_texts(comp)

            _log_event(
                event_log_path,
                "comparison_loaded",
                comparison_id=comp_id,
                comparison_index=i,
                rater=rater,
            )

            if comp_id not in new_comparisons:
                # Add new comparison without original annotations
                new_comp = copy.deepcopy(comp)
                new_comp["annotations"] = {
                    trait_to_annotator_id[trait]: {"pref": "irrelevant"}
                    for trait in traits
                }
                new_comparisons[comp_id] = new_comp
            else:
                new_comp = new_comparisons[comp_id]

            updates: List[Any] = [
                i,
                gr.update(
                    value=_compute_stats(event_log_path, len(comparisons), rater)
                ),
                gr.update(value=prompt, visible=bool(prompt)),
                text_a,
                text_b,
            ]

            annotations: Dict[str, Any] = new_comparisons.get(comp["id"], {}).get(
                "annotations", {}
            )

            for trait in traits:
                annotator_id = trait_to_annotator_id[trait]
                existing = annotations.get(annotator_id, {}).get("pref")
                updates.append(gr.update(value=_value_from_annotation(existing)))

            # Update index display first, then prompt/texts, then trait controls
            return updates

        def on_save_and_next(i: int, *trait_values: str):
            from_index = int(i)
            comp = comparisons[from_index]
            comp_id = comp["id"]
            new_comp = new_comparisons[comp_id]

            # Build annotations for ALL traits from current control values
            all_annotations: Dict[str, Any] = {}
            for trait_name, value in zip(traits, trait_values):
                annotator_id = trait_to_annotator_id[trait_name]
                new_pref = _annotation_from_value(value)
                all_annotations[annotator_id] = {"pref": new_pref}

            all_annotations[DEFAULT_ANNOTATOR_HASH] = {
                "pref": comp["annotations"]
                .get(DEFAULT_ANNOTATOR_HASH, {})
                .get("pref", "irrelevant")
            }

            new_comparisons[comp_id]["annotations"] = all_annotations
            new_ap["comparisons"] = list(new_comparisons.values())
            _save(new_ap, output_path)

            _log_event(
                event_log_path,
                "comparison_saved",
                comparison_id=comp_id,
                rater=rater,
            )

            to_index = _find_unannotated(
                from_index,
                comparisons,
                new_comparisons,
                trait_to_annotator_id,
            )
            _log_event(
                event_log_path,
                "next_clicked",
                from_index=from_index,
                to_index=to_index,
                rater=rater,
            )
            return load_index(to_index)

        # Outputs list: idx_display, prompt_md, text_a_box, text_b_box, then one per trait
        output_components: List[gr.components.Component] = [
            idx_display,
            stats_display,
            prompt_md,
            text_a_box,
            text_b_box,
        ] + [trait_controls[t] for t in traits]

        trait_inputs: List[gr.components.Component] = [
            trait_controls[t] for t in traits
        ]

        click_fn = getattr(btn_save_next, "click")
        click_fn(
            on_save_and_next,
            inputs=[idx_display] + trait_inputs,
            outputs=output_components,
        )

        # Log trait changes
        def on_trait_change_log(i: int, *trait_values: str):
            idx = max(0, min(int(i), len(comparisons) - 1))
            comp = comparisons[idx]
            comp_id = comp["id"]

            old_annotations = new_comparisons[comp_id].get("annotations", {})

            for trait_name, value in zip(traits, trait_values):
                annotator_id = trait_to_annotator_id[trait_name]
                new_pref = _annotation_from_value(value)
                old_pref = old_annotations.get(annotator_id, {}).get("pref")
                if old_pref != new_pref:
                    # This is always triggered for all traits. Only log the actually changed one.
                    _log_event(
                        event_log_path,
                        "trait_changed",
                        comparison_id=comp_id,
                        trait=trait_name,
                        old_value=old_pref,
                        new_value=new_pref,
                        rater=rater,
                    )

        # Wire trait controls to log changes
        for _trait_name, ctrl in trait_controls.items():
            ctrl.select(
                on_trait_change_log,
                inputs=[idx_display] + trait_inputs,
                outputs=[],
            )

        def on_index_change(new_idx):
            idx = max(0, min(int(new_idx), len(comparisons) - 1))
            return load_index(idx)

        idx_display.change(
            on_index_change,
            inputs=[idx_display],
            outputs=output_components,
        )

        # Initialize first example
        def on_page_load():
            initial_idx = _find_unannotated(
                -1,
                comparisons,
                new_comparisons,
                trait_to_annotator_id,
            )
            return load_index(initial_idx)

        load_fn = getattr(demo, "load")
        load_fn(
            on_page_load,
            inputs=[],
            outputs=output_components,
        )

    return demo


def run():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal Gradio interface to add human personality-trait annotations "
            "to an AnnotatedPairs JSON."
        )
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to input AnnotatedPairs JSON file",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="Path to output AnnotatedPairs JSON (autosaved). Defaults to <input>_traits.json",
    )
    parser.add_argument(
        "--traits",
        type=str,
        default=None,
        help="Comma-separated list of traits to annotate, e.g. 'Politeness,Helpfulness'",
    )
    parser.add_argument(
        "--use-standard-traits",
        action="store_true",
        help="Use most important traits on LMArena dataset",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the Gradio interface with the world",
        default=False,
    )
    parser.add_argument(
        "--rater",
        type=str,
        required=True,
        help="Identifier of the human rater (used to track annotations per person)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the interface on (default: 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1). Use 0.0.0.0 for network accessibility",
    )

    args = parser.parse_args()

    demo = build_interface(
        args.input, args.out, args.rater, args.traits, args.use_standard_traits
    )
    demo.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    run()

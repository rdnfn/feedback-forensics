import argparse
import pathlib
from typing import Any, Dict, List, Tuple
import copy

import gradio as gr
from loguru import logger

from feedback_forensics.data.operations.core import load_ap, save_ap
from inverse_cai.data.annotated_pairs_format import hash_string


PERSONALITY_TRAITS_DEFAULT: List[str] = [
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness",
]


def _ensure_trait_annotators_exist(
    ap: Dict[str, Any], traits: List[str]
) -> Dict[str, str]:
    """Ensure that an annotator entry exists for every trait.

    Returns mapping from trait -> annotator_id (stable, hashed).
    """
    annotators: Dict[str, Any] = ap.setdefault("annotators", {})
    trait_to_annotator_id: Dict[str, str] = {}

    for trait in traits:
        # Create a stable short annotator id from the trait name
        annotator_id = hash_string(f"trait::{trait}")[:8]
        trait_to_annotator_id[trait] = annotator_id

        if annotator_id not in annotators:
            annotators[annotator_id] = {
                "name": trait,
                "description": f"Personality trait annotation for {trait}",
                "type": "trait",
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


def build_interface(
    input_path: pathlib.Path,
    traits: List[str],
    output_path: pathlib.Path | None,
) -> gr.Blocks:
    ap: Dict[str, Any] = load_ap(input_path)
    new_ap: Dict[str, Any] = {
        "metadata": {
            "version": "2.0",
            "description": "AnnotatedPairs with human annotations for personality traits.",
            "dataset_name": "ff-model-personality",
            "available_metadata_keys_per_comparison": ap.get(
                "available_metadata_keys_per_comparison", []
            ),
        },
        "annotators": {},
        "comparisons": [],
    }
    comparisons: List[Dict[str, Any]] = ap.get("comparisons", [])
    new_comparisons: Dict[str, Any] = {}
    trait_to_annotator_id = _ensure_trait_annotators_exist(new_ap, traits)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_traits.json")

    # Save immediately to ensure annotators are present in the output file
    _save(new_ap, output_path)

    with gr.Blocks(title="Feedback Forensics: Human Trait Annotation") as demo:
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
            idx_display = gr.Number(
                label="Index (out of {len(comparisons)})",
                value=0,
                precision=0,
                interactive=False,
                container=False,
            )

        with gr.Row():
            btn_prev = gr.Button("Prev")
            btn_next = gr.Button("Next")

        with gr.Group():
            prompt_md = gr.Textbox(label="Prompt", lines=4)
            with gr.Row():
                text_a_box = gr.Textbox(label="Text A", lines=10)
                text_b_box = gr.Textbox(label="Text B", lines=10)

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
            i = max(0, min(i, len(comparisons) - 1))
            comp = comparisons[i]
            prompt, text_a, text_b = _read_pair_texts(comp)

            updates: List[Any] = [i, gr.update(value=prompt), text_a, text_b]

            annotations: Dict[str, Any] = comp.get("annotations", {})
            for trait in traits:
                annotator_id = trait_to_annotator_id[trait]
                existing = annotations.get(annotator_id, {}).get("pref")
                updates.append(gr.update(value=_value_from_annotation(existing)))

            # Update index display first, then prompt/texts, then trait controls
            return updates

        # Wire navigation
        def on_prev(i):
            return load_index(int(i) - 1)

        def on_next(i):
            return load_index(int(i) + 1)

        # Outputs list: idx_display, prompt_md, text_a_box, text_b_box, then one per trait
        output_components: List[gr.components.Component] = [
            idx_display,
            prompt_md,
            text_a_box,
            text_b_box,
        ] + [trait_controls[t] for t in traits]

        # Bind using attribute lookup to appease static type checkers
        click_fn = getattr(btn_prev, "click")
        click_fn(
            on_prev,
            inputs=[idx_display],
            outputs=output_components,
        )
        click_fn = getattr(btn_next, "click")
        click_fn(
            on_next,
            inputs=[idx_display],
            outputs=output_components,
        )

        # Autosave handlers for each trait control
        def make_on_change(trait_name: str):
            annotator_id = trait_to_annotator_id[trait_name]

            def on_change(i: int, value: str):
                idx = max(0, min(int(i), len(comparisons) - 1))
                comp = comparisons[idx]
                comp_id = comp["id"]
                if comp_id not in new_comparisons:
                    # add new comaprison without original annotations
                    new_comp = copy.deepcopy(comp)
                    new_comp["annotations"] = {}
                    new_comparisons[comp_id] = new_comp
                else:
                    new_comp = new_comparisons[comp_id]

                new_comparisons[comp_id]["annotations"][annotator_id] = {
                    "pref": _annotation_from_value(value)
                }

                new_ap["comparisons"] = list(new_comparisons.values())
                _save(new_ap, output_path)
                return

            return on_change

        for trait_name, ctrl in trait_controls.items():
            handler = make_on_change(trait_name)
            ctrl.change(
                handler,
                inputs=[idx_display, ctrl],
                outputs=[],
            )

        # Initialize first example
        load_fn = getattr(demo, "load")
        load_fn(
            load_index,
            inputs=[idx_display],
            outputs=output_components,
        )

    return demo


def run():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal Gradio interface to add human personality-trait annotations "
            "to an AnnotatedPairs JSON (ICAI v2.0)."
        )
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to input AnnotatedPairs JSON file",
    )
    parser.add_argument(
        "--traits",
        type=str,
        default=",".join(PERSONALITY_TRAITS_DEFAULT),
        help=(
            "Comma-separated list of traits to annotate, e.g. 'Politeness,Helpfulness'"
        ),
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="Path to output AnnotatedPairs JSON (autosaved). Defaults to <input>_traits.json",
    )

    args = parser.parse_args()

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    if not traits:
        raise ValueError(
            "No traits specified. Provide --traits with at least one trait."
        )

    demo = build_interface(args.input, traits, args.out)
    demo.launch()


if __name__ == "__main__":
    run()

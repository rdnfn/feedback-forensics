import gradio as gr
import pandas as pd

from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    VERSION,
    PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS,
    EXAMPLE_VIEWER_NO_DATA_MESSAGE,
    ENABLE_EXAMPLE_VIEWER,
    EXAMPLE_BASE_URL,
    DEFAULT_SHOWN_METRIC,
)
from feedback_forensics.data.datasets import (
    get_available_datasets,
    get_default_dataset_names,
    get_available_datasets_names,
)
from feedback_forensics.app.info_texts import (
    get_datasets_description,
)

from feedback_forensics.app.styling import CUSTOM_CSS, THEME
from feedback_forensics.app.utils import get_gradio_image_path
from feedback_forensics.app.metrics import get_default_avail_metrics
import feedback_forensics.app.callbacks
from loguru import logger


def _add_title_row(title: str):
    """Add a title row to the interface.

    Args:
        title (str): Title text to display
    """
    with gr.Row(elem_classes="title-row"):
        gr.Markdown(f"## {title}")


def _create_header():
    """Create the app header with logo and links."""
    image_path = get_gradio_image_path("feedback_forensics_logo.png")
    link_button_variant = "secondary"
    link_button_size = "md"

    with gr.Row(variant="default"):
        with gr.Column(scale=4, min_width="300px"):
            link_style = "opacity: 0.9; color: var(--body-text-color); text-decoration: none; background-color: var(--button-secondary-background-fill); padding: 4px"
            text_style = "opacity: 0.5"
            image = f'<img src="{image_path}" alt="Logo" width="330">'
            spacer = f'<span style="{text_style}"> | </span>'
            text_powered_by = f'<span style="{text_style}">Powered by the <a href="https://github.com/rdnfn/icai" style="opacity: 0.9; color: var(--body-text-color);">Inverse Constitutional AI</a> (ICAI) pipeline</span>'
            text_version = f'<span style="{text_style}">v{VERSION}</span>'
            link_github = f'<a href="https://github.com/rdnfn/feedback-forensics" style="{link_style}">🌟&nbsp;Star&nbsp;on&nbsp;GitHub</a>'
            link_report_bug = f'<a href="https://github.com/rdnfn/feedback-forensics/issues/new?template=Blank+issue" style="{link_style}">✍️&nbsp;Report&nbsp;bug</a>'
            link_get_in_touch = f'<a href="mailto:forensics@arduin.io" style="{link_style}">📮&nbsp;Get&nbsp;in&nbsp;touch</a>'
            gr.HTML(
                image
                + '<div style="margin-left: 20px; margin-top: 5px; padding-bottom: 10px; font-size: 1.2em; line-height: 1.8;">'
                + text_version
                + spacer
                + text_powered_by
                + '<div style="float: right;">'
                + link_github
                + spacer
                + link_report_bug
                + spacer
                + link_get_in_touch
                + "</div>"
                + "</div>",
                padding=False,
            )


def _create_getting_started_section():
    button_size = "sm"
    with gr.Accordion("👋 Getting started: pre-configured examples", open=True):
        with gr.Row(equal_height=True):
            tutorial_domain = EXAMPLE_BASE_URL  # make this "" to use local instance
            gr.Button(
                "🤖 Example 1: Compare GPT-4o's personality to other models",
                size=button_size,
                link=f"{tutorial_domain}?data=lmarena_2024&ann_cols=model_gpt4o20240513,model_claude35sonnet20240620,model_gemini15proapi0514,model_mistrallarge2407,model_deepseekv2api0628",
            )
            gr.Button(
                "📚 Example 2: Personality traits encouraged by feedback datasets",
                size=button_size,
                link=f"{tutorial_domain}?data=lmarena_2024,alpacaeval,prism,anthropic_helpful,anthropic_harmless",
            )
            gr.Button(
                "📝 Example 3: Preferred personality traits across writing tasks",
                link=f"{tutorial_domain}?data=lmarena_2024&col=narrower_category&col_vals=songwriting_prompts,resume_and_cover_letter_writing,professional_email_communication,creative_writing_prompts&analysis_mode=advanced_settings",
                size=button_size,
            )


def _initialize_state(state: dict):
    """Initialize the state of the app."""
    available_datasets = get_available_datasets()
    state["app_url"] = gr.State(value="")
    state["datapath"] = gr.State(value="")
    state["df"] = gr.State(value=pd.DataFrame())
    state["unfiltered_df"] = gr.State(value=pd.DataFrame())
    state["dataset_name"] = gr.State(value="")
    state["active_dataset"] = gr.State(value="")
    state["cache"] = gr.State(value={})
    state["avail_datasets"] = gr.State(
        value={dataset.name: dataset for dataset in available_datasets}
    )
    state["computed_annotator_metrics"] = gr.State(value={})
    state["computed_overall_metrics"] = gr.State(value={})
    state["default_annotator_cols"] = gr.State(value=[])
    state["default_annotator_rows"] = gr.State(value=[])
    state["votes_dicts"] = gr.State(value={})
    return state


def _create_configuration_panel(inp: dict, state: dict):
    """Create the configuration panel of the interface."""

    _add_title_row("Configuration")

    with gr.Row(variant="panel", render=True):
        with gr.Column():
            # Get dataset names and set default value safely
            dataset_names = get_available_datasets_names()
            default_datasets = get_default_dataset_names()

            with gr.Group():
                inp["active_datasets_dropdown"] = gr.Dropdown(
                    label="💽 Dataset selection",
                    choices=dataset_names,
                    value=(
                        default_datasets[0] if default_datasets else None
                    ),  # Single selection by default
                    interactive=True,
                    multiselect=False,  # Default to single selection
                )

                with gr.Accordion("ℹ️ Dataset details", open=False):
                    datasets = get_available_datasets()
                    inp["dataset_info_v2"] = gr.Markdown(
                        get_datasets_description(datasets),
                        container=True,
                    )

            with gr.Group():
                inp["analysis_type_radio"] = gr.Radio(
                    label="🔎 Analysis mode",
                    choices=[
                        ("👥 Human/AI feedback analysis", "annotation_analysis"),
                        ("🤖 Model analysis", "model_analysis"),
                        ("🔧 Advanced settings", "advanced_settings"),
                    ],
                    value="annotation_analysis",
                    interactive=True,
                )

                # single dataset configuration
                inp["multi_dataset_warning_md"] = gr.Markdown(
                    value="<div style='opacity: 0.6'>⚠️ <i>Some configuration options (grouping by column, selecting multiple col annotators) only work correctly when selecting a single dataset. Select a single dataset to use these features.</i></div>",
                    visible=True,
                    container=True,
                )
                inp["models_to_compare_dropdown"] = gr.Dropdown(
                    label="📌 Select models to compare",
                    info="Select model(s) to investigate in terms of personality traits.",
                    choices=None,
                    value=None,
                    multiselect=True,
                )
                inp["reference_models_dropdown"] = gr.Dropdown(
                    label="🧭 Select reference models",
                    info="Select *reference* models to compare *selected* models to. Metrics can be interpreted as how much the selected model(s) exhibit(s) personality traits relative to the reference model(s). **If none are selected, all available models will be used as references.** *Example:* With GPT-4o as *reference* model, personality traits of selected models are computed relative to GPT-4o, i.e. only using datapoints directly comparing selected models with GPT-4o.",
                    choices=None,
                    value=None,
                    multiselect=True,
                )
                inp["annotations_to_compare_dropdown"] = gr.Dropdown(
                    label="🗂️ Select feedback annotations to compare (AI or human)",
                    info="Analyse personality traits encouraged by different pairwise feedback annotations",
                    choices=None,
                    value=None,
                    multiselect=True,
                )
                inp["enable_dataviewer_checkbox"] = gr.Checkbox(
                    label="Enable dataviewer (experimental)",
                    info="Enable the dataviewer to view individual datapoints by clicking on the annotator table. This is experimental.",
                    value=ENABLE_EXAMPLE_VIEWER,
                    interactive=True,
                    visible=False,
                )

                inp["enable_multiple_datasets_checkbox"] = gr.Checkbox(
                    label="Enable multiple dataset selection",
                    info="Allow selecting multiple datasets simultaneously. Some features (like column grouping) only work with single datasets.",
                    value=False,
                    interactive=True,
                    visible=False,
                )

                inp["split_col_dropdown"] = gr.Dropdown(
                    label="🗃️ Group dataset by column",
                    info="Create separate results for data subsets grouped by this column's values. If no column is selected, entire original dataset will be analyzed. ",
                    choices=[NONE_SELECTED_VALUE],
                    value=NONE_SELECTED_VALUE,
                    interactive=False,
                    visible=False,
                )
                inp["split_col_selected_vals_dropdown"] = gr.Dropdown(
                    label="🏷️ Column values to show",
                    info="For each selected value, separate results will be created. If no values selected, all values will be used. Requires column to be selected above.",
                    choices=[],
                    value=None,
                    multiselect=True,
                    interactive=False,
                    visible=False,
                )
                inp["annotator_cols_dropdown"] = gr.Dropdown(
                    label="👥→ Annotator columns",
                    info="Select the annotators to be included as a column in the results table. By default only a single (ground-truth) annotator is included.",
                    choices=None,
                    value=None,
                    multiselect=True,
                )
                inp["annotator_rows_dropdown"] = gr.Dropdown(
                    label="👥↓ Annotator rows",
                    info=f'Select the annotators to be included as a row in the results table. By default only objective-following AI annotators are included (named as "{PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS} \\<OBJECTIVE\\>").',
                    choices=None,
                    value=None,
                    multiselect=True,
                )

            # final button to run analysis
            inp["load_btn"] = gr.Button("Run analysis", variant="secondary")


def _create_numerical_results_panel(inp: dict, out: dict):

    inp["numerical_results_col"] = gr.Column(scale=1, variant="panel")

    with inp["numerical_results_col"]:

        gr.Markdown("## Numerical overview")

        gr.Markdown(
            "### Overall statistics\nSee [guide here](https://feedback-forensics.readthedocs.io/en/latest/method/metrics#general-statistics) for metric details"
        )
        out["overall_metrics_table"] = gr.Dataframe(
            value=pd.DataFrame(),
            headers=["No data loaded"],
        )
        gr.Markdown(
            "---\n### Annotation metrics\n👉 *Click on values to view example datapoints* | See [guide here](https://feedback-forensics.readthedocs.io/en/latest/method/metrics.html) to learn how each metric is computed and can be interpreted"
        )

        with gr.Group():
            # Add control dropdowns for the annotator table
            with gr.Row():
                inp["metric_name_dropdown"] = gr.Dropdown(
                    label="Metric",
                    choices=get_default_avail_metrics(),
                    value=DEFAULT_SHOWN_METRIC,
                    interactive=True,
                )
                inp["sort_by_dropdown"] = gr.Dropdown(
                    label="Sort by",
                    choices=None,
                    value=None,
                    interactive=True,
                )
                inp["sort_order_dropdown"] = gr.Dropdown(
                    label="Sort order",
                    choices=["Descending", "Ascending"],
                    value="Descending",
                    interactive=True,
                )

            out["annotator_table"] = gr.Dataframe(
                value=pd.DataFrame(),
                headers=["No data loaded"],
                interactive=False,
            )


def _create_example_viewer(inp: dict, out: dict):
    """Create viewer for individual datapoints as examples."""

    inp["example_view_col"] = gr.Column(variant="panel")
    with inp["example_view_col"]:

        gr.Markdown("## Datapoint viewer")

        gr.Markdown("### Controls")
        # Input controls
        with gr.Group():

            # dataset dropdown is hidden by default
            # to simplify the interface
            inp["example_dataset_dropdown"] = gr.Dropdown(
                label="📊 Dataset",
                choices=[],
                value=None,
                interactive=True,
                visible=False,
            )

            with gr.Row():
                inp["example_annotator_1"] = gr.Dropdown(
                    label="👥 Annotator 1",
                    choices=[],
                    value=None,
                    interactive=True,
                )

                inp["example_annotator_2"] = gr.Dropdown(
                    label="👥 Annotator 2",
                    choices=[],
                    value=None,
                    interactive=True,
                )

            inp["example_subset_dropdown"] = gr.Dropdown(
                label="🔍 Filter subset",
                choices=[
                    ("All", "all"),
                    ("Agree", "agree"),
                    ("Disagree", "disagree"),
                    (
                        "Only annotator 1 does not apply",
                        "only annotator 1 does not apply",
                    ),
                    (
                        "Only annotator 2 does not apply",
                        "only annotator 2 does not apply",
                    ),
                    ("Neither apply", "neither apply"),
                ],
                value="all",
                interactive=True,
            )

            inp["example_index_slider"] = gr.Slider(
                label="📋 Example index",
                minimum=0,
                maximum=100,
                step=1,
                value=0,
                interactive=True,
            )

        # Output displays
        gr.Markdown("### Datapoint")
        out["example_message"] = gr.Markdown(
            EXAMPLE_VIEWER_NO_DATA_MESSAGE,
            visible=False,
        )
        out["example_details_group"] = gr.Group(
            visible=False,
        )
        with out["example_details_group"]:

            out["example_comparison_id"] = gr.Textbox(
                label="🏷️ Comparison ID",
                value="",
                interactive=False,
            )

            out["example_prompt"] = gr.Textbox(
                label="💬 Prompt",
                value="",
                interactive=False,
                type="text",
                lines=5,
                max_lines=20,
            )

            with gr.Row():
                out["example_response_a_model"] = gr.Textbox(
                    label="🤖 Model A",
                    value="",
                    interactive=False,
                    type="text",
                )
                out["example_response_b_model"] = gr.Textbox(
                    label="🤖 Model B",
                    value="",
                    interactive=False,
                    type="text",
                )

            with gr.Row():
                out["example_response_a"] = gr.Textbox(
                    label="📝 Response A",
                    value="",
                    interactive=False,
                    type="text",
                    lines=10,
                    max_lines=20,
                )

                out["example_response_b"] = gr.Textbox(
                    label="📝 Response B",
                    value="",
                    interactive=False,
                    type="text",
                    lines=10,
                    max_lines=20,
                )

            out["example_annotator_1_result"] = gr.Textbox(
                label="👥 Annotator 1 preference",
                value="",
                interactive=False,
            )

            out["example_annotator_2_result"] = gr.Textbox(
                label="👥 Annotator 2 preference",
                value="",
                interactive=False,
            )

            out["example_metadata"] = gr.JSON(
                label="📋 Metadata",
                value={},
            )


def _create_results_panel(inp: dict, out: dict):

    _add_title_row("Results")

    with gr.Row():
        inp["results_view_radio"] = gr.Radio(
            label="🎛️ View",
            choices=[
                ("📊 Numerical overview", "numerical_results"),
                ("🔎 Datapoint viewer", "example_viewer"),
            ],
            value="numerical_results",
            interactive=True,
        )
        out["share_link"] = gr.Textbox(
            label="🔗 Share link",
            value="",
            show_copy_button=True,
            scale=2,
            interactive=True,
            show_label=True,
        )

    _create_numerical_results_panel(inp, out)
    _create_example_viewer(inp, out)


def _force_dark_theme(block):
    block.load(
        None,
        None,
        js="""
  () => {
  const params = new URLSearchParams(window.location.search);
  if (!params.has('__theme')) {
    params.set('__theme', 'dark');
    window.location.search = params.toString();
  }
  }""",
    )


def generate():

    inp = {}
    state = {}
    out = {}

    with gr.Blocks(theme=THEME, css=CUSTOM_CSS) as demo:

        state = _initialize_state(state)

        # _force_dark_theme(demo)
        _create_header()
        _create_getting_started_section()
        _create_configuration_panel(inp, state)
        _create_results_panel(inp, out)

        with gr.Row():
            gr.HTML(f"<center>Feedback Forensics app v{VERSION}</center>")

        callbacks = feedback_forensics.app.callbacks.generate(inp, state, out)
        feedback_forensics.app.callbacks.attach(inp, state, out, callbacks, demo)

    return demo

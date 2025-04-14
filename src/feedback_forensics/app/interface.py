import gradio as gr
import pandas as pd

from feedback_forensics.app.callbacks import generate_callbacks, attach_callbacks
from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    VERSION,
    PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS,
)
from feedback_forensics.app.datasets import (
    get_available_datasets,
    get_default_dataset_names,
    get_available_datasets_names,
)
from feedback_forensics.app.info_texts import (
    get_datasets_description,
    METRICS_DESCRIPTION,
)

from feedback_forensics.app.styling import CUSTOM_CSS, THEME
from feedback_forensics.app.utils import get_gradio_image_path
from feedback_forensics.app.metrics import METRIC_COL_OPTIONS


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
            text_version = (
                f'<span style="{text_style}">v{VERSION} (Alpha Preview)</span>'
            )
            link_github = f'<a href="https://github.com/rdnfn/feedback-forensics" style="{link_style}">📁&nbsp;GitHub</a>'
            link_report_bug = f'<a href="https://github.com/rdnfn/feedback-forensics/issues/new?template=Blank+issue" style="{link_style}">✍️&nbsp;Report&nbsp;bug</a>'
            link_get_in_touch = f'<a href="mailto:forensics@arduin.io" style="{link_style}">✉️&nbsp;Get&nbsp;in&nbsp;touch</a>'
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
            tutorial_domain = "https://app.feedbackforensics.com"  # make this "" to use local instance
            gr.Button(
                "🤖 Example 1: How is GPT-4o different to other models?",
                size=button_size,
                link=f"{tutorial_domain}?data=chatbot_arena&col=winner_model&col_vals=gpt4o20240513,claude35sonnet20240620,gemini15proapi0514,mistrallarge2407,deepseekv2api0628",
            )
            gr.Button(
                "📚 Example 2: How do popular preference datasets differ?",
                size=button_size,
                link=f"{tutorial_domain}?data=chatbot_arena,alpacaeval,prism,anthropic_helpful,anthropic_harmless",
            )
            gr.Button(
                "📝 Example 3: How do user preferences vary across writing tasks?",
                link=f"{tutorial_domain}?data=chatbot_arena&col=narrower_category&col_vals=songwriting_prompts,resume_and_cover_letter_writing,professional_email_communication,creative_writing_prompts",
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
            with gr.Group():
                # Get dataset names and set default value safely
                dataset_names = get_available_datasets_names()
                default_datasets = get_default_dataset_names()

                inp["active_datasets_dropdown"] = gr.Dropdown(
                    label="💽 Active datasets",
                    choices=dataset_names,
                    value=default_datasets,
                    interactive=True,
                    multiselect=True,
                )
                with gr.Accordion("ℹ️ Dataset details", open=False):
                    datasets = get_available_datasets()
                    inp["dataset_info_v2"] = gr.Markdown(
                        get_datasets_description(datasets),
                        container=True,
                    )

                # single dataset configuration
                inp["split_col_non_available_md"] = gr.Markdown(
                    value="<div style='opacity: 0.6'><i>Some configuration options (grouping by column, selecting annotators) are only available when selecting a single dataset. Select a single dataset to use these features.</i></div>",
                    visible=True,
                    container=True,
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
                    info="For each selected value, separate results will be created. If no values selected, all values will be used.",
                    choices=[],
                    value=None,
                    multiselect=True,
                    interactive=False,
                    visible=False,
                )
                inp["advanced_settings_accordion"] = gr.Accordion(
                    "🔧 Advanced settings", open=False
                )
                with inp["advanced_settings_accordion"]:
                    inp["annotator_cols_dropdown"] = gr.Dropdown(
                        label="👥→ Annotator columns",
                        info="Select the annotators to be included as a column in the results table. By default only a single (ground-truth) annotator is included.",
                        choices=None,
                        value=None,
                        multiselect=True,
                    )
                    inp["annotator_rows_dropdown"] = gr.Dropdown(
                        label="👥↓ Annotator rows",
                        info=f'Select the annotators to be included as a row in the results table. By default only objective-following AI annotators are included (named as "{PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS} \<OBJECTIVE\>").',
                        choices=None,
                        value=None,
                        multiselect=True,
                    )

                # final button to run analysis
                inp["load_btn"] = gr.Button("Run analysis", variant="secondary")


def _create_results_panel(inp: dict, out: dict):

    _add_title_row("Results")
    with gr.Column(scale=1, variant="panel"):
        with gr.Group():
            out["share_link"] = gr.Textbox(
                label="🔗 Share link",
                value="",
                show_copy_button=True,
                scale=1,
                interactive=True,
                show_label=True,
            )

        gr.Markdown("### Overall metrics")
        out["overall_metrics_table"] = gr.Dataframe(
            value=pd.DataFrame(),
            headers=["No data loaded"],
        )
        gr.Markdown("### Annotation metrics")

        with gr.Group():
            # Add control dropdowns for the annotator table
            with gr.Row():
                inp["metric_name_dropdown"] = gr.Dropdown(
                    label="Metric",
                    choices=list(METRIC_COL_OPTIONS.keys()),
                    value="strength",
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

            with gr.Accordion("ℹ️ Metrics explanation", open=False):
                out["metrics_info"] = gr.Markdown(
                    METRICS_DESCRIPTION,
                    container=True,
                )

            out["annotator_table"] = gr.Dataframe(
                value=pd.DataFrame(),
                headers=["No data loaded"],
            )


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

        callbacks = generate_callbacks(inp, state, out)
        attach_callbacks(inp, state, out, callbacks, demo)

    return demo

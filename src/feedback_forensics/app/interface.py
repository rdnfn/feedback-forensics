import gradio as gr
import pandas as pd
import json

from feedback_forensics.app.callbacks import generate_callbacks, attach_callbacks
from feedback_forensics.app.constants import (
    NONE_SELECTED_VALUE,
    VERSION,
    ALLOW_LOCAL_RESULTS,
    DEFAULT_DATASET_PATH,
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
from feedback_forensics.app.metrics import METRIC_COL_OPTIONS
from feedback_forensics.app.styling import CUSTOM_CSS, THEME
from feedback_forensics.app.utils import get_gradio_image_path


def add_title_row(title: str):
    """Add a title row to the interface.

    Args:
        title (str): Title text to display
    """
    with gr.Row(elem_classes="title-row"):
        gr.Markdown(f"## {title}")


def create_header():
    """Create the app header with logo and links."""
    image_path = get_gradio_image_path("feedback_forensics_logo.png")
    link_button_variant = "secondary"
    link_button_size = "md"

    with gr.Row(variant="default"):
        with gr.Column(scale=4, min_width="300px"):
            link_style = "opacity: 0.9; color: white; text-decoration: none; background-color: #404040; padding: 4px"
            text_style = "opacity: 0.5"
            image = f'<img src="{image_path}" alt="Logo" width="330">'
            spacer = f'<span style="{text_style}"> | </span>'
            text_powered_by = f'<span style="{text_style}">Powered by the <a href="https://github.com/rdnfn/icai" style="opacity: 0.9; color: white;">Inverse Constitutional AI</a> (ICAI) pipeline</span>'
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


def create_getting_started_section():
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


def create_data_loader(inp: dict, state: dict):
    """Create the data loader section of the interface."""
    # Get the current list of available datasets
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

    create_getting_started_section()

    add_title_row("Configuration")

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
                inp["split_col_non_available_md"] = gr.Markdown(
                    value="<div style='opacity: 0.6'><i>Grouping dataset by the values of a column is only available when selecting a single dataset. Select a single dataset to use this feature.</i></div>",
                    visible=True,
                    container=True,
                )
                inp["load_btn"] = gr.Button("Run analysis", variant="secondary")

    # TODO: remove old dataset selection panel (including from callbacks etc.)
    with gr.Row(variant="panel", render=False):
        with gr.Column(scale=3):
            with gr.Accordion("Select dataset to analyze"):
                inp["dataset_btns"] = {}
                for dataset in available_datasets:
                    inp["dataset_btns"][dataset.name] = gr.Button(
                        dataset.name, variant="secondary"
                    )

        with gr.Column(scale=3):
            with gr.Accordion("Add your own dataset", visible=ALLOW_LOCAL_RESULTS):
                with gr.Group():
                    inp["datapath"] = gr.Textbox(
                        label="💾 Path",
                        value=DEFAULT_DATASET_PATH,
                    )
                    # inp["load_btn"] = gr.Button("Load")

    # TODO: remove old config panel (including from callbacks etc.)
    inp["config"] = gr.Row(visible=True, variant="panel", render=False)
    with inp["config"]:
        with gr.Column(
            scale=3,
        ):
            inp["simple_config_dropdown_placeholder"] = gr.Markdown(
                "*No simple dataset configuration available. Load different dataset or use advanced config.*",
                container=True,
            )
            inp["simple_config_dropdown"] = gr.Dropdown(
                label="🔧 Data subset to analyze",
                # info='Show principles\' performance reconstructing ("explaining") the selected feedback subset. *Example interpretation: If the principle "Select the more concise response" reconstructs GPT-4 wins well, GPT-4 may be more concise than other models in this dataset.*',
                visible=False,
            )
        with gr.Column(
            scale=3,
        ):
            with gr.Accordion(label="⚙️ Advanced config", open=False, visible=True):
                gr.Markdown(
                    "Advanced configuration options that enable filtering the dataset and changing other visibility settings. If available, settings to the left of this menu will automatically set these advanced options. Set advanced options here manually to override."
                )
                with gr.Group():
                    # button to disable efficient
                    inp["show_individual_prefs_dropdown"] = gr.Dropdown(
                        label="🗂️ Show individual preferences (slow)",
                        info="Whether to show individual preference examples. May slow down the app.",
                        choices=[False, True],
                        value=False,
                        interactive=True,
                    )

                    inp["plot_col_name_dropdown"] = gr.Dropdown(
                        label="Show plot across values of column",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )
                    inp["plot_col_value_dropdown"] = gr.Dropdown(
                        label="Values to show (if none selected, all values are shown)",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=True,
                        multiselect=True,
                    )

                    inp["pref_order_dropdown"] = gr.Dropdown(
                        label="📊 Order of reconstructed preferences",
                        choices=[
                            "By reconstruction success",
                            "Original (random) order",
                        ],
                        value="By reconstruction success",
                        interactive=True,
                    )
                    metric_choices = [
                        (
                            f"{metric['name']}",
                            key,
                        )
                        for key, metric in METRIC_COL_OPTIONS.items()
                    ]
                    inp["metrics_dropdown"] = gr.Dropdown(
                        multiselect=True,
                        label="📈 Metrics to show",
                        choices=metric_choices,
                        value=["perf", "relevance", "acc"],
                        interactive=True,
                    )

                inp["filter_accordion"] = gr.Accordion(
                    label="🎚️ Filter 1", open=False, visible=True
                )
                with inp["filter_accordion"]:
                    inp["filter_col_dropdown"] = gr.Dropdown(
                        label="Filter by column",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )
                    # add equal sign between filter_dropdown and filter_text

                    inp["filter_value_dropdown"] = gr.Dropdown(
                        label="equal to",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )
                inp["filter_accordion_2"] = gr.Accordion(
                    label="🎚️ Filter 2", open=False, visible=True
                )
                with inp["filter_accordion_2"]:
                    inp["filter_col_dropdown_2"] = gr.Dropdown(
                        label="Filter by column",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )
                    # add equal sign between filter_dropdown and filter_text

                    inp["filter_value_dropdown_2"] = gr.Dropdown(
                        label="equal to",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )

    # TODO: remove old dataset description panel (including from callbacks etc.)
    with gr.Row(variant="panel", render=False):
        with gr.Accordion("ℹ️ Dataset description", open=True):
            inp["dataset_info"] = gr.Markdown("*No dataset loaded*", container=True)


def create_principle_view(out: dict):
    with gr.Row(variant="panel"):
        with gr.Group():
            out["share_link"] = gr.Textbox(
                label="🔗 Share link",
                value="",
                show_copy_button=True,
                scale=1,
                interactive=True,
                show_label=True,
            )
            with gr.Accordion("ℹ️ Metrics explanation", open=False):
                out["metrics_info"] = gr.Markdown(
                    METRICS_DESCRIPTION,
                    container=True,
                )
            out["plot"] = gr.Plot()


def force_dark_theme(block):
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

        force_dark_theme(demo)

        create_header()
        create_data_loader(inp, state)

        add_title_row("Results")
        create_principle_view(out)

        with gr.Row():
            gr.HTML(f"<center>Feedback Forensics app v{VERSION}</center>")

        callbacks = generate_callbacks(inp, state, out)
        attach_callbacks(inp, state, out, callbacks, demo)

    return demo

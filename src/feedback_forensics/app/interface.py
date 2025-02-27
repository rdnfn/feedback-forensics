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
from feedback_forensics.app.datasets import BUILTIN_DATASETS
from feedback_forensics.app.info_texts import (
    METHOD_INFO_TEXT,
    METHOD_INFO_HEADING,
    TLDR_TEXT,
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
            gr.HTML(
                (
                    f'<img src="{image_path}" alt="Logo" width="330">'
                    '<div style="margin-left: 20px; margin-top: 5px; font-size: 1.2em; line-height: 1.8;">'
                    '<span style="opacity: 0.3">'
                    # "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                    f"v{VERSION} (Alpha Preview) | </span>"
                    # "&nbsp;&nbsp;&nbsp;"
                    # "<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                    f'<a href="https://github.com/rdnfn/feedback-forensics" style="{link_style}">📁&nbsp;GitHub</a>'
                    # "&nbsp;&nbsp;&nbsp;"
                    '<span style="opacity: 0.3"> | </span>'
                    f'<a href="https://github.com/rdnfn/feedback-forensics/issues/new?template=Blank+issue" style="{link_style}">✍️&nbsp;Report&nbsp;bug</a>'
                    '<span style="opacity: 0.3">'
                    " | "
                    f"Powered by the <a href='https://github.com/rdnfn/icai' style='opacity: 0.9; color: white;'>Inverse Constitutional AI</a> (ICAI) pipeline</span>"
                    "</div>"
                ),
                padding=False,
            )


def create_data_loader(inp: dict, state: dict):
    """Create the data loader section of the interface."""
    state["app_url"] = gr.State(value="")
    state["datapath"] = gr.State(value="")
    state["df"] = gr.State(value=pd.DataFrame())
    state["unfiltered_df"] = gr.State(value=pd.DataFrame())
    state["dataset_name"] = gr.State(value="")
    state["active_dataset"] = gr.State(value="")
    state["cache"] = gr.State(value={})
    state["avail_datasets"] = gr.State(
        value={dataset.name: dataset for dataset in BUILTIN_DATASETS}
    )

    add_title_row("Data selection")
    with gr.Row(variant="panel", render=True):
        with gr.Column(scale=2):
            inp["active_datasets_dropdown"] = gr.Dropdown(
                label="Active datasets",
                choices=[dataset.name for dataset in BUILTIN_DATASETS],
                value=[BUILTIN_DATASETS[-1].name],
                interactive=True,
                multiselect=True,
            )
            inp["load_btn"] = gr.Button("Load")
        with gr.Column(scale=1):
            inp["split_col_dropdown"] = gr.Dropdown(
                label="Split by column",
                choices=[NONE_SELECTED_VALUE],
                value=NONE_SELECTED_VALUE,
                interactive=False,
                visible=False,
            )
            inp["split_col_selected_vals_dropdown"] = gr.Dropdown(
                label="Column values to show",
                choices=[],
                value=None,
                multiselect=True,
                interactive=False,
                visible=False,
            )
            inp["split_col_non_available_md"] = gr.Markdown(
                value="<div style='opacity: 0.5'>ℹ️ Splitting dataset by the values of a column is only available when selecting a single dataset. Select a single dataset to use this feature.</div>",
                visible=True,
            )

    # TODO: remove old dataset selection panel (including from callbacks etc.)
    with gr.Row(variant="panel", render=False):
        with gr.Column(scale=3):
            with gr.Accordion("Select dataset to analyze"):
                inp["dataset_btns"] = {}
                for dataset in BUILTIN_DATASETS:
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
    with gr.Row():
        with gr.Column(scale=2, variant="panel"):
            with gr.Group():
                out["share_link"] = gr.Textbox(
                    label="🔗 Share link",
                    value="",
                    show_copy_button=True,
                    scale=1,
                    interactive=True,
                    # container=False,
                    show_label=True,
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

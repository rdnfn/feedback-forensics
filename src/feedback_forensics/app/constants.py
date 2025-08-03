"""Constants for the app."""

import os
import json
import pathlib

# App/package version
import importlib.metadata

from inverse_cai.data.annotated_pairs_format import (
    hash_string,
    DEFAULT_ANNOTATOR_DESCRIPTION,
)

VERSION = importlib.metadata.version("feedback_forensics")

# Constants from environment vars
# Github token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Data directory, to clone web datasets to
DATA_DIR = pathlib.Path("forensics-data")

# Base url used for share link
APP_BASE_URL = os.getenv("FF_APP_BASE_URL", "")
# Base url used for example links
# (setting this to hf space url can help avoid re-opening a new tab)
EXAMPLE_BASE_URL = os.getenv("FF_EXAMPLE_BASE_URL", APP_BASE_URL)

# Default dataset names
DEFAULT_DATASET_NAMES = json.loads(os.getenv("FF_DEFAULT_DATASET_NAMES", "[]"))
# Whether to enable the example viewer
ENABLE_EXAMPLE_VIEWER = os.getenv("FF_ENABLE_EXAMPLE_VIEWER", "true").lower() == "true"
# Default metrics available (in non-full metric mode)
DEFAULT_AVAIL_METRICS = json.loads(
    os.getenv("FF_AVAIL_METRICS", '["strength", "relevance", "cohens_kappa"]')
)
# Metric selected by default
DEFAULT_SHOWN_METRIC = os.getenv("FF_SHOWN_METRIC", "strength")

# App username and password
# Will block app behind login if env vars are set
USERNAME = os.getenv("FF_APP_USER")
PASSWORD = os.getenv("FF_APP_PW")


# Text style
FONT_FAMILY = '"Open Sans", verdana, arial, sans-serif'
FONT_COLOR = "white"

# Writing
PREFIX_DEFAULT_ANNOTATOR = "Default: "
PREFIX_OTHER_ANNOTATOR_WITH_VARIANT = "{variant}: "
PREFIX_COL_ANNOTATOR = "Data Column: "
PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS = "AI: "
PREFIX_MODEL_IDENTITY_ANNOTATORS = "Model: "
MODEL_IDENTITY_ANNOTATOR_TYPE = "model_identity"
PRINCIPLE_ANNOTATOR_TYPE = "icai_principle"


### Colors
LIGHT_GREEN = "#d9ead3"
DARK_GREEN = "#38761d"
LIGHT_RED = "#f4cacb"
DARK_RED = "#a61d00"
LIGHTER_GREY = "#fafafa"
LIGHT_GREY = "#e4e4e7"
DARK_GREY = "#3f3f46"  # "rgba(192, 192, 192, 0.8)"
VERY_DARK_GREY = "#27272a"  # "rgba(48, 48, 48, 0.8)"
# used for the background of reconstruction votes
COLORS_DICT = {
    "Agree": "#93c37d",
    "Disagree": "#c17c92",
    "Not applicable": DARK_GREY,
    "Invalid": "black",
}
DARK_COLORS_DICT = {
    "Agree": DARK_GREEN,
    "Disagree": DARK_RED,
    "Not applicable": VERY_DARK_GREY,
}
PAPER_BACKGROUND_COLOR = "#27272a"  # "white"  # LIGHT_GREY
PLOT_BACKGROUND_COLOR = "#27272a"  # "white"  # LIGHT_GREY

NONE_SELECTED_VALUE = "(None selected)"
DEFAULT_ANNOTATOR_COL_NAME = "preferred_text"
DEFAULT_ANNOTATOR_VISIBLE_NAME = PREFIX_DEFAULT_ANNOTATOR + DEFAULT_ANNOTATOR_COL_NAME
DEFAULT_ANNOTATOR_HASH = hash_string(DEFAULT_ANNOTATOR_DESCRIPTION)

DISABLE_SKLEARN_WARNINGS = True


EXAMPLE_VIEWER_NO_DATA_MESSAGE = "⚠️ No examples found"
EXAMPLE_VIEWER_MULTIPLE_DATASETS_MESSAGE = (
    "⚠️ Multiple datasets selected. Select single dataset to view examples."
)

# Mode for Gradio web app (downloads special data)
# FOR INTERNAL USE ONLY: this won't work
# without the right HF_TOKEN and setup
WEBAPP_MODE = os.getenv("FF_WEBAPP_MODE", "False").lower() == "true"

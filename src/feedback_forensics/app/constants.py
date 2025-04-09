"""Constants for the app."""

import os
import json

# App/package version
import importlib.metadata

from inverse_cai.data.annotated_pairs_format import (
    hash_string,
    DEFAULT_ANNOTATOR_DESCRIPTION,
)

VERSION = importlib.metadata.version("feedback_forensics")

# Constants from environment vars
# get env var with github token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
APP_BASE_URL = os.getenv("FF_APP_BASE_URL")
DEFAULT_DATASET_NAMES = json.loads(os.getenv("FF_DEFAULT_DATASET_NAMES", "[]"))

# App username and password
# Will block app behind login if env vars are set
USERNAME = os.getenv("FF_APP_USER")
PASSWORD = os.getenv("FF_APP_PW")


# Text style
FONT_FAMILY = '"Open Sans", verdana, arial, sans-serif'
FONT_COLOR = "white"

# Writing
PREFIX_PRINICIPLE_FOLLOWING_ANNOTATORS = "AI: "


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
DEFAULT_ANNOTATOR_NAME = "preferred_text"
DEFAULT_ANNOTATOR_HASH = hash_string(DEFAULT_ANNOTATOR_DESCRIPTION)


# Plotly config
# values to be removed from the modebar]
# (tool bar that has no use for us, but is difficult to
# remove when working with Gradio and Plotly combined)
PLOTLY_MODEBAR_POSSIBLE_VALUES = [
    "autoScale2d",
    "autoscale",
    "editInChartStudio",
    "editinchartstudio",
    "hoverCompareCartesian",
    "hovercompare",
    "lasso",
    "lasso2d",
    "orbitRotation",
    "orbitrotation",
    "pan",
    "pan2d",
    "pan3d",
    "reset",
    "resetCameraDefault3d",
    "resetCameraLastSave3d",
    "resetGeo",
    "resetSankeyGroup",
    "resetScale2d",
    "resetViewMap",
    "resetViewMapbox",
    "resetViews",
    "resetcameradefault",
    "resetcameralastsave",
    "resetsankeygroup",
    "resetscale",
    "resetview",
    "resetviews",
    "select",
    "select2d",
    "sendDataToCloud",
    "senddatatocloud",
    "tableRotation",
    "tablerotation",
    "toImage",
    "toggleHover",
    "toggleSpikelines",
    "togglehover",
    "togglespikelines",
    "toimage",
    "zoom",
    "zoom2d",
    "zoom3d",
    "zoomIn2d",
    "zoomInGeo",
    "zoomInMap",
    "zoomInMapbox",
    "zoomOut2d",
    "zoomOutGeo",
    "zoomOutMap",
    "zoomOutMapbox",
    "zoomin",
    "zoomout",
]

"""Constants for the app."""

import os
import json

# App/package version
import importlib.metadata

VERSION = importlib.metadata.version("feedback_forensics")

DEFAULT_DATASET_PATH = "exp/outputs/prism_v2"

# Constants from environment vars
# get env var with github token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
APP_BASE_URL = os.getenv("FF_APP_BASE_URL")
DEFAULT_DATASET_NAMES = json.loads(os.getenv("FF_DEFAULT_DATASET_NAMES", "[]"))

# App username and password
# Will block app behind login if env vars are set
USERNAME = os.getenv("ICAI_APP_USER")
PASSWORD = os.getenv("ICAI_APP_PW")
ALLOW_LOCAL_RESULTS = os.getenv("ICAI_ALLOW_LOCAL_RESULTS", "true").lower() == "true"

### Layout and dimensions
PRINCIPLE_SHORT_LENGTH = 70  # length of principle shown before cutting off
# this sets where the actual plot starts and ends (individual datapoints)
END_RECONSTRUCTION_PLOT_X = 0.99

# px values for different components
FIG_HEIGHT_PER_PRINCIPLE = 20  # height of each principle in px
FIG_HEIGHT_HEADER = 45
FIG_HEIGHT_BOTTOM = 10
FIG_GAP_BETWEEN_TABLES = 50
FIG_GAP_BETWEEN_HEADER_AND_TABLE = 20

# columns size
SPACE_PER_NUM_COL = 0.05


def get_fig_proportions_y(num_principles: int, num_metrics: int):
    """Get the y-proportions for the figure for different components.

    This is all relative, with 0 being the bottom of the figure and 1 being the top."""

    metrics_table_height_px = (num_metrics + 2) * FIG_HEIGHT_PER_PRINCIPLE

    principles_height_px = FIG_HEIGHT_PER_PRINCIPLE * num_principles

    total_height_px = (
        FIG_HEIGHT_HEADER
        + metrics_table_height_px
        + FIG_GAP_BETWEEN_TABLES
        + principles_height_px
        + FIG_HEIGHT_BOTTOM
    )

    metrics_table_bottom_y_px = (
        FIG_HEIGHT_BOTTOM + principles_height_px + FIG_GAP_BETWEEN_TABLES
    )
    metrics_table_top_y_px = metrics_table_bottom_y_px + metrics_table_height_px

    table_top_y_px = FIG_HEIGHT_BOTTOM + principles_height_px
    table_bottom_y_px = FIG_HEIGHT_BOTTOM
    header_bottom_y_px = table_top_y_px + FIG_GAP_BETWEEN_HEADER_AND_TABLE

    return {
        "principle_table": {
            "relative": {
                "heading_y": header_bottom_y_px / total_height_px,
                "table_top_y": table_top_y_px / total_height_px,
                "table_bottom_y": table_bottom_y_px / total_height_px,
                "metrics_table_top_y": metrics_table_top_y_px / total_height_px,
                "metrics_table_bottom_y": metrics_table_bottom_y_px / total_height_px,
                "row_height": FIG_HEIGHT_PER_PRINCIPLE / total_height_px,
            },
            "absolute": {
                "heading_bottom_y": header_bottom_y_px,
                "table_top_y": table_top_y_px,
                "table_bottom_y": table_bottom_y_px,
                "total_height": total_height_px,
                "principles_height": principles_height_px,
            },
        }
    }


PRINCIPLE_END_X = 0.40
METRICS_START_X = PRINCIPLE_END_X + 0.01
MENU_X = 0.04


# Text style
FONT_FAMILY = '"Open Sans", verdana, arial, sans-serif'
FONT_COLOR = "white"


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

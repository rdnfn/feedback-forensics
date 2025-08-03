import argparse
import gradio.themes.utils.fonts
import gradio as gr
from loguru import logger

import feedback_forensics.app.interface as interface
from feedback_forensics.app.constants import USERNAME, PASSWORD, HF_TOKEN, WEBAPP_MODE
import feedback_forensics.data.datasets

# make gradio work offline
gradio.themes.utils.fonts.GoogleFont.stylesheet = lambda self: None


def run():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Launch the Feedback Forensics visualisation app."
    )
    parser.add_argument(
        "-d",
        "--datapath",
        type=str,
        action="append",
        help="Path to AnnotatedPairs dataset to analyse. Can be used multiple times (e.g. -d path1 -d path2).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        action="append",
        help="Path to directory containing AnnotatedPairs datasets to analyse. Will be recursively searched for .json files. Can be used multiple times (e.g. --dir dirpath1 --dir dirpath2).",
    )
    parser.add_argument(
        "--load-web-datasets",
        action="store_true",
        help="Load datasets from HuggingFace. Does not require HF_TOKEN to be set.",
    )

    args = parser.parse_args()

    # Try to load local datasets if provided
    if args.datapath:
        for datapath in args.datapath:
            local_dataset = feedback_forensics.data.datasets.get_dataset_from_ap_json(
                datapath
            )
            if local_dataset is not None:
                feedback_forensics.data.datasets.add_dataset(local_dataset)

    if args.dir:
        for dirpath in args.dir:
            datasets = feedback_forensics.data.datasets.get_datasets_from_dir(dirpath)
            for dataset in datasets:
                feedback_forensics.data.datasets.add_dataset(dataset)

    if args.load_web_datasets or WEBAPP_MODE:
        logger.info("Loading web datasets from HuggingFace...")
        feedback_forensics.data.datasets.load_standard_web_datasets()
    else:
        logger.info("Note: only local datasets will be loaded.")

    if WEBAPP_MODE:
        logger.info("Loading special webapp datasets from HuggingFace...")
        feedback_forensics.data.datasets.load_webapp_datasets()

    # Get the current available datasets
    available_datasets = feedback_forensics.data.datasets.get_available_datasets()

    if len(available_datasets) == 0:
        logger.error(
            "No datasets available. No local or standard datasets could be loaded. Please provide a path to a local dataset via --datapath (-d) flag (or alternatively enable loading online mode datasets via HF_TOKEN if you have permissions)."
        )
        return

    # Log available datasets before interface generation
    logger.info(
        f"Available datasets for interface: {[ds.name for ds in available_datasets]}"
    )

    # setup the gradio app
    demo = interface.generate()

    # run the app
    if USERNAME and PASSWORD:
        auth = (USERNAME, PASSWORD)
        auth_message = "Welcome to the ICAI App demo!"
    else:
        auth = None
        auth_message = None

    demo.launch(auth=auth, auth_message=auth_message)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

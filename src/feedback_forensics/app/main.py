import argparse
import gradio.themes.utils.fonts
import gradio as gr
from loguru import logger

import feedback_forensics.app.interface as interface
from feedback_forensics.app.constants import USERNAME, PASSWORD, HF_TOKEN
import feedback_forensics.app.datasets

# make gradio work offline
gradio.themes.utils.fonts.GoogleFont.stylesheet = lambda self: None


def run():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", "-d", type=str, help="Path to dataset")
    args = parser.parse_args()

    # Try to load local dataset if provided
    if args.datapath:
        local_dataset = feedback_forensics.app.datasets.create_local_dataset(
            args.datapath
        )
        feedback_forensics.app.datasets.add_dataset(local_dataset)
        logger.info(f"Added local dataset to available datasets ({args.datapath}).")

    if HF_TOKEN:
        logger.info("HF_TOKEN found. Attempting to load HuggingFace datasets...")
        loaded_count = feedback_forensics.app.datasets.load_datasets_from_hf()
    else:
        logger.info("Note: only local datasets will be loaded.")

    # Get the current available datasets
    available_datasets = feedback_forensics.app.datasets.get_available_datasets()

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

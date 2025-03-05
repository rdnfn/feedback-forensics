import argparse
import gradio.themes.utils.fonts
import gradio as gr
from loguru import logger

import forensics.app.interface as interface
from forensics.app.constants import USERNAME, PASSWORD
import forensics.app.datasets

# make gradio work offline
gradio.themes.utils.fonts.GoogleFont.stylesheet = lambda self: None


def run():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", "-d", type=str, help="Path to dataset")
    args = parser.parse_args()
    if args.datapath:
        forensics.app.datasets.BUILTIN_DATASETS.append(
            forensics.app.datasets.create_local_dataset(args.datapath)
        )
        logger.info(f"Added local dataset to available datasets ({args.datapath}).")

    if len(forensics.app.datasets.BUILTIN_DATASETS) == 0:
        logger.error(
            "No datasets available. No local or standard datasets could be loaded. Please either provide a path to a local dataset via --datapath (-d) flag or provide the correct HuggingFace token via the HF_TOKEN environment variable."
        )
        return

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

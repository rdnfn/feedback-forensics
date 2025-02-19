import argparse
import gradio.themes.utils.fonts
import gradio as gr

import feedback_forensics.app.interface as interface
from feedback_forensics.app.constants import USERNAME, PASSWORD
import feedback_forensics.app.datasets

# make gradio work offline
gradio.themes.utils.fonts.GoogleFont.stylesheet = lambda self: None


def run():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", "-d", type=str, help="Path to dataset")
    args = parser.parse_args()
    if args.datapath:
        feedback_forensics.app.datasets.BUILTIN_DATASETS.append(
            feedback_forensics.app.datasets.create_local_dataset(args.datapath)
        )
        gr.Info(f"Added local dataset to available datasets ({args.datapath}).")

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

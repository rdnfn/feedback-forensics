import argparse
import gradio.themes.utils.fonts

import feedback_forensics.app.interface as interface
from feedback_forensics.app.constants import USERNAME, PASSWORD
import feedback_forensics.app.datasets

# make gradio work offline
gradio.themes.utils.fonts.GoogleFont.stylesheet = lambda self: None


demo = interface.generate()


def run():
    if USERNAME and PASSWORD:
        auth = (USERNAME, PASSWORD)
        auth_message = "Welcome to the ICAI App demo!"
    else:
        auth = None
        auth_message = None

    demo.launch(auth=auth, auth_message=auth_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add argument for dataset path (including short form -d)
    parser.add_argument("-d", "--datapath", type=str, help="Path to dataset")
    args = parser.parse_args()
    if args.datapath:
        feedback_forensics.app.datasets.BUILTIN_DATASETS.append(
            feedback_forensics.app.datasets.create_local_dataset(args.datapath)
        )
    run()  # pylint: disable=no-value-for-parameter

from feedback_forensics.app.main import run
import time
import subprocess
import pytest
import signal


def test_gradio_app_runs():

    # start gradio app via subprocess
    gradio_process = subprocess.Popen(
        [
            "python",
            "-m",
            "feedback_forensics.app.main",
            "--datapath",
            "data/output/example/results/070_annotations_train_ap.json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # This makes the output strings instead of bytes
    )

    # Wait for 10 seconds to allow the app to start
    time.sleep(10)

    # close the app - safely terminate only the process we started
    gradio_process.terminate()
    # Wait for the process to actually terminate (with timeout)
    try:
        returncode = gradio_process.wait(timeout=5)
        # Get the stdout and stderr from the process
        stdout, stderr = gradio_process.communicate()

        # Check if there were any errors
        # Note: Exit code -15 means the process was terminated by SIGTERM (our terminate() call)
        # which is expected behavior, not an error
        if returncode != 0 and returncode != -signal.SIGTERM:
            error_msg = f"Gradio process exited with error code: {returncode}"
            if stderr:
                error_msg += f"\nError output: {stderr}"
            pytest.fail(error_msg)

    except subprocess.TimeoutExpired:
        # Force kill if terminate() doesn't work
        gradio_process.kill()
        # Still get any output that was generated
        stdout, stderr = gradio_process.communicate()

        error_msg = "Gradio process did not terminate gracefully and had to be killed"
        if stderr:
            error_msg += f"\nError output from the killed process: {stderr}"
        pytest.fail(error_msg)

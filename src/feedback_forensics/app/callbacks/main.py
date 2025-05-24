"""Module to generate all callbacks for the app.

Combines all callbacks into a single dictionary.
"""

import feedback_forensics.app.callbacks.utils as utils
import feedback_forensics.app.callbacks.update_options as update_options
import feedback_forensics.app.callbacks.loading as loading


def generate(inp: dict, state: dict, out: dict) -> dict:
    """Create all callbacks for the app."""

    utils_callbacks = utils.generate(inp, state, out)

    update_options_callbacks = update_options.generate(
        inp,
        state,
        out,
        utils_callbacks=utils_callbacks,
    )

    loading_callbacks = loading.generate(
        inp,
        state,
        out,
        utils_callbacks=utils_callbacks,
        update_options_callbacks=update_options_callbacks,
    )

    return {
        **utils_callbacks,
        **update_options_callbacks,
        **loading_callbacks,
    }

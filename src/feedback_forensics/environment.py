"""Process-wide set-up, run once on `import feedback_forensics`.

- Environment variables (API keys, app settings such as FF_APP_USER) are loaded from
  a `secrets.toml` file in the working directory, if there is one.
- Logging (loguru) is set to INFO level on stderr, with a compact format.

Before annotation became native, both were side effects of importing inverse_cai
(`inverse_cai.local_secrets` and `inverse_cai.utils.setup_logger`).
"""

import os
import sys

import loguru
from dotenv import load_dotenv
from loguru import logger

SECRETS_PATH = "./secrets.toml"
LOG_FORMAT = "📜  | {level: <4} | <level>{message}</level>"
LOGURU_DEFAULT_HANDLER_ID = 0


def load_secrets(path: str = SECRETS_PATH) -> None:
    """Load environment variables from a dotenv-format file, if it exists.

    Variables that are already set in the environment take precedence. As by ICAI,
    a file that can't be loaded (e.g. no permission, not UTF-8, a null character)
    is skipped (with a warning), rather than making `import feedback_forensics`
    fail.
    """
    try:
        load_dotenv(path)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            f"Could not load environment variables from {path} (skipped): "
            f"{type(e).__name__}: {e}. Is it a readable, UTF-8 encoded text file?"
        )


def setup_logging() -> None:
    """Replace loguru's default handler (DEBUG level, verbose format) with an INFO
    level handler on stderr with a compact format.

    The LOGURU_LEVEL and LOGURU_FORMAT environment variables take precedence (also
    if set in secrets.toml), as do other LOGURU_* variables (e.g. LOGURU_COLORIZE;
    only if set in the environment, as loguru reads them when it is imported).
    Does nothing if logging was set up otherwise, e.g. before importing FF: if
    loguru's default handler was removed, or not added (LOGURU_AUTOINIT=False, or
    no `sys.stderr`, e.g. with pythonw).

    As the default handler is replaced (ICAI removed all handlers), code that
    calls `logger.remove(0)` after importing FF fails: use `logger.remove()`, or
    set up logging before importing FF.
    """
    if not (_loguru_autoinit() and sys.stderr):
        # Else loguru adds no default handler (handler id 0 is the first handler
        # added by the user, if any), and there may be no stderr to log to
        return
    try:
        logger.remove(LOGURU_DEFAULT_HANDLER_ID)
    except ValueError:
        return
    logger.level("INFO", color="")  # do not highlight (bold) INFO messages
    logger.add(
        sys.stderr,
        level=os.environ.get("LOGURU_LEVEL", "INFO"),
        format=os.environ.get("LOGURU_FORMAT", LOG_FORMAT),
    )


def _loguru_autoinit() -> bool:
    """Whether loguru added its default handler on import (LOGURU_AUTOINIT)."""
    # The value loguru read on import (private, hence with a fallback)
    return getattr(getattr(loguru, "_defaults", None), "LOGURU_AUTOINIT", True)

"""Tests for the set-up done on `import feedback_forensics` (secrets and logging).

The set-up runs on import, so each test imports FF in a new Python process.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

import feedback_forensics

SRC_DIR = pathlib.Path(feedback_forensics.__file__).parents[1]
LOG_CODE = """
import feedback_forensics
from loguru import logger
logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
"""
# Environment variables with which loguru colours log lines by default (as on
# GitHub Actions)
CI_ENV_VARS = {"CI": "true", "GITHUB_ACTIONS": "true"}


def run_python(code: str, cwd: pathlib.Path, **env_vars) -> subprocess.CompletedProcess:
    """Run Python code (importing this FF source tree) in a new process."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("FF_", "LOGURU_")) and key != "HF_TOKEN"
    }
    # Plain log lines, also where loguru colours them by default (e.g. on CI)
    env["LOGURU_COLORIZE"] = "False"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    env.update(env_vars)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_secrets_toml_loaded_on_import(tmp_path):
    """E.g. the app login (FF_APP_USER/FF_APP_PW) can be configured in secrets.toml."""
    (tmp_path / "secrets.toml").write_text(
        'FF_APP_USER="alice"\nFF_APP_PW="s3cret"\nHF_TOKEN="hf_from_file"\n'
    )
    result = run_python(
        "import json\n"
        "from feedback_forensics.app import constants\n"
        "print(json.dumps([constants.USERNAME, constants.PASSWORD, constants.HF_TOKEN]))",
        cwd=tmp_path,
        HF_TOKEN="hf_from_env",
    )
    # Variables set in the environment take precedence over secrets.toml
    assert json.loads(result.stdout.splitlines()[-1]) == [
        "alice",
        "s3cret",
        "hf_from_env",
    ]


@pytest.mark.parametrize("env_vars", [{}, CI_ENV_VARS], ids=["default", "ci"])
def test_logging_set_up_on_import(tmp_path, env_vars):
    result = run_python(LOG_CODE, cwd=tmp_path, **env_vars)
    log_lines = result.stderr.splitlines()
    assert "📜  | INFO | info message" in log_lines
    assert "📜  | WARNING | warning message" in log_lines
    assert "debug message" not in result.stderr
    assert "message" not in result.stdout  # e.g. `ff-data ... -` writes data to stdout


def test_logging_respects_loguru_level(tmp_path):
    result = run_python(LOG_CODE, cwd=tmp_path, LOGURU_LEVEL="DEBUG")
    assert "📜  | DEBUG | debug message" in result.stderr.splitlines()


def test_logging_settings_in_secrets_toml(tmp_path):
    """LOGURU_LEVEL and LOGURU_FORMAT also apply if set in secrets.toml."""
    (tmp_path / "secrets.toml").write_text(
        'LOGURU_LEVEL="DEBUG"\nLOGURU_FORMAT="from secrets: {message}"\n'
    )
    result = run_python(LOG_CODE, cwd=tmp_path)
    assert result.stderr.splitlines()[-3:] == [
        "from secrets: debug message",
        "from secrets: info message",
        "from secrets: warning message",
    ]


@pytest.mark.parametrize(
    "setup_code, env_vars",
    [
        ("from loguru import logger\nlogger.remove()\n", {}),
        # loguru adds no default handler: the first handler added gets its id (0)
        ("from loguru import logger\n", {"LOGURU_AUTOINIT": "False"}),
        # Neither without stderr (e.g. pythonw). Regression: FF removed the user's
        # handler instead, and failed to add one logging to stderr (None)
        ("sys.stderr = None\nfrom loguru import logger\n", {}),
    ],
    ids=["default", "no_autoinit", "no_stderr"],
)
def test_logging_configured_before_import_kept(tmp_path, setup_code, env_vars):
    result = run_python(
        "import sys\n"
        + setup_code
        + "logger.add(sys.stdout, format='custom: {message}')\n"
        + LOG_CODE,
        cwd=tmp_path,
        **env_vars,
    )
    assert "info message" not in result.stderr
    assert result.stdout.splitlines()[-3:] == [
        "custom: debug message",
        "custom: info message",
        "custom: warning message",
    ]


@pytest.mark.parametrize(
    "problem", ["not_utf8", "utf16_without_bom", "null_character", "unreadable"]
)
def test_secrets_toml_that_cannot_be_read_is_skipped(tmp_path, problem):
    """Regression: such a secrets.toml made `import feedback_forensics` fail (and with
    it the app and every CLI). As ICAI did, it is skipped (with a warning)."""
    secrets_path = tmp_path / "secrets.toml"
    if problem == "not_utf8":
        # UTF-16, as written by Windows PowerShell 5.1 (`echo ... > secrets.toml`)
        secrets_path.write_text('FF_APP_USER="alice"\n', encoding="utf-16")
    elif problem == "utf16_without_bom":
        # Valid UTF-8 (null characters), but not as environment variable names
        secrets_path.write_text('FF_APP_USER="alice"\n', encoding="utf-16-le")
    elif problem == "null_character":
        secrets_path.write_text('FF_APP_USER="al\x00ice"\n')
    else:
        secrets_path.write_text('FF_APP_USER="alice"\n')
        secrets_path.chmod(0)
        if os.access(secrets_path, os.R_OK):
            pytest.skip("Files without read permission are readable (e.g. as root)")

    result = run_python("import feedback_forensics\nprint('imported')", cwd=tmp_path)

    assert result.stdout.splitlines() == ["imported"]
    assert "Could not load environment variables from ./secrets.toml" in result.stderr

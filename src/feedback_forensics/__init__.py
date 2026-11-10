from feedback_forensics.environment import load_secrets, setup_logging

# Before importing anything else: settings are read from env vars on import
load_secrets()
setup_logging()

from feedback_forensics.data.handler import DatasetHandler  # noqa: E402

__all__ = ["DatasetHandler"]

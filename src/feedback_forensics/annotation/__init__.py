"""Native principle-following annotation (formerly run via the ICAI package)."""

from feedback_forensics.annotation.pipeline import annotate_dataset
from feedback_forensics.annotation.principles import get_default_principles

__all__ = ["annotate_dataset", "get_default_principles"]

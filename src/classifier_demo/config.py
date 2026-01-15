"""Configuration constants and types."""

from typing import Literal

DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"

Sentiment = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]

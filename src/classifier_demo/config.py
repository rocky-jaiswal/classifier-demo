"""Configuration constants and types."""

from typing import Literal, get_args

DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"

Sentiment = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
VALID_SENTIMENTS = list(get_args(Sentiment))

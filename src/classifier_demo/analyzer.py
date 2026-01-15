from pathlib import Path
from typing import TypedDict, Any
import dspy
from guardrails import Guard
from guardrails.validators import (
    Validator,
    ValidationResult,
    PassResult,
    FailResult,
    register_validator,
)

from .config import DEFAULT_MODEL, Sentiment, VALID_SENTIMENTS


@register_validator(name="valid_choices", data_type="string")
class ValidChoices(Validator):
    """Validates that a value is one of the allowed choices."""

    def __init__(self, choices: list[str], on_fail: str = "exception"):
        super().__init__(on_fail=on_fail)
        self.choices = choices

    def validate(self, value: Any, metadata: dict = {}) -> ValidationResult:
        if value in self.choices:
            return PassResult()
        return FailResult(
            error_message=f"Value '{value}' is not in allowed choices: {self.choices}"
        )


@register_validator(name="no_profanity", data_type="string")
class NoProfanity(Validator):
    """Validates that input text does not contain profane/toxic words."""

    # Basic blocklist - in production, use a proper library like `profanity-check` or `better-profanity`
    DEFAULT_BLOCKLIST = {"shit", "damn", "crap"}

    def __init__(self, blocklist: set[str] | None = None, on_fail: str = "exception"):
        super().__init__(on_fail=on_fail)
        self.blocklist = blocklist or self.DEFAULT_BLOCKLIST

    def validate(self, value: Any, metadata: dict = {}) -> ValidationResult:
        text_lower = value.lower()
        found = [word for word in self.blocklist if word in text_lower]
        if not found:
            return PassResult()
        return FailResult(
            error_message="Input contains prohibited content. Please rephrase."
        )


class SentimentResult(TypedDict):
    """Structured result from sentiment analysis."""

    sentiment: Sentiment
    explanation: str


class SentimentSignature(dspy.Signature):
    """Analyze the sentiment of text."""

    text: str = dspy.InputField(desc="The text to analyze")
    sentiment: Sentiment = dspy.OutputField(desc="The sentiment classification")
    explanation: str = dspy.OutputField(desc="Brief explanation for the classification")


class SentimentAnalyzer(dspy.Module):
    """Analyze text sentiment using Claude via DSPy."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        optimized_path: Path | str | None = None,
    ):
        super().__init__()
        dspy.configure(lm=dspy.LM(model))
        self.predict = dspy.Predict(SentimentSignature)

        # Guardrail: Validate input doesn't contain profanity
        self.input_guard = Guard().use(NoProfanity(on_fail="exception"))

        # Guardrail: Validate sentiment output is one of the allowed values
        self.output_guard = Guard().use(
            ValidChoices(choices=VALID_SENTIMENTS, on_fail="exception")
        )

        if optimized_path is not None:
            self.predict.load(Path(optimized_path))

    def forward(self, text: str) -> dspy.Prediction:
        return self.predict(text=text)

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment and return structured result."""
        # Validate input before sending to LLM
        self.input_guard.validate(text)

        result = self(text=text)

        # Validate output with guardrail
        self.output_guard.validate(result.sentiment)

        return SentimentResult(
            sentiment=result.sentiment,
            explanation=result.explanation,
        )

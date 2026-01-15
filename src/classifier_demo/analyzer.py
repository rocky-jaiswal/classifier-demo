"""Sentiment analysis using Claude via DSPy with Guardrails validation."""

from pathlib import Path
from typing import TypedDict

import dspy
from guardrails import Guard

from .config import COMPETITORS, DEFAULT_MODEL, Sentiment, VALID_SENTIMENTS
from .validators import NoCompetitors, NoPII, NoProfanity, ValidChoices


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

        # Guardrail: Validate input (chained validators)
        self.input_guard = (
            Guard()
            .use(NoProfanity(on_fail="exception"))
            .use(NoPII(on_fail="exception"))
            .use(NoCompetitors(competitors=COMPETITORS, on_fail="exception"))
        )

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

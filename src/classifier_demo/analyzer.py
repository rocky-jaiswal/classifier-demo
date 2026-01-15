from pathlib import Path
from typing import TypedDict
import dspy

from .config import DEFAULT_MODEL, Sentiment


class SentimentResult(TypedDict):
    """Structured result from sentiment analysis."""

    sentiment: Sentiment
    explanation: str


class SentimentSignature(dspy.Signature):
    """Analyze the sentiment of text."""

    text: str = dspy.InputField(desc="The text to analyze")
    sentiment: Sentiment = dspy.OutputField(desc="The sentiment classification")
    explanation: str = dspy.OutputField(
        desc="Brief explanation for the classification"
    )


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

        if optimized_path is not None:
            self.predict.load(Path(optimized_path))

    def forward(self, text: str) -> dspy.Prediction:
        return self.predict(text=text)

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment and return structured result."""
        result = self(text=text)
        return SentimentResult(
            sentiment=result.sentiment,
            explanation=result.explanation,
        )

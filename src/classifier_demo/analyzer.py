from typing import Literal, TypedDict
import dspy


class SentimentResult(TypedDict):
    """Structured result from sentiment analysis."""

    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    explanation: str


class SentimentSignature(dspy.Signature):
    """Analyze the sentiment of text."""

    text: str = dspy.InputField(desc="The text to analyze")
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = dspy.OutputField(
        desc="The sentiment classification"
    )
    explanation: str = dspy.OutputField(
        desc="Brief explanation for the classification"
    )


class SentimentAnalyzer(dspy.Module):
    """Analyze text sentiment using Claude via DSPy."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4-20250514"):
        super().__init__()
        dspy.configure(lm=dspy.LM(model))
        self.predict = dspy.Predict(SentimentSignature)

    def forward(self, text: str) -> dspy.Prediction:
        return self.predict(text=text)

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment and return structured result."""
        result = self(text=text)
        return SentimentResult(
            sentiment=result.sentiment,
            explanation=result.explanation,
        )

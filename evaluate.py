"""Simple evaluation for the Sentiment Analyzer.

Evaluates:
1. Classification accuracy (does sentiment match expected?)
2. Explanation quality (quick LLM check - is explanation grounded in text?)
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
import dspy

from classifier_demo import SentimentAnalyzer
from classifier_demo.optimization import load_dataset

warnings.filterwarnings("ignore")

OPTIMIZED_MODEL_PATH = Path(__file__).parent / "optimized_sentiment.json"


class ExplanationCheckSignature(dspy.Signature):
    """Check if an explanation is grounded in the original text."""

    original_text: str = dspy.InputField(desc="The original text being analyzed")
    explanation: str = dspy.InputField(desc="The explanation given for the sentiment")
    is_grounded: bool = dspy.OutputField(
        desc="True if explanation only references content from original text, False if it invents things"
    )


def main():
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found")
        return

    # Load our sentiment analyzer first (configures DSPy with Sonnet)
    optimized_path = OPTIMIZED_MODEL_PATH if OPTIMIZED_MODEL_PATH.exists() else None
    analyzer = SentimentAnalyzer(optimized_path=optimized_path)

    # Create explanation checker with Haiku for fast grounding checks
    haiku_lm = dspy.LM("anthropic/claude-haiku-4-5-20251001")
    explanation_checker = dspy.Predict(ExplanationCheckSignature)
    explanation_checker.lm = haiku_lm  # Use Haiku instead of default

    # Load test data
    data_path = Path(__file__).parent / "data" / "sentiment_dataset.json"
    data = load_dataset(data_path)

    sample_size = 10
    sample_data = data[:sample_size]

    print("Sentiment Analyzer Evaluation")
    print("-" * 50)
    print(f"Evaluating {sample_size} samples...")
    if optimized_path:
        print("(Using optimized model)")
    print()

    correct_sentiment = 0
    grounded_explanations = 0

    for i, item in enumerate(sample_data):
        result = analyzer.analyze(item["text"])

        # Check sentiment accuracy
        sentiment_correct = result["sentiment"] == item["sentiment"]
        if sentiment_correct:
            correct_sentiment += 1

        # Check if explanation is grounded (using Haiku - fast)
        check = explanation_checker(
            original_text=item["text"],
            explanation=result["explanation"],
        )
        if check.is_grounded:
            grounded_explanations += 1

        # Print progress
        status = "✓" if sentiment_correct else "✗"
        grounded = "grounded" if check.is_grounded else "NOT grounded"
        print(f"  [{i + 1}/{sample_size}] {status} {result['sentiment']} ({grounded})")

    # Results
    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(
        f"Sentiment Accuracy:    {correct_sentiment}/{sample_size} ({correct_sentiment / sample_size:.0%})"
    )
    print(
        f"Grounded Explanations: {grounded_explanations}/{sample_size} ({grounded_explanations / sample_size:.0%})"
    )


if __name__ == "__main__":
    main()

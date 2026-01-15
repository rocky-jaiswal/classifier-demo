import json
from pathlib import Path
from typing import TypedDict

import dspy

from .analyzer import SentimentSignature
from .config import DEFAULT_MODEL, Sentiment


class Example(TypedDict):
    text: str
    sentiment: Sentiment


def load_dataset(path: Path) -> list[Example]:
    """Load sentiment dataset from JSON file."""
    with open(path) as f:
        return json.load(f)


def sentiment_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> bool:
    """Metric: does predicted sentiment match expected sentiment?"""
    return prediction.sentiment == example.sentiment


def split_dataset(
    data: list[Example], train_ratio: float = 0.7
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Split dataset into train and test sets as DSPy Examples."""
    examples = [
        dspy.Example(text=item["text"], sentiment=item["sentiment"]).with_inputs("text")
        for item in data
    ]
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]


def optimize(
    trainset: list[dspy.Example],
    model: str = DEFAULT_MODEL,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
) -> dspy.Module:
    """Optimize the SentimentAnalyzer using BootstrapFewShot."""
    dspy.configure(lm=dspy.LM(model))

    # Create a simple predictor for optimization
    predictor = dspy.Predict(SentimentSignature)

    # Use BootstrapFewShot optimizer
    optimizer = dspy.BootstrapFewShot(
        metric=sentiment_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )

    optimized = optimizer.compile(predictor, trainset=trainset)
    return optimized


def evaluate(module: dspy.Module, testset: list[dspy.Example]) -> float:
    """Evaluate module accuracy on test set."""
    correct = 0
    for example in testset:
        prediction = module(text=example.text)
        if prediction.sentiment == example.sentiment:
            correct += 1
    return correct / len(testset)

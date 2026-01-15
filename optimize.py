"""Script to run DSPy optimization on the sentiment analyzer."""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

import dspy

from classifier_demo.optimization import (
    evaluate,
    load_dataset,
    optimize,
    split_dataset,
)
from classifier_demo.analyzer import SentimentSignature
from classifier_demo.config import DEFAULT_MODEL

# Suppress DSPy/LiteLLM serialization warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def main():
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment.")
        return

    # Load and split dataset
    data_path = Path(__file__).parent / "data" / "sentiment_dataset.json"
    data = load_dataset(data_path)
    trainset, testset = split_dataset(data, train_ratio=0.7)

    print(f"Dataset: {len(data)} examples")
    print(f"Train: {len(trainset)}, Test: {len(testset)}")
    print("-" * 50)

    # Configure DSPy
    dspy.configure(lm=dspy.LM(DEFAULT_MODEL))

    # Baseline evaluation (unoptimized)
    print("\n1. Evaluating baseline (unoptimized)...")
    baseline = dspy.Predict(SentimentSignature)
    baseline_acc = evaluate(baseline, testset)
    print(f"   Baseline accuracy: {baseline_acc:.1%}")

    # Optimize
    print("\n2. Running BootstrapFewShot optimization...")
    print("   (This will make several LLM calls to find good few-shot examples)")
    optimized = optimize(trainset)

    # Evaluate optimized
    print("\n3. Evaluating optimized model...")
    optimized_acc = evaluate(optimized, testset)
    print(f"   Optimized accuracy: {optimized_acc:.1%}")

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Baseline accuracy:  {baseline_acc:.1%}")
    print(f"Optimized accuracy: {optimized_acc:.1%}")
    print(f"Improvement:        {(optimized_acc - baseline_acc):+.1%}")

    # Save optimized module
    output_path = Path(__file__).parent / "optimized_sentiment.json"
    optimized.save(output_path)
    print(f"\nOptimized model saved to: {output_path}")


if __name__ == "__main__":
    main()

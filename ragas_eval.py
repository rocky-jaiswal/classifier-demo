"""RAGAs evaluation for the Sentiment Analyzer.

Uses RAGAs to evaluate if our sentiment explanations are:
- Faithful: Is the explanation grounded in the original text?
- Not hallucinating: Does it only reference things actually in the input?

This catches cases where the model invents reasons not present in the text.
"""

import os
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic
from datasets import Dataset

from classifier_demo import SentimentAnalyzer
from classifier_demo.optimization import load_dataset

warnings.filterwarnings("ignore")

OPTIMIZED_MODEL_PATH = Path(__file__).parent / "optimized_sentiment.json"


def main():
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found")
        return

    # Load our sentiment analyzer
    optimized_path = OPTIMIZED_MODEL_PATH if OPTIMIZED_MODEL_PATH.exists() else None
    analyzer = SentimentAnalyzer(optimized_path=optimized_path)

    # Load test data
    data_path = Path(__file__).parent / "data" / "sentiment_dataset.json"
    data = load_dataset(data_path)

    # Take a sample for evaluation (RAGAs is slow, uses LLM as judge)
    # Faithfulness makes multiple LLM calls per sample, so keep this small
    sample_size = 2
    sample_data = data[:sample_size]

    print("Sentiment Analyzer - RAGAs Evaluation")
    print("-" * 50)

    print(f"Evaluating {sample_size} samples...")
    if optimized_path:
        print("(Using optimized model)")
    print()

    # Run sentiment analysis on each sample
    print("Running sentiment analysis...")
    t0 = time.time()

    results_data = {
        "user_input": [],  # "What is the sentiment of: <text>?"
        "response": [],  # The explanation from our analyzer
        "retrieved_contexts": [],  # The original text (context for the explanation)
    }

    for item in sample_data:
        result = analyzer.analyze(item["text"])

        results_data["user_input"].append(
            f"What is the sentiment of this text and why? Text: {item['text']}"
        )
        results_data["response"].append(
            f"Sentiment: {result['sentiment']}. {result['explanation']}"
        )
        results_data["retrieved_contexts"].append([item["text"]])

    print(f"  Sentiment analysis took {time.time() - t0:.1f}s")

    dataset = Dataset.from_dict(results_data)

    # RAGAs evaluation using Claude as judge
    print("Running RAGAs evaluation (using Haiku as judge)...")
    t0 = time.time()
    print("-" * 50)

    # Use Haiku for faster/cheaper evaluation judging
    evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001"))

    eval_results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness()],
        llm=evaluator_llm,
    )

    print(f"  RAGAs evaluation took {time.time() - t0:.1f}s")

    # Convert to pandas for easier access
    df = eval_results.to_pandas()

    # Calculate average faithfulness score
    avg_faithfulness = df["faithfulness"].mean()

    # Results
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Faithfulness Score: {avg_faithfulness:.1%}")
    print()
    print("What this measures:")
    print("  - Are the explanations grounded in the input text?")
    print("  - 100% = explanations only reference what's in the text")
    print("  - Lower = model is hallucinating/inventing reasons")
    print()

    # Per-sample breakdown
    print("Per-sample breakdown:")
    print("-" * 50)
    for i, row in df.iterrows():
        original_text = (
            sample_data[i]["text"][:50] + "..."
            if len(sample_data[i]["text"]) > 50
            else sample_data[i]["text"]
        )
        print(f'\n  Text: "{original_text}"')
        print(f"  Response: {row['response'][:80]}...")
        print(f"  Faithfulness: {row['faithfulness']:.0%}")


if __name__ == "__main__":
    main()

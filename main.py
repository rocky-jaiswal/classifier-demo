import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from classifier_demo import SentimentAnalyzer

OPTIMIZED_MODEL_PATH = Path(__file__).parent / "optimized_sentiment.json"

# Suppress DSPy/LiteLLM serialization warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def main():
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment.")
        print("Copy .env.example to .env and add your API key.")
        return

    print("Sentiment Analyzer")
    print("-" * 40)

    text = input("Enter text to analyze: ")

    if not text.strip():
        print("No text provided.")
        return

    print("\nAnalyzing...")

    # Use optimized model if available
    optimized_path = OPTIMIZED_MODEL_PATH if OPTIMIZED_MODEL_PATH.exists() else None
    if optimized_path:
        print("(Using optimized model)")

    analyzer = SentimentAnalyzer(optimized_path=optimized_path)
    result = analyzer.analyze(text)

    print(f"\nSentiment: {result['sentiment']}")
    print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()

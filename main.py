import os
import warnings
from dotenv import load_dotenv
from classifier_demo import SentimentAnalyzer

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

    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(text)

    print(f"\nSentiment: {result['sentiment']}")
    print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()

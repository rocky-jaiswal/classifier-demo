# Classifier Demo

A sentiment analysis CLI using Claude and DSPy for prompt management and optimization.

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure your API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```

## Usage

### Basic Sentiment Analysis

```bash
uv run python main.py
```

Enter text when prompted, and the analyzer will classify it as POSITIVE, NEGATIVE, or NEUTRAL with an explanation.

### DSPy Optimization

The project uses DSPy's `BootstrapFewShot` optimizer to automatically find effective few-shot examples that improve classification accuracy.

**Run optimization:**
```bash
uv run python optimize.py
```

This will:
1. Load the dataset from `data/sentiment_dataset.json` (102 examples)
2. Split into 70% train / 30% test
3. Evaluate baseline accuracy (no few-shot examples)
4. Run `BootstrapFewShot` to find optimal few-shot demonstrations
5. Evaluate optimized accuracy
6. Save the optimized model to `optimized_sentiment.json`

**How it works:**
- DSPy tries different combinations of training examples as few-shot demos
- It evaluates which combinations improve the sentiment accuracy metric
- The best combination gets saved and automatically used by `main.py`

**Using the optimized model:**

Once `optimized_sentiment.json` exists, `main.py` automatically loads it. You'll see "(Using optimized model)" in the output.

You can also load it programmatically:
```python
from classifier_demo import SentimentAnalyzer

analyzer = SentimentAnalyzer(optimized_path="optimized_sentiment.json")
result = analyzer.analyze("I love this product!")
print(result["sentiment"])  # POSITIVE
```

## Project Structure

```
classifier-demo/
├── main.py                 # CLI interface
├── optimize.py             # DSPy optimization script
├── data/
│   └── sentiment_dataset.json
├── src/classifier_demo/
│   ├── __init__.py
│   ├── analyzer.py         # SentimentAnalyzer class
│   ├── config.py           # Shared constants and types
│   └── optimization.py     # Optimization utilities
└── optimized_sentiment.json  # Generated after optimization
```

## Configuration

Edit `src/classifier_demo/config.py` to change:
- `DEFAULT_MODEL` - The Claude model to use
- `Sentiment` - The sentiment categories (POSITIVE, NEGATIVE, NEUTRAL)

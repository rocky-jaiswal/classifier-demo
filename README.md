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

## Evaluation

Two evaluation approaches are available to check if sentiment explanations are grounded (not hallucinated):

### Simple Evaluation (Fast)

```bash
uv run python evaluate.py
```

Uses DSPy with Claude Haiku as a grounding checker. Evaluates 10 samples quickly and reports:
- **Sentiment Accuracy** - Does the predicted sentiment match the expected?
- **Grounded Explanations** - Is the explanation based only on the input text?

### RAGAs Evaluation (Thorough)

```bash
uv run python ragas_eval.py
```

Uses the RAGAs framework with the **Faithfulness** metric. This is the industry-standard approach for evaluating if LLM outputs are grounded in context.

**Note:** RAGAs is slower because the Faithfulness metric makes multiple LLM calls per sample:
1. Extracts atomic claims from the response
2. Verifies each claim against the context
3. Calculates `supported_claims / total_claims`

## Project Structure

```
classifier-demo/
├── main.py                 # CLI interface
├── optimize.py             # DSPy optimization script
├── evaluate.py             # Fast DSPy-based evaluation
├── ragas_eval.py           # RAGAs evaluation (Faithfulness)
├── data/
│   └── sentiment_dataset.json
├── src/classifier_demo/
│   ├── __init__.py
│   ├── analyzer.py         # SentimentAnalyzer class
│   ├── config.py           # Shared constants and types
│   └── optimization.py     # Optimization utilities
└── optimized_sentiment.json  # Generated after optimization
```

## MLflow Tracing

The project uses MLflow to capture and inspect prompts sent to Anthropic.

**Run the analyzer** (traces are captured automatically):
```bash
uv run python main.py
```

**View traces in MLflow UI:**
```bash
uv run mlflow ui
```

Open http://localhost:5000 to see:
- Full prompts sent to the LLM (including few-shot examples)
- Model responses
- Latency and token usage
- Experiment history

## Configuration

Edit `src/classifier_demo/config.py` to change:
- `DEFAULT_MODEL` - The Claude model to use
- `Sentiment` - The sentiment categories (POSITIVE, NEGATIVE, NEUTRAL)

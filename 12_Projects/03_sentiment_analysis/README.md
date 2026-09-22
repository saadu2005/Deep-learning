# Project 3 · Sentiment analysis

A trainable binary sentiment classifier on the official Keras **IMDB** integer-tokenized review dataset. The model uses token embeddings, global average pooling and dropout. Train/validation are drawn from the training partition; the official test partition stays untouched until evaluation.

```bash
python 12_Projects/03_sentiment_analysis/main.py --epochs 2 --limit 2000
```

Drop `--limit` for full training. The example predicts on a pre-tokenized IMDB test sample, not arbitrary raw text: deploying arbitrary text requires reproducing the exact IMDB vocabulary and encoding rules. The first run downloads the dataset. Models are written to the ignored `artifacts/` folder.

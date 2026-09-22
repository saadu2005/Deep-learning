# Project 2 · Fashion image classifier

A complete ten-class convolutional classifier using **Fashion-MNIST** (clothing images). No custom folder or Kaggle token is required. The built-in dataset downloads once; a held-out test split is only used for evaluation. Includes softmax prediction, model saving and a JSON class mapping.

```bash
python 12_Projects/02_image_classifier/main.py --epochs 2 --limit 5000
```

Remove `--limit` for full training. Model and class mapping are created in `artifacts/`, which is excluded from Git. Images must be 28×28 grayscale, float32, scaled to [0,1] for inference; this is **not** a general-purpose arbitrary image classifier.

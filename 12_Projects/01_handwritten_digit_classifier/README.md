# Project 1 · Handwritten digit classifier

A TensorFlow CNN trained on the bundled MNIST train split and evaluated once on the independent test split. Downloads MNIST on first run.

```bash
python 12_Projects/01_handwritten_digit_classifier/main.py --epochs 2 --limit 5000 --output artifacts/mnist.keras
```

Remove `--limit` for full training. `--output` writes a trained model locally; it is ignored by Git. Accuracy varies by environment, number of epochs and data size. Training and test data must not overlap.

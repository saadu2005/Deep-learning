# Learning guide

## How to use the lessons

Run the numbered lessons from the repo root so shared helper imports resolve: `python 05_ANN/01_simple_ann.py`. Conceptual lessons use only Python and NumPy. Framework lessons require their respective dependencies. Four end-to-end projects follow the 11 learning chapters.

## Essential rules

- Classification: split by target first, then fit preprocessing on training samples only; validation/test inputs may only be transformed with those learned parameters.
- Time series: choose splits in chronological order and construct each supervised input solely from observations before its target. Fit scaler on training history only.
- Never call training accuracy test accuracy. Evaluate a held-out split after model selection; for realistic projects, keep a separate validation split for tuning.
- Save preprocessing and class labels alongside learned model parameters when deploying.
- Set seeds for reproducibility, but small floating-point differences across devices and software versions can remain.
- Pretrained MobileNetV2 lessons download ImageNet weights at first run; these network-requiring examples are not part of offline CI.

## Troubleshooting

If `ModuleNotFoundError: tensorflow` occurs, install `requirements-tensorflow.txt`; for `torch`, install `requirements-pytorch.txt`. For Apple Silicon / CUDA / other acceleration, use the framework's official platform instructions rather than assuming this CPU-oriented setup covers every device. Network access is required on the first execution of dataset/weight download examples. Run from the repo root. On memory-limited devices, pass a smaller `--limit` to the first three projects and fewer `--epochs` to all projects.

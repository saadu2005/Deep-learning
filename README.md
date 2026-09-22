# Deep Learning · From First Neuron to Real Projects

A hands-on, beginner-to-intermediate learning repository by **Saad Ahmed**. Short, independently runnable Python lessons on neural networks, TensorFlow, PyTorch and modern architectures, followed by four end-to-end projects.

[![Fast quality checks](https://github.com/saadu2005/Deep-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/saadu2005/Deep-learning/actions/workflows/ci.yml)

## Learning path

| Chapter | Topic |
| --- | --- |
| `01_Introduction` | AI, ML and deep learning concepts |
| `02_Neural_Network_Basics` | Neurons, activations and backpropagation |
| `03_TensorFlow` | Tensors and neural network models |
| `04_PyTorch` | Tensors, autograd, training loops |
| `05_ANN` | Regression and classification |
| `06_CNN` | Convolutional networks and MNIST |
| `07_RNN` | Recurrence and sequences |
| `08_LSTM` | LSTM and text classification |
| `09_Regularization` | Dropout, batch normalization, early stopping |
| `10_Transfer_Learning` | Feature extraction and fine-tuning setup |
| `11_Transformers` | Attention and encoder classification |
| `12_Projects` | Four end-to-end projects |

## Setup

Use Python 3.11 in a fresh environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python 01_Introduction/01_what_is_deep_learning.py
```

For lighter installs, choose `requirements-tensorflow.txt` or `requirements-pytorch.txt`; the core math examples and tests use `requirements-dev.txt`. Framework dependencies are large, and some pretrained weights or datasets download on first run.

## Projects

| Project | Dataset | What it does |
| --- | --- | --- |
| [Digit classifier](12_Projects/01_handwritten_digit_classifier/) | MNIST | CNN train/evaluate/save |
| [Image classifier](12_Projects/02_image_classifier/) | Fashion-MNIST | CNN with label mapping and example predictions |
| [Sentiment analysis](12_Projects/03_sentiment_analysis/) | IMDB | Embedding classifier evaluated on held-out test data |
| [Time-series forecast](12_Projects/04_time_series_prediction/) | Synthetic signal | LSTM with chronological splits and baseline |

Run from the repository root:

```bash
python 12_Projects/01_handwritten_digit_classifier/main.py --epochs 2 --limit 5000
python 12_Projects/02_image_classifier/main.py --epochs 2 --limit 5000
python 12_Projects/03_sentiment_analysis/main.py --epochs 2 --limit 2000
python 12_Projects/04_time_series_prediction/main.py --epochs 10
```

For full training, omit `--limit`. Metrics vary by configuration. Synthetic forecasting results are not real-world forecasts.

## Quality and reproducibility

- Train-only scaling in tabular classification; chronological splits and train-only scaling in time-series examples.
- Held-out test sets in full projects; toy text lessons report training accuracy only.
- Reproducible seeds for full projects. Downloaded data and trained model files are excluded from Git.
- CI covers syntax and lightweight tests; full TensorFlow/PyTorch integration training is not yet validated in CI.
- Transfer-learning chapters demonstrate freeze/unfreeze configuration, not full fine-tuning.

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q .
python -m pytest -q
```

See [learning guide](docs/LEARNING_GUIDE.md), [contribution guidelines](CONTRIBUTING.md), and [MIT license](LICENSE).

**Maintainer:** Saad Ahmed · Python / AI / Machine Learning.

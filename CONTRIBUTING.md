# Contributing

Thank you for helping improve these learning examples. Keep each lesson runnable independently and explain the data source, preprocessing, training, evaluation, and limitations. Do not commit datasets, private credentials, personal information, or trained weights.

## Local checks

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q .
python -m pytest -q
```

For changes to TensorFlow/PyTorch lessons, install the relevant optional requirements and run the changed scripts before claiming runtime compatibility. Keep the test split independent of scaler fitting, tuning and training. Prefer small, deterministic smoke tests for CI and avoid downloads in unit tests. Open an issue or pull request describing your changes.

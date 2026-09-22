# Project 4 · Time-series prediction

A reproducible LSTM forecasting example using a **synthetic** trend and periodic signals (not a real stock-market model). Targets are chronologically split 60% / 20% / 20%. The standardization scaler is fit only on the training observations, and validation/test windows use past context without using future values. The final evaluation reports MAE in the original units and compares with a last-value baseline.

```bash
python 12_Projects/04_time_series_prediction/main.py --epochs 10 --lookback 24
```

The model and training-only scaler metadata are written to ignored `artifacts/`. A low error on synthetic data does not establish performance on financial or real-world series. Train on task-specific data and compare with meaningful baselines before deployment.

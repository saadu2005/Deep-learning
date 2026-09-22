"""Small, testable data utilities shared by the learning examples.

All preprocessing parameters are learned from training data only.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def classification_split(X, y, *, test_size=0.2, random_state=42):
    """Stratified split, then fit StandardScaler ONLY on the training features."""
    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(X), np.asarray(y), test_size=test_size,
        random_state=random_state, stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype('float32')
    X_test = scaler.transform(X_test).astype('float32')
    return X_train, X_test, y_train, y_test, scaler


def chronological_windows(values, lookback, target_start, target_end):
    """Windows for targets in [target_start, target_end), using only past observations.

    `values` must be scaled using a scaler fit on *training observations only*.
    Validation/test windows may use observations immediately preceding their target.
    """
    data = np.asarray(values, dtype='float32').reshape(-1)
    if not 1 <= lookback <= target_start < target_end <= len(data):
        raise ValueError('Require 1 <= lookback <= target_start < target_end <= len(values)')
    X = np.stack([data[i-lookback:i] for i in range(target_start,target_end)])
    y = data[target_start:target_end]
    return X[..., None], y

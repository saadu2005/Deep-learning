"""Fast tests: no large model downloads and no neural-network framework needed."""
import numpy as np
import pytest
from dl_utils import classification_split, chronological_windows


def test_classification_scaler_fitted_to_train_only():
    X = np.arange(60, dtype='float32').reshape(20, 3)
    y = np.array([0, 1] * 10)
    x_train, x_test, y_train, y_test, scaler = classification_split(X, y)
    assert x_train.shape == (16, 3)
    assert x_test.shape == (4, 3)
    assert len(y_train) == 16 and len(y_test) == 4
    np.testing.assert_allclose(x_train.mean(axis=0), 0, atol=1e-6)
    # Reverse the scaling to validate the statistics on *training* observations.
    np.testing.assert_allclose(scaler.mean_, scaler.inverse_transform(x_train).mean(axis=0))
    # Compare held-out values after a proper transform.
    np.testing.assert_allclose(scaler.transform(scaler.inverse_transform(x_test)), x_test, atol=1e-6)
    assert not np.allclose(scaler.mean_, X.mean(axis=0))


def test_windows_are_chronological_and_do_not_include_target():
    series = np.arange(10, dtype='float32')
    x, y = chronological_windows(series, lookback=3, target_start=6, target_end=9)
    assert x.shape == (3, 3, 1)
    np.testing.assert_array_equal(x[0, :, 0], [3, 4, 5])
    np.testing.assert_array_equal(y, [6, 7, 8])
    assert all(row[-1, 0] < target for row, target in zip(x, y))


@pytest.mark.parametrize('args', [(np.arange(4), 0, 1, 3),
                                  (np.arange(4), 2, 1, 3),
                                  (np.arange(4), 2, 2, 5),
                                  (np.arange(4), 2, 3, 3)])
def test_windows_reject_invalid_bounds(args):
    with pytest.raises(ValueError):
        chronological_windows(*args)

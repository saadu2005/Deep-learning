"""Time-series LSTM: chronological holdouts, leakage-free scaling, baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense
from pathlib import Path
import sys

# Allow this lesson to run directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dl_utils import chronological_windows


def run(epochs=10, output='artifacts/time_series.keras', lookback=24):
    if not 1 <= lookback < 100:
        raise ValueError('--lookback must be between 1 and 99')
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    time = np.arange(600, dtype='float32')
    values = (0.003 * time + np.sin(time / 12) +
              0.3 * np.sin(time / 3) + rng.normal(0, 0.08, len(time))).astype('float32')
    train_end, val_end, test_end = 360, 480, len(values)

    # Fit scaler only on historical train observations, NOT validation/test.
    scaler = StandardScaler().fit(values[:train_end, None])
    scaled = scaler.transform(values[:, None]).ravel()
    x_train, y_train = chronological_windows(scaled, lookback, lookback, train_end)
    x_val, y_val = chronological_windows(scaled, lookback, train_end, val_end)
    x_test, y_test = chronological_windows(scaled, lookback, val_end, test_end)

    model = Sequential([Input(shape=(lookback, 1)), LSTM(32), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                                  restore_best_weights=True)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
              batch_size=32, callbacks=[early_stop], verbose=2, shuffle=False)
    predicted = scaler.inverse_transform(model.predict(x_test, verbose=0)).ravel()
    actual = scaler.inverse_transform(y_test[:, None]).ravel()
    baseline = values[val_end - 1:test_end - 1]  # last-value forecast, no peeking.
    mae = float(np.mean(np.abs(actual - predicted)))
    baseline_mae = float(np.mean(np.abs(actual - baseline)))
    print(f'Chronological test MAE: {mae:.4f} | naive last-value MAE: {baseline_mae:.4f}')

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    target.with_suffix('.scaler.json').write_text(json.dumps({
        'mean': float(scaler.mean_[0]), 'scale': float(scaler.scale_[0]),
        'lookback': lookback, 'train_end': train_end, 'val_end': val_end,
        'data_source': 'synthetic demonstration, not investment advice',
    }, indent=2) + '\n')
    print(f'Saved model and training-only scaler metadata in: {target.parent}')
    return mae, baseline_mae


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lookback', type=int, default=24)
    parser.add_argument('--output', default='artifacts/time_series.keras')
    args = parser.parse_args()
    run(args.epochs, args.output, args.lookback)

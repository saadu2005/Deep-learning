"""IMDB sentiment classification with train/validation/test separation."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.utils import pad_sequences

VOCABULARY_SIZE = 10000
SEQUENCE_LENGTH = 200


def run(epochs=3, limit=None, output='artifacts/imdb_sentiment.keras'):
    tf.keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCABULARY_SIZE)
    if limit is not None:
        if limit < 200:
            raise ValueError('--limit must be >= 200')
        x_train, y_train = x_train[:limit], y_train[:limit]
    x_train = pad_sequences(x_train, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')
    model = Sequential([Input(shape=(SEQUENCE_LENGTH,)),
                        Embedding(VOCABULARY_SIZE, 32, mask_zero=False),
                        GlobalAveragePooling1D(), Dense(32, activation='relu'),
                        Dropout(0.3), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=128, verbose=2)
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'IMDB held-out test accuracy: {test_accuracy:.4f}')
    probability = float(model.predict(x_test[:1], verbose=0)[0, 0])
    print(f'Sample test review positive probability: {probability:.3f}; actual label: {int(y_test[0])}')
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    print(f'Saved model: {target}')
    return test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--limit', type=int, default=None, help='Optional small training subset')
    parser.add_argument('--output', default='artifacts/imdb_sentiment.keras')
    args = parser.parse_args()
    run(args.epochs, args.limit, args.output)

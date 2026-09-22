"""MNIST digit classification project: python main.py --epochs 2 --limit 5000."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


def run(epochs=3, limit=None, output=None):
    tf.keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if limit is not None:
        if limit < 100:
            raise ValueError('--limit must be >= 100')
        x_train, y_train = x_train[:limit], y_train[:limit]
    x_train = x_train.astype('float32')[..., None] / 255.0
    x_test = x_test.astype('float32')[..., None] / 255.0
    model = Sequential([Input(shape=(28, 28, 1)), Conv2D(32, 3, activation='relu'),
                        MaxPooling2D(), Flatten(), Dense(64, activation='relu'),
                        Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=128,
              validation_split=0.1, verbose=2)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'MNIST held-out test accuracy: {accuracy:.4f}')
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(target)
        print(f'Saved model: {target}')
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--limit', type=int, default=None, help='Optional training subset for a quick demo')
    parser.add_argument('--output', default=None, help='Optional .keras model path')
    args = parser.parse_args()
    run(args.epochs, args.limit, args.output)

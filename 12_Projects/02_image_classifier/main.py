"""Fashion-MNIST clothing classifier; uses built-in dataset, no user dataset needed."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def run(epochs=3, limit=None, output='artifacts/fashion_mnist.keras'):
    tf.keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    if limit is not None:
        if limit < 100:
            raise ValueError('--limit must be >= 100')
        x_train, y_train = x_train[:limit], y_train[:limit]
    x_train = x_train.astype('float32')[..., None] / 255.0
    x_test = x_test.astype('float32')[..., None] / 255.0
    model = Sequential([Input(shape=(28, 28, 1)), Conv2D(32, 3, activation='relu'),
                        MaxPooling2D(), Conv2D(64, 3, activation='relu'), MaxPooling2D(),
                        Flatten(), Dense(64, activation='relu'), Dropout(0.3),
                        Dense(len(LABELS), activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.1, epochs=epochs,
              batch_size=64, verbose=2)
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'Fashion-MNIST held-out test accuracy: {test_accuracy:.4f}')
    probabilities = model.predict(x_test[:1], verbose=0)[0]
    print(f'Sample prediction: {LABELS[int(np.argmax(probabilities))]} | true: {LABELS[int(y_test[0])]}')
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    target.with_suffix('.labels.json').write_text(json.dumps(LABELS, indent=2) + '\n')
    print(f'Saved model and labels under: {target.parent}')
    return test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--limit', type=int, default=None, help='Optional small training subset')
    parser.add_argument('--output', default='artifacts/fashion_mnist.keras')
    args = parser.parse_args()
    run(args.epochs, args.limit, args.output)

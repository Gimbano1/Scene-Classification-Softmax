from pathlib import Path

import numpy as np
from PIL import Image


def load_image_dataset(data_dir, image_size=(32, 32)):
    """Load images from class folders."""
    data_dir = Path(data_dir)
    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    X, y, paths = [], [], []

    for label, class_name in enumerate(class_names):
        for image_path in sorted((data_dir / class_name).glob("*.jpg")):
            image = Image.open(image_path).convert("RGB").resize(image_size)
            X.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
            y.append(label)
            paths.append(str(image_path))

    return np.asarray(X), np.asarray(y), class_names, paths


def train_test_split(X, y, paths, test_size=0.25, seed=42):
    """Create a simple stratified split."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        [paths[i] for i in train_idx],
        [paths[i] for i in test_idx],
    )


def one_hot(y, num_classes):
    """Encode labels as one-hot vectors."""
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded


def softmax(logits):
    """Convert logits into class probabilities."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class SoftmaxClassifier:
    """Single linear layer followed by softmax."""

    def __init__(self, input_dim, num_classes, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(input_dim, num_classes))
        self.b = np.zeros((1, num_classes))

    def predict_proba(self, X):
        logits = X @ self.W + self.b
        return softmax(logits)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X, y, epochs=250, learning_rate=0.4, l2=0.001):
        y_encoded = one_hot(y, self.b.shape[1])
        history = []

        for _ in range(epochs):
            probabilities = self.predict_proba(X)
            loss = -np.mean(np.sum(y_encoded * np.log(probabilities + 1e-12), axis=1))
            loss += 0.5 * l2 * np.sum(self.W * self.W)

            error = probabilities - y_encoded
            grad_W = (X.T @ error) / len(X) + l2 * self.W
            grad_b = np.mean(error, axis=0, keepdims=True)

            self.W -= learning_rate * grad_W
            self.b -= learning_rate * grad_b

            accuracy = np.mean(self.predict(X) == y)
            history.append({"loss": float(loss), "accuracy": float(accuracy)})

        return history


def confusion_matrix(y_true, y_pred, num_classes):
    """Build a confusion matrix without external ML libraries."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    return matrix

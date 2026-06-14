import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        k_indices = np.argsort(distances)[:self.k]

        k_labels = self.y_train[k_indices]

        # Majority vote
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
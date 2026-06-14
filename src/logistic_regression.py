import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            error = predictions - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                cost = self.compute_cost(y, predictions)
                self.cost_history.append(cost)

    def compute_cost(self, y, predictions):
        n_samples = len(y)
        epsilon = 1e-9  # avoid log(0)
        cost = -(1 / n_samples) * np.sum(
            y * np.log(predictions + epsilon) +
            (1 - y) * np.log(1 - predictions + epsilon)
        )
        return cost

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
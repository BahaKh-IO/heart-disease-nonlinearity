import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, n_iterations=1000, C=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # convert labels from {0,1} to {-1,+1}
        y_svm = np.where(y == 1, 1, -1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent loop
        for _ in range(self.n_iterations):

            # compute gradient for EACH training point
            for i in range(n_samples):

                margin_condition = y_svm[i] * (np.dot(X[i], self.weights) + self.bias)

                if margin_condition >= 1:
                    # Case 1: correctly classified, outside margin
                    # only regularization gradient (no hinge loss)
                    dw = self.weights
                    db = 0

                else:
                    # Case 2: violation (inside margin OR wrong side)
                    # regularization + hinge loss gradient
                    dw = self.weights - self.C * y_svm[i] * X[i]
                    db = -self.C * y_svm[i]
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        raw_scores = np.dot(X, self.weights) + self.bias

        # (positive score → class 1, negative → class 0)
        return np.where(raw_scores >= 0, 1, 0)
    
    
    
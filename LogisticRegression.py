import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            lin_pred = X @ self.w + self.b
            y_pred = sigmoid(lin_pred)

            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def predict(self, X):
        lin_pred = X @ self.w + self.b
        y_pred = sigmoid(lin_pred)
        class_pred = [0 if y < 0.5 else 1 for y in y_pred]

        return class_pred

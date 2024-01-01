import numpy as np

class SVM:
    def __init__(self, lr=0.01, lp=0.01, n_iters=100):
        self.lr = lr
        self.lp = lp
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.random.randn(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):

                if y_[idx] * (x_i @ self.w) + self.b >= 1:
                    self.w = self.w - self.lr * (2 * self.lp * self.w)
                else:
                    self.w = self.w - self.lr * (2 * self.lp * self.w - np.dot(x_i, y_[idx]))
                    self.b = self.b - self.lr * y_[idx]

    def predict(self, X):
        approx = X @ self.w - self.b

        return np.sign(approx)

import numpy as np

class NeuralNetwork:
    def __init__(self, lr=0.01, n_iters=100, hidden_l=30):
        self.lr = lr
        self.n_iters = n_iters
        self.hidden_l = hidden_l

    def fit(self, X, y):
        np.random.seed(42)

        n_samples, n_features = X.shape
        n_predict = len(np.unique(y))

        # initializing weights and biases
        self.w1 = np.random.randn(self.hidden_l, n_features)
        self.b1 = np.zeros((self.hidden_l, 1))
        self.w2 = np.random.randn(n_predict, self.hidden_l)
        self.b2 = np.zeros((n_predict, 1))
        
        # implementing gradient descent
        for i in range(self.n_iters):
            z1, A1, z2, A2 = self.forward_prop(X)
            dw1, db1, dw2, db2 = self.back_prop(X, y, z1, A1, z2, A2)

            # updating weights and biases
            self.w1 = self.w1 - self.lr * dw1
            self.b1 = self.b1 - self.lr * db1
            self.w2 = self.w2 - self.lr * dw2
            self.b2 = self.b2 - self.lr * db2

            # if i % 10 == 0:
            print(f'{i} {self.accuracy(self._predict(A2), y)}')

    def accuracy(self, y_pred, y):
        return np.sum(y_pred == y) / y.size

    def predict(self, X):
        return self._predict(self.forward_prop(X)[-1])

    def _predict(self, y):
        return np.argmax(y, 0)

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward_prop(self, X):
        z1 = self.w1 @ X.T + self.b1
        A1 = self.relu(z1)
        z2 = self.w2 @ A1 + self.b2
        A2 = self.softmax(z2)

        return z1, A1, z2, A2

    def one_hot(self, y):
        one_hot_y = np.zeros((y.size, len(np.unique(y))))
        one_hot_y[np.arange(y.size), y] = 1

        return one_hot_y.T

    def deriv_relu(self, z):
        return z > 0
    
    def back_prop(self, X, y, z1, A1, z2, A2):
        m = y.size
        one_hot_y = self.one_hot(y)

        dz2 = A2 - one_hot_y
        dw2 = 1 / m * dz2 @ A1.T
        db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)

        dz1 = self.w2.T @ dz2 * self.deriv_relu(z1)
        dw1 = 1 / m * dz1 @ X
        db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

        return dw1, db1, dw2, db2

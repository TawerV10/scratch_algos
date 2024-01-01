import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self.single_predict(x_test) for x_test in X_test]

        return predictions

    def distance_calculator(self, q, p, method='euclidean'):
        if method == 'euclidean':
            distance = np.sqrt(np.sum((q - p) ** 2))

        elif method == 'manhattan':
            distance = np.sum(np.abs(q - p))

        return distance

    def single_predict(self, x):
        # Compute distance from x to each point in X
        distances = [self.distance_calculator(x, x_train) for x_train in self.X_train]

        # Get indices of k-nearest training data points
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k-nearest training data points
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label among the k-nearest neighbors
        most_common = np.bincount(k_nearest_labels).argmax()

        return most_common

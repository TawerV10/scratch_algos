import numpy as np
from collections import Counter
from DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=10, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)

            X_sample, y_sample = self._bootstra_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def _bootstra_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)

        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        raw_predict = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(raw_predict, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])

        return predictions

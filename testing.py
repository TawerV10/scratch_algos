from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from KNearestNeighbours import KNN
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from NaiveBayes import NaiveBayes
from SupportVectorMachine import SVM
from PrincipalComponentAnalysis import PCA
from KMeans import KMeans
from NeuralNetwork import NeuralNetwork

def knn():
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    plt.figure()
    plt.scatter(X[:, 2], X[:, 3], c=y, cmap=ListedColormap(['#336699', '#CC3366', '#669933']), edgecolors='k', s=20)
    plt.show()

    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    accuracy = np.sum(predict == y_test) / len(y_test)
    print(accuracy)

def linear_regression():
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression(lr=0.1, n_iters=100)
    reg.fit(X_train, y_train)
    predict = reg.predict(X_test)

    mse = np.mean((y_test - predict) ** 2)
    mae = np.mean(np.abs(y_test - predict))
    print(f'MSE = {mse}, MAE = {mae}')

    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color='g', s=30)
    plt.scatter(X_test, y_test, color='r', s=30)
    plt.plot(X, reg.predict(X), color='black')
    plt.show()

def logistic_regression():
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(lr=0.01, n_iters=1000)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    accuracy = np.sum(predict == y_test) / len(y_test)
    print(accuracy)

def decision_tree():
    dataset = datasets.load_wine()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTree(min_samples_split=10, max_depth=100)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    accuracy = np.sum(predict == y_test) / len(y_test)
    print(accuracy)

def random_forest():
    dataset = datasets.load_wine()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForest(n_trees=10, min_samples_split=10, max_depth=100)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    accuracy = np.sum(predict == y_test) / len(y_test)
    print(accuracy)

def naive_bayes():
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    accuracy = np.sum(predict == y_test) / len(y_test)
    print(accuracy)

def svm():
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=2, random_state=42)
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = SVM(lr=0.001)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    accuracy = np.sum(predict == y_test) / len(y_test)
    print(accuracy)

    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualize_svm()

def pca():
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print(X.shape, X_projected.shape)

    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, alpha=0.8, edgecolors='none', cmap=plt.cm.get_cmap('viridis', 3))
    plt.show()

def kmeans():
    X, y = datasets.make_blobs(n_samples=500, n_features=2, centers=3, shuffle=True, random_state=42)
    clusters = len(np.unique(y))

    k = KMeans(K=clusters, max_iters=50, plot_steps=True)
    k.predict(X)
    k.plot()

def nn():
    dataset = datasets.load_digits()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nn = NeuralNetwork(lr=0.01, n_iters=150, hidden_l=33)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)

    accuracy = nn.accuracy(y_pred, y_test)
    print(accuracy)

def main():
    pass

if __name__ == '__main__':
    main()
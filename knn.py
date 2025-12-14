import numpy as np
import pandas as pd
import time
from collections import Counter


def manual_train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy_score_manual(y_true, y_pred):
    return np.mean(y_true == y_pred)


def kappa_manual_weighted(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))
    num_ratings = max_rating - min_rating + 1
    if num_ratings < 2: return 1.0
    conf_mat = np.zeros((num_ratings, num_ratings))
    for t, p in zip(y_true, y_pred):
        conf_mat[t - min_rating][p - min_rating] += 1
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i][j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    hist_true = np.sum(conf_mat, axis=1)
    hist_pred = np.sum(conf_mat, axis=0)
    n = np.sum(conf_mat)
    expected = np.outer(hist_true, hist_pred) / n
    numerator = np.sum(weights * conf_mat)
    denominator = np.sum(weights * expected)
    if denominator == 0: return 1.0
    return 1 - (numerator / denominator)



class StandardScalerScratch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)





class KNNClassifierScratch:
    def __init__(self, k=11, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        # KNN is a "lazy learner" - it just remembers the data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - x), axis=1)

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]









if __name__ == "__main__":
    print("Loading Data...")
    df = pd.read_csv("winequality_white.csv", sep=";")

    df["quality"] = df["quality"].replace({3: 4, 8: 7, 9: 7})

    X = df.drop("quality", axis=1).values
    y = df["quality"].values

    X_train_raw, X_test_raw, y_train, y_test = manual_train_test_split(X, y, test_size=0.2, seed=42)

    print("Normalizing Features...")
    scaler = StandardScalerScratch()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- MANUAL GRID SEARCH ---
    print("\nStarting Grid Search to find best Hyperparameters...")
    print(f"{'Metric':<12} | {'K':<3} | {'Accuracy':<10} | {'Kappa'}")
    print("-" * 45)

    best_score = 0
    best_params = {}

    # Test K values and both metrics
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 29]
    metrics = ['euclidean', 'manhattan']

    for metric in metrics:
        for k in k_values:
            knn = KNNClassifierScratch(k=k, metric=metric)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            acc = accuracy_score_manual(y_test, y_pred)
            kappa = kappa_manual_weighted(y_test, y_pred)

            print(f"{metric:<12} | {k:<3} | {acc * 100:.2f}%     | {kappa:.4f}")

            if kappa > best_score:
                best_score = kappa
                best_params = {'k': k, 'metric': metric, 'acc': acc}

    print("\n==========================================")
    print(" WINNER PARAMETERS")
    print("==========================================")
    print(f"Best K       : {best_params['k']}")
    print(f"Best Metric  : {best_params['metric']}")
    print(f"Best Accuracy: {best_params['acc'] * 100:.2f}%")
    print(f"Best Kappa   : {best_score:.4f}")
    print("==========================================")
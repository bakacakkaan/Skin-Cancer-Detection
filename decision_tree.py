import pandas as pd
import numpy as np
from collections import Counter
import random

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv("winequality_white.csv", sep=";")

# Keep only qualities 4, 5, 6, 7, 8 (remove very rare 3 and 9)
df = df[df["quality"].isin([4, 5, 6, 7, 8])]

print("Class counts:")
print(df["quality"].value_counts().sort_index())

# Features and labels
X = df.drop("quality", axis=1).values   # shape: (n_samples, n_features)
y = df["quality"].values                # shape: (n_samples,)

n_samples, n_features = X.shape

# Train / test split (80/20)
np.random.seed(42)
indices = np.random.permutation(n_samples)
train_size = int(0.8 * n_samples)

train_idx = indices[:train_size]
test_idx  = indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

feature_names = df.drop("quality", axis=1).columns


# ============================================================
# 2. DECISION TREE (CART, GINI) FROM SCRATCH
# ============================================================

def gini_impurity(y):
    """Compute Gini impurity of label vector y."""
    counts = Counter(y)
    total = len(y)
    impurity = 1.0
    for c in counts.values():
        p = c / total
        impurity -= p ** 2
    return impurity

def majority_class(y):
    """Return the most common label in y."""
    counts = Counter(y)
    return counts.most_common(1)[0][0]

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=5, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # number of features to consider per split
        self.tree_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features_
        self.tree_ = self._build_tree(X, y, depth=0)

    def _best_split(self, X, y):
        """Find best (feature, threshold) to split on."""
        n_samples, n_features = X.shape
        if n_samples < 2:
            return None, None

        # Choose random subset of features (for Random Forest behaviour)
        features_idx = np.random.choice(
            n_features,
            size=self.max_features,
            replace=False
        )

        best_feature = None
        best_threshold = None
        best_impurity = 1.0  # max gini

        current_impurity = gini_impurity(y)

        for feature in features_idx:
            # Sort by this feature
            values = X[:, feature]
            # Get unique sorted values
            unique_vals = np.unique(values)
            if len(unique_vals) == 1:
                continue

            # Try midpoints between unique values as thresholds
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for t in thresholds:
                left_mask = values <= t
                right_mask = values > t

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                # Weighted gini
                n_left = len(y_left)
                n_right = len(y_right)
                n_total = n_left + n_right

                g_left = gini_impurity(y_left)
                g_right = gini_impurity(y_right)

                impurity = (n_left / n_total) * g_left + (n_right / n_total) * g_right

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = t

        if best_feature is None:
            return None, None

        # Only split if it improves impurity
        if best_impurity >= current_impurity:
            return None, None

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        node = {}

        # Stopping conditions
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            node["leaf"] = True
            node["prediction"] = majority_class(y)
            return node

        feature, threshold = self._best_split(X, y)

        if feature is None:
            node["leaf"] = True
            node["prediction"] = majority_class(y)
            return node

        # Split
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            node["leaf"] = True
            node["prediction"] = majority_class(y)
            return node

        node["leaf"] = False
        node["feature"] = feature
        node["threshold"] = threshold

        node["left"] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["prediction"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])


# ============================================================
# 3. RANDOM FOREST FROM SCRATCH
# ============================================================

class RandomForest:
    def __init__(self, n_trees=20, max_depth=10, min_samples_split=5, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _get_max_features(self, n_features):
        if isinstance(self.max_features, int):
            return self.max_features
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        # default: all
        return n_features

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = []
        max_feats = self._get_max_features(n_features)

        for i in range(self.n_trees):
            print(f"Training tree {i+1}/{self.n_trees}...")
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feats
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from all trees
        all_preds = np.array([tree.predict(X) for tree in self.trees])  # shape: (n_trees, n_samples)
        all_preds = all_preds.T  # shape: (n_samples, n_trees)

        # Majority vote
        final_preds = []
        for row in all_preds:
            counts = Counter(row)
            pred = counts.most_common(1)[0][0]
            final_preds.append(pred)

        return np.array(final_preds)


# ============================================================
# 4. TRAIN RANDOM FOREST AND EVALUATE
# ============================================================

rf = RandomForest(
    n_trees=25,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt'
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("\n=================================")
print(f"Random Forest accuracy: {accuracy * 100:.2f}%")
print("=================================\n")

# Confusion table
confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print("Confusion table:")
print(confusion)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("winequality_white.csv", sep=";")

print("Columns:", list(df.columns))

# Optionally drop very rare classes 3 and 9
df = df[df["quality"].isin([4, 5, 6, 7, 8])]

print("\nClass counts:")
print(df["quality"].value_counts().sort_index())

# ============================================================
# 2. FEATURES & TARGET
# ============================================================

X = df.drop("quality", axis=1).values     # all 11 chemical features
y = df["quality"].values                  # multi-class: 4,5,6,7,8

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3. SCALING (for logistic regression)
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# 4. MULTINOMIAL LOGISTIC REGRESSION
# ============================================================

log_reg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=2000
)

print("\nTraining Multinomial Logistic Regression...")
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
print("\n===================================")
print(f"Logistic Regression Accuracy: {acc_lr * 100:.2f}%")
print("===================================\n")

print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

print("Confusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_lr))

# ============================================================
# 5. DECISION TREE
# ============================================================

tree = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

print("\nTraining Decision Tree...")
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print("\n===================================")
print(f"Decision Tree Accuracy: {acc_tree * 100:.2f}%")
print("===================================\n")

print("Classification Report (Decision Tree):")
print(classification_report(y_test, y_pred_tree))

print("Confusion Matrix (Decision Tree):")
print(confusion_matrix(y_test, y_pred_tree))

# ============================================================
# 6. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

print("\nTraining Random Forest...")
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("\n===================================")
print(f"Random Forest Accuracy: {acc_rf * 100:.2f}%")
print("===================================\n")

print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

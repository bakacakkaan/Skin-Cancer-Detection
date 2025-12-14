# Diabetes Risk Classification (Statistical Learning from Scratch)

Multi-class classification project to predict diabetes stage:
- **No Diabetes**
- **Pre-Diabetes**
- **Type 2 Diabetes**

Implemented core ML algorithms **from scratch** in Python using only essential libraries (NumPy/Pandas).

## Methods Implemented (from scratch)
- Logistic Regression (One-vs-Rest)
- k-Nearest Neighbors
- Decision Tree
- Gaussian Naive Bayes

## Dataset & Preprocessing
- Large dataset (100k+ samples, dozens of features) including demographic, lifestyle, and clinical variables.
- Preprocessing:
  - Removed highly diagnostic/leaky variables (e.g., risk score / HbA1c / etc.)
  - One-hot encoded categorical variables
  - Train/test split (80/20)
  - Handled class imbalance with class weighting

## Evaluation
Reported:
- Accuracy
- Confusion matrix
- Quadratic weighted Cohen’s Kappa

## Results Snapshot
> Fill with your final numbers (example format)

| Model | Accuracy | Weighted Kappa | Notes |
|------|----------|----------------|------|
| Logistic Regression (OvR) | XX% | X.XX | fast + balanced |
| k-NN | XX% | X.XX | expensive on large data |
| Decision Tree | XX% | X.XX | strong accuracy, slower |
| Naive Bayes | XX% | X.XX | very fast baseline |

## How to Run
Example:
```bash
pip install -r requirements.txt
python src/preprocess.py
python src/logistic_regression.py
python src/decision_tree.py
python src/knn.py
python src/naive_bayes.py

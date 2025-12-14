import pandas as pd
import matplotlib.pyplot as plt


# ================================
# 1. Load data
# ================================
df = pd.read_csv("winequality_white.csv", sep=";")

# ================================
# 2. Remove rare classes (3 and 9)
# ================================
df = df[(df["quality"] != 3) & (df["quality"] != 9)]

print("Remaining class counts:")
print(df["quality"].value_counts().sort_index())

# ================================
# 3. Compute correlation matrix
# ================================
corr = df.corr(numeric_only=True)

print("\nCorrelation matrix:")
print(corr)

# ================================
# 4. Plot heatmap
# ================================
plt.figure(figsize=(12, 8))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Map")
plt.show()

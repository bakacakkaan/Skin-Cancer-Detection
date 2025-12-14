import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================
# Note: The separator is a semicolon ";"
df = pd.read_csv("winequality_white.csv", sep=";")

print("Data Loaded Successfully!")
print(f"Shape: {df.shape} (Rows, Columns)")

# ==========================================
# 2. THE SANITY CHECK (Structure & Missing Values)
# ==========================================
print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics (Look for outliers) ---")
# .T transposes the table to make it easier to read
print(df.describe().T)

# Check for duplicates (Common in this dataset)
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
# Note: We usually keep duplicates in this specific dataset because
# different wines can technically have the same chemical signature.

# ==========================================
# 3. VISUALIZING THE TARGET IMBALANCE
# ==========================================
# This explains why your first model failed on classes 3, 4, and 8.
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Distribution of Wine Quality (The Imbalance Problem)')
plt.xlabel('Quality Score')
plt.ylabel('Count of Wines')
plt.grid(axis='y', alpha=0.3)
plt.show()

# ==========================================
# 4. CORRELATION HEATMAP
# ==========================================
# This tells us which features are "Linear" friends with Quality.
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()

# We mask the upper triangle because it's a mirror image (redundant)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            mask=mask,
            linewidths=0.5)
plt.title('Correlation Matrix: Which chemicals affect Quality?')
plt.show()

# ==========================================
# 5. BOXPLOTS: THE "WHY"
# ==========================================
# These plots show how the distribution of a chemical changes as quality increases.

# 5a. Alcohol vs Quality (The strongest predictor)
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='alcohol', data=df, palette='Blues')
plt.title('Alcohol Content vs. Wine Quality')
plt.show()
# Insight: You should see the boxes move UP as quality moves UP.

# 5b. Density vs Quality (Negative correlation)
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='density', data=df, palette='Reds')
plt.title('Density vs. Wine Quality')
plt.ylim(0.985, 1.005) # Zoom in to ignore outliers
plt.show()
# Insight: Higher quality wines tend to be less dense (lighter).

# 5c. Volatile Acidity vs Quality (The "Vinegar" factor)
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='volatile acidity', data=df, palette='Purples')
plt.title('Volatile Acidity vs. Wine Quality')
plt.show()
# Insight: High volatile acidity usually means lower quality (bad taste).

# ==========================================
# 6. DISTRIBUTION ANALYSIS (Outliers)
# ==========================================
# Residual sugar often has extreme outliers that confuse models
plt.figure(figsize=(10, 5))
sns.histplot(df['residual sugar'], kde=True, bins=30, color='orange')
plt.title('Distribution of Residual Sugar (Check for Skew)')
plt.xlabel('Residual Sugar')
plt.show()
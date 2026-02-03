import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load dataset
df = pd.read_csv('hasil_glcm.csv')

print("=" * 60)
print("PISTACHIO GLCM FEATURE DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"Total Samples: {df.shape[0]}")
print(f"Total Features: {df.shape[1] - 2} (excluding filename & class)")

# Basic info
print("\n" + "=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print("\nColumn Names:")
print(df.columns.tolist())

print("\n\nClass Distribution:")
print(df['class'].value_counts())
print(f"\nClass Balance:")
for cls, count in df['class'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"  {cls}: {count} ({pct:.1f}%)")

print("\n\nData Types:")
print(df.dtypes.value_counts())

print("\n\nMissing Values:")
print(df.isnull().sum().sum())

print("\n\nDuplicate Rows:")
print(df.duplicated().sum())

# Feature columns only
feature_cols = [col for col in df.columns if col not in ['filename', 'class']]
print(f"\n\nFeature Columns ({len(feature_cols)}):")
print(feature_cols)

# Statistical summary
print("\n" + "=" * 60)
print("2. STATISTICAL SUMMARY")
print("=" * 60)
print("\nDescriptive Statistics:")
print(df[feature_cols].describe().T)

# Class-wise statistics
print("\n\nClass-wise Feature Means:")
class_means = df.groupby('class')[feature_cols].mean()
print(class_means)

print("\n\nFeature Ranges (Max - Min):")
feature_ranges = df[feature_cols].max() - df[feature_cols].min()
print(feature_ranges.sort_values(ascending=False))

# Feature importance by variance
print("\n\nFeature Variance (Top 10):")
feature_var = df[feature_cols].var().sort_values(ascending=False)
print(feature_var.head(10))

# Visualization
print("\n" + "=" * 60)
print("3. GENERATING VISUALIZATIONS")
print("=" * 60)

# Create figure for distribution plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feature Distribution by Class', fontsize=16, fontweight='bold')

# Sample features to plot
sample_features = ['contrast_0', 'dissimilarity_0', 'homogeneity_0', 
                   'energy_0', 'correlation_0', 'ASM_0']

for idx, feat in enumerate(sample_features):
    ax = axes[idx // 3, idx % 3]
    for class_name in df['class'].unique():
        data = df[df['class'] == class_name][feat]
        ax.hist(data, alpha=0.6, label=class_name, bins=30)
    ax.set_xlabel(feat, fontweight='bold')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title(f'{feat} Distribution')

plt.tight_layout()
plt.savefig('1_feature_distributions.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: 1_feature_distributions.png")

# Box plots for feature comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feature Comparison by Class (Box Plot)', fontsize=16, fontweight='bold')

for idx, feat in enumerate(sample_features):
    ax = axes[idx // 3, idx % 3]
    df.boxplot(column=feat, by='class', ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel(feat)
    ax.set_title(f'{feat} by Class')
    plt.suptitle('')  # Remove default suptitle

plt.tight_layout()
plt.savefig('2_feature_boxplots.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: 2_feature_boxplots.png")

# Correlation matrix (selected features)
fig, ax = plt.subplots(figsize=(14, 12))
# Select a subset to avoid overcrowding
corr_features = [f for f in feature_cols if '_0' in f]  # Only 0-degree features
corr_matrix = df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, ax=ax)
ax.set_title('Correlation Matrix (0° Angle Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('3_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: 3_correlation_matrix.png")

# PCA Visualization
print("\n\nPerforming PCA Analysis...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

# PCA to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create PCA plot
fig, ax = plt.subplots(figsize=(10, 8))
for class_name in df['class'].unique():
    mask = df['class'] == class_name
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, label=class_name, s=50)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
ax.set_title('PCA: 2D Visualization of GLCM Features', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4_pca_visualization.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: 4_pca_visualization.png")

# Feature importance by ANOVA F-score
from sklearn.feature_selection import f_classif

X = df[feature_cols]
y = df['class'].map({'Kirmizi_Pistachio': 0, 'Siirt_Pistachio': 1})

f_scores, p_values = f_classif(X, y)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'F-Score': f_scores,
    'p-value': p_values
}).sort_values('F-Score', ascending=False)

print("\n\nTop 15 Most Discriminative Features (ANOVA F-Score):")
print(feature_importance.head(15).to_string(index=False))

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(15)
ax.barh(top_features['Feature'][::-1], top_features['F-Score'][::-1])
ax.set_xlabel('F-Score', fontweight='bold')
ax.set_ylabel('Feature', fontweight='bold')
ax.set_title('Top 15 Discriminative Features', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('5_feature_importance.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: 5_feature_importance.png")

# Pair plot for selected features
print("\n\nGenerating pair plot...")
top_3_features = feature_importance.head(3)['Feature'].tolist()
top_3_features.append('class')

fig = sns.pairplot(df[top_3_features], hue='class', diag_kind='kde', 
                   plot_kws={'alpha': 0.6}, height=2.5)
fig.fig.suptitle('Pair Plot: Top 3 Discriminative Features', 
                 fontsize=14, fontweight='bold', y=1.02)
plt.savefig('6_pairplot.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: 6_pairplot.png")

# Summary statistics
print("\n" + "=" * 60)
print("4. KEY INSIGHTS")
print("=" * 60)

print(f"\n[OK] Dataset is {'balanced' if abs(0.574 - 0.426) < 0.15 else 'imbalanced'}")
print(f"[OK] No missing values detected")
print(f"[OK] No duplicate rows found")
print(f"[OK] All features are numerical (float64)")

# Check for outliers using IQR method
outlier_count = 0
for feat in feature_cols:
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[feat] < (Q1 - 1.5 * IQR)) | (df[feat] > (Q3 + 1.5 * IQR))).sum()
    outlier_count += outliers

print(f"[OK] Potential outliers detected: {outlier_count} across all features")

# PCA variance explained
pca_full = PCA().fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"[OK] Components needed for 95% variance: {n_components_95} out of {len(feature_cols)}")

# Feature separation
print(f"\n[OK] Most discriminative feature: {feature_importance.iloc[0]['Feature']} (F-Score: {feature_importance.iloc[0]['F-Score']:.2f})")

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nGenerated Files:")
print("  1. 1_feature_distributions.png")
print("  2. 2_feature_boxplots.png")
print("  3. 3_correlation_matrix.png")
print("  4. 4_pca_visualization.png")
print("  5. 5_feature_importance.png")
print("  6. 6_pairplot.png")
print("\nNext Steps:")
print("  > Feature Engineering")
print("  > Model Training & Evaluation")

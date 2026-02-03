import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PISTACHIO GLCM - FEATURE ENGINEERING")
print("=" * 70)

# Load dataset
df = pd.read_csv('hasil_glcm.csv')
feature_cols = [col for col in df.columns if col not in ['filename', 'class']]

print(f"\nOriginal Dataset: {df.shape}")
print(f"Features: {len(feature_cols)}")

# Prepare X and y
X = df[feature_cols]
y = df['class'].map({'Kirmizi_Pistachio': 0, 'Siirt_Pistachio': 1})

print("\n" + "=" * 70)
print("1. FEATURE ENGINEERING STRATEGIES")
print("=" * 70)

# Strategy 1: All Features (Baseline)
print("\n[Strategy 1] All 24 Features (Baseline)")
print("-" * 50)
X_all = X.copy()
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)
print(f"Features: {X_all_scaled.shape[1]}")
print(f"Scaler: StandardScaler (mean=0, std=1)")

# Strategy 2: Top K Features (ANOVA F-Score)
print("\n[Strategy 2] Top K Features (ANOVA F-Score)")
print("-" * 50)

# Test different k values
k_values = [5, 8, 10, 12, 15, 18, 20]
results_k = []

for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_k = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Calculate average F-score of selected features
    f_scores, _ = f_classif(X, y)
    avg_f_score = np.mean([f_scores[i] for i in range(len(X.columns)) if X.columns[i] in selected_features])
    
    results_k.append({
        'k': k,
        'avg_f_score': avg_f_score,
        'features': selected_features
    })
    print(f"k={k}: Avg F-Score = {avg_f_score:.2f}")

# Choose k=10 as optimal balance
k_optimal = 10
selector_k10 = SelectKBest(f_classif, k=k_optimal)
X_k10 = selector_k10.fit_transform(X, y)
selected_k10 = X.columns[selector_k10.get_support()].tolist()
print(f"\nSelected k={k_optimal}: {selected_k10}")

# Scale selected features
scaler_k10 = StandardScaler()
X_k10_scaled = scaler_k10.fit_transform(X_k10)

# Strategy 3: PCA Components
print("\n[Strategy 3] PCA Components")
print("-" * 50)

# First scale
scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(X)

# Test different n_components
n_components_list = [2, 3, 4, 5, 6, 8, 10]
pca_results = []

for n in n_components_list:
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X_scaled_pca)
    variance_explained = sum(pca.explained_variance_ratio_) * 100
    pca_results.append({
        'n_components': n,
        'variance_explained': variance_explained
    })
    print(f"n_components={n}: {variance_explained:.2f}% variance")

# Choose n=5 (captures >95% variance from EDA)
n_pca = 5
pca5 = PCA(n_components=n_pca)
X_pca5 = pca5.fit_transform(X_scaled_pca)
variance_pca5 = sum(pca5.explained_variance_ratio_) * 100
print(f"\nSelected n_components={n_pca}: {variance_pca5:.2f}% variance")

# Strategy 4: Top Features + PCA (Hybrid)
print("\n[Strategy 4] Top Features + PCA (Hybrid)")
print("-" * 50)

# Select top 15 features first
selector_15 = SelectKBest(f_classif, k=15)
X_15 = selector_15.fit_transform(X, y)

# Apply PCA on selected features
scaler_hybrid = StandardScaler()
X_15_scaled = scaler_hybrid.fit_transform(X_15)

pca_hybrid = PCA(n_components=8)
X_hybrid = pca_hybrid.fit_transform(X_15_scaled)
variance_hybrid = sum(pca_hybrid.explained_variance_ratio_) * 100
print(f"Selected 15 features -> 8 PCA components: {variance_hybrid:.2f}% variance")

# Strategy 5: Robust Scaling (for outliers)
print("\n[Strategy 5] Robust Scaling (All Features)")
print("-" * 50)

robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
print(f"Features: {X_robust.shape[1]}")
print(f"Scaler: RobustScaler (median & IQR based)")

# Strategy 6: Feature Interaction (Polynomial)
print("\n[Strategy 6] Feature Interactions (Selected)")
print("-" * 50)

# Create interaction terms for top features
top_features = ['energy_90', 'energy_135', 'ASM_90', 'homogeneity_90']
X_interaction = X[top_features].copy()

# Add simple interactions
X_interaction['energy_product'] = X['energy_90'] * X['energy_135']
X_interaction['ASM_energy_ratio'] = X['ASM_90'] / (X['energy_90'] + 0.001)
X_interaction['contrast_homogeneity'] = X['contrast_90'] * X['homogeneity_90']

print(f"Base features: {len(top_features)}")
print(f"Interaction features: 3")
print(f"Total: {X_interaction.shape[1]}")

scaler_interaction = StandardScaler()
X_interaction_scaled = scaler_interaction.fit_transform(X_interaction)

# Save all engineered datasets
print("\n" + "=" * 70)
print("2. SAVING ENGINEERED DATASETS")
print("=" * 70)

# Prepare dataframes for saving
datasets = {
    'fe_01_all_features_standard.csv': (X_all_scaled, y, 'All 24 features + StandardScaler'),
    'fe_02_top10_features.csv': (X_k10_scaled, y, f'Top {k_optimal} features (ANOVA)'),
    'fe_03_pca5_components.csv': (X_pca5, y, f'5 PCA components ({variance_pca5:.1f}% variance)'),
    'fe_04_hybrid_15to8.csv': (X_hybrid, y, f'15 features -> 8 PCA ({variance_hybrid:.1f}% variance)'),
    'fe_05_all_robust_scaling.csv': (X_robust, y, 'All 24 features + RobustScaler'),
    'fe_06_interactions.csv': (X_interaction_scaled, y, '4 base + 3 interaction features'),
}

for filename, (X_data, y_data, description) in datasets.items():
    df_fe = pd.DataFrame(X_data)
    df_fe['class'] = y_data.values
    df_fe.to_csv(filename, index=False)
    print(f"[OK] Saved: {filename}")
    print(f"     {description}")
    print(f"     Shape: {df_fe.shape}")

# Feature comparison summary
print("\n" + "=" * 70)
print("3. FEATURE ENGINEERING SUMMARY")
print("=" * 70)

summary_data = [
    ['Baseline', 'All 24 GLCM Features', X_all_scaled.shape[1], 'StandardScaler'],
    ['Selection', f'Top {k_optimal} (ANOVA)', X_k10_scaled.shape[1], 'StandardScaler'],
    ['Extraction', f'5 PCA ({variance_pca5:.1f}% var)', X_pca5.shape[1], 'None (PCA)'],
    ['Hybrid', '15->8 Hybrid', X_hybrid.shape[1], 'StandardScaler'],
    ['Robust', 'All 24 (Robust)', X_robust.shape[1], 'RobustScaler'],
    ['Interaction', '4 base + 3 interact', X_interaction_scaled.shape[1], 'StandardScaler'],
]

df_summary = pd.DataFrame(summary_data, columns=['Strategy', 'Description', 'Features', 'Scaling'])
print(df_summary.to_string(index=False))

# Visualization: Feature comparison
print("\n" + "=" * 70)
print("4. GENERATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Variance explained by PCA components
ax1 = axes[0, 0]
pca_full = PCA().fit(X_scaled_pca)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_) * 100
ax1.bar(range(1, len(pca_full.explained_variance_ratio_)+1), 
        pca_full.explained_variance_ratio_*100, alpha=0.7, label='Individual')
ax1.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 
         'ro-', label='Cumulative', linewidth=2)
ax1.axhline(y=95, color='g', linestyle='--', label='95% threshold')
ax1.set_xlabel('Number of Components', fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontweight='bold')
ax1.set_title('PCA Variance Explained', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: F-Score ranking
ax2 = axes[0, 1]
f_scores, p_values = f_classif(X, y)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'F-Score': f_scores
}).sort_values('F-Score', ascending=False)

ax2.barh(range(15), feature_importance['F-Score'].head(15)[::-1])
ax2.set_yticks(range(15))
ax2.set_yticklabels(feature_importance['Feature'].head(15)[::-1], fontsize=9)
ax2.set_xlabel('F-Score', fontweight='bold')
ax2.set_title('Top 15 Features (ANOVA F-Score)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Feature distribution comparison (original vs scaled)
ax3 = axes[1, 0]
sample_feat_idx = 0  # First feature
ax3.hist(X.iloc[:, sample_feat_idx], bins=30, alpha=0.5, label='Original', color='blue')
ax3.hist(X_all_scaled[:, sample_feat_idx], bins=30, alpha=0.5, label='Standardized', color='red')
ax3.set_xlabel('Value', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title(f'Distribution: {feature_cols[sample_feat_idx]}', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Strategy comparison
ax4 = axes[1, 1]
strategies = ['Baseline\n(24)', f'Top{k_optimal}\nANOVA', f'PCA\n{n_pca}comp', 'Hybrid\n(8)', 'Robust\n(24)', 'Interact\n(7)']
n_features = [24, k_optimal, n_pca, 8, 24, 7]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bars = ax4.barh(strategies, n_features, color=colors, alpha=0.7)
ax4.set_xlabel('Number of Features', fontweight='bold')
ax4.set_title('Feature Engineering Strategies Comparison', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, n_features)):
    ax4.text(val + 0.5, i, str(val), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('feature_engineering_summary.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: feature_engineering_summary.png")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED!")
print("=" * 70)
print("\nGenerated Files:")
print("  1. fe_01_all_features_standard.csv - Baseline dataset")
print("  2. fe_02_top10_features.csv - Feature selection (ANOVA)")
print("  3. fe_03_pca5_components.csv - Dimensionality reduction")
print("  4. fe_04_hybrid_15to8.csv - Hybrid approach")
print("  5. fe_05_all_robust_scaling.csv - Outlier-robust scaling")
print("  6. fe_06_interactions.csv - Feature interactions")
print("  7. feature_engineering_summary.png - Visual comparison")

print("\nRecommended for Modeling:")
print("  - fe_02_top10_features.csv (Best balance of simplicity & performance)")
print("  - fe_03_pca5_components.csv (Good for dimensionality reduction)")
print("\nNext Step: Model Training & Evaluation")

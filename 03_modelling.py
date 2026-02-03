import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PISTACHIO CLASSIFICATION - MODEL TRAINING & EVALUATION")
print("=" * 80)

# Load all engineered datasets
datasets = {
    'Baseline (24 feat)': 'fe_01_all_features_standard.csv',
    'Top-10 ANOVA': 'fe_02_top10_features.csv',
    'PCA (5 comp)': 'fe_03_pca5_components.csv',
    'Hybrid (8 feat)': 'fe_04_hybrid_15to8.csv',
    'Robust (24 feat)': 'fe_05_all_robust_scaling.csv',
    'Interactions (7 feat)': 'fe_06_interactions.csv',
}

print("\n[Loading Datasets]")
loaded_data = {}
for name, file in datasets.items():
    df = pd.read_csv(file)
    X = df.drop('class', axis=1).values
    y = df['class'].values
    loaded_data[name] = (X, y)
    print(f"  {name}: {X.shape}")

# Define models
print("\n" + "=" * 80)
print("1. MODEL DEFINITIONS")
print("=" * 80)

models = {
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Naive Bayes': GaussianNB(),
}

print(f"\nTotal Models: {len(models)}")
for name, model in models.items():
    print(f"  - {name}")

# Evaluation on all datasets
print("\n" + "=" * 80)
print("2. MODEL EVALUATION (All Datasets)")
print("=" * 80)

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for dataset_name, (X, y) in loaded_data.items():
    print(f"\n[{dataset_name}]")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    dataset_results = []
    
    for model_name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # ROC-AUC (if probability available)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = np.nan
        
        dataset_results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'CV-Acc': cv_mean,
            'CV-Std': cv_std,
            'ROC-AUC': roc_auc,
        })
        
        print(f"  {model_name:20s}: Acc={acc:.4f}, F1={f1:.4f}, CV={cv_mean:.4f} (+/-{cv_std:.4f})")
    
    results.extend(dataset_results)

# Create results dataframe
df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("3. BEST MODELS PER DATASET")
print("=" * 80)

best_models = df_results.loc[df_results.groupby('Dataset')['F1-Score'].idxmax()]
print(best_models[['Dataset', 'Model', 'Accuracy', 'F1-Score', 'CV-Acc', 'ROC-AUC']].to_string(index=False))

# Hyperparameter tuning for best models
print("\n" + "=" * 80)
print("4. HYPERPARAMETER TUNING (Top Models)")
print("=" * 80)

# Use Top-10 ANOVA dataset for tuning (best balance)
X_tune, y_tune = loaded_data['Top-10 ANOVA']
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_tune, y_tune, test_size=0.2, random_state=42, stratify=y_tune
)

tuned_results = []

# SVM Tuning
print("\n[SVM Hyperparameter Tuning]")
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
svm_grid.fit(X_train_t, y_train_t)
best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test_t)
svm_f1 = f1_score(y_test_t, y_pred_svm)
svm_acc = accuracy_score(y_test_t, y_pred_svm)
print(f"  Best Params: {svm_grid.best_params_}")
print(f"  Test Accuracy: {svm_acc:.4f}")
print(f"  Test F1-Score: {svm_f1:.4f}")

tuned_results.append({
    'Model': 'SVM (Tuned)',
    'Accuracy': svm_acc,
    'F1-Score': svm_f1,
    'Best Params': str(svm_grid.best_params_)
})

# Random Forest Tuning
print("\n[Random Forest Hyperparameter Tuning]")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_t, y_train_t)
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test_t)
rf_f1 = f1_score(y_test_t, y_pred_rf)
rf_acc = accuracy_score(y_test_t, y_pred_rf)
print(f"  Best Params: {rf_grid.best_params_}")
print(f"  Test Accuracy: {rf_acc:.4f}")
print(f"  Test F1-Score: {rf_f1:.4f}")

tuned_results.append({
    'Model': 'Random Forest (Tuned)',
    'Accuracy': rf_acc,
    'F1-Score': rf_f1,
    'Best Params': str(rf_grid.best_params_)
})

# KNN Tuning
print("\n[KNN Hyperparameter Tuning]")
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
knn_grid.fit(X_train_t, y_train_t)
best_knn = knn_grid.best_estimator_
y_pred_knn = best_knn.predict(X_test_t)
knn_f1 = f1_score(y_test_t, y_pred_knn)
knn_acc = accuracy_score(y_test_t, y_pred_knn)
print(f"  Best Params: {knn_grid.best_params_}")
print(f"  Test Accuracy: {knn_acc:.4f}")
print(f"  Test F1-Score: {knn_f1:.4f}")

tuned_results.append({
    'Model': 'KNN (Tuned)',
    'Accuracy': knn_acc,
    'F1-Score': knn_f1,
    'Best Params': str(knn_grid.best_params_)
})

print("\n" + "=" * 80)
print("5. FINAL MODEL COMPARISON")
print("=" * 80)

df_tuned = pd.DataFrame(tuned_results)
print(df_tuned.to_string(index=False))

# Final evaluation on best model
print("\n" + "=" * 80)
print("6. BEST MODEL - DETAILED EVALUATION")
print("=" * 80)

# Find best model
best_model_name = df_tuned.loc[df_tuned['F1-Score'].idxmax(), 'Model']
print(f"\nBest Model: {best_model_name}")

if 'SVM' in best_model_name:
    best_model = best_svm
elif 'Random Forest' in best_model_name:
    best_model = best_rf
else:
    best_model = best_knn

# Final predictions
y_pred_final = best_model.predict(X_test_t)
y_proba_final = best_model.predict_proba(X_test_t)[:, 1]

# Classification report
print("\n[Classification Report]")
print(classification_report(y_test_t, y_pred_final, 
                           target_names=['Kirmizi_Pistachio', 'Siirt_Pistachio']))

# Confusion Matrix
cm = confusion_matrix(y_test_t, y_pred_final)
print("\n[Confusion Matrix]")
print(cm)

# Visualization
print("\n" + "=" * 80)
print("7. GENERATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(20, 14))

# Plot 1: Model Comparison (Accuracy)
ax1 = plt.subplot(3, 3, 1)
df_pivot = df_results.pivot(index='Model', columns='Dataset', values='Accuracy')
df_pivot.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Model Accuracy Comparison (All Datasets)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_xlabel('Model', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')

# Plot 2: F1-Score Comparison
ax2 = plt.subplot(3, 3, 2)
df_pivot_f1 = df_results.pivot(index='Model', columns='Dataset', values='F1-Score')
df_pivot_f1.plot(kind='bar', ax=ax2, width=0.8)
ax2.set_title('Model F1-Score Comparison (All Datasets)', fontweight='bold', fontsize=11)
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_xlabel('Model', fontweight='bold')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')

# Plot 3: Confusion Matrix
ax3 = plt.subplot(3, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
           xticklabels=['Kirmizi', 'Siirt'], 
           yticklabels=['Kirmizi', 'Siirt'])
ax3.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold', fontsize=11)
ax3.set_ylabel('True Label', fontweight='bold')
ax3.set_xlabel('Predicted Label', fontweight='bold')

# Plot 4: ROC Curve
ax4 = plt.subplot(3, 3, 4)
fpr, tpr, _ = roc_curve(y_test_t, y_proba_final)
roc_auc = auc(fpr, tpr)
ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('False Positive Rate', fontweight='bold')
ax4.set_ylabel('True Positive Rate', fontweight='bold')
ax4.set_title(f'ROC Curve - {best_model_name}', fontweight='bold', fontsize=11)
ax4.legend(loc="lower right")
ax4.grid(True, alpha=0.3)

# Plot 5: Tuned Models Comparison
ax5 = plt.subplot(3, 3, 5)
models_tuned = df_tuned['Model'].tolist()
acc_tuned = df_tuned['Accuracy'].tolist()
f1_tuned = df_tuned['F1-Score'].tolist()
x = np.arange(len(models_tuned))
width = 0.35
bars1 = ax5.bar(x - width/2, acc_tuned, width, label='Accuracy', alpha=0.8)
bars2 = ax5.bar(x + width/2, f1_tuned, width, label='F1-Score', alpha=0.8)
ax5.set_ylabel('Score', fontweight='bold')
ax5.set_title('Tuned Models Performance', fontweight='bold', fontsize=11)
ax5.set_xticks(x)
ax5.set_xticklabels(models_tuned, rotation=15, ha='right')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: CV Scores for Top-10 Dataset
ax6 = plt.subplot(3, 3, 6)
top10_results = df_results[df_results['Dataset'] == 'Top-10 ANOVA']
top10_results = top10_results.sort_values('CV-Acc', ascending=True)
ax6.barh(top10_results['Model'], top10_results['CV-Acc'], xerr=top10_results['CV-Std'], 
        alpha=0.8, color='steelblue', error_kw={'capsize': 5})
ax6.set_xlabel('CV Accuracy (5-fold)', fontweight='bold')
ax6.set_title('Cross-Validation Scores (Top-10 ANOVA)', fontweight='bold', fontsize=11)
ax6.grid(True, alpha=0.3, axis='x')

# Plot 7: Feature Importance (if Random Forest or Gradient Boosting is best)
ax7 = plt.subplot(3, 3, 7)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [f'F{i+1}' for i in range(len(importances))]
    ax7.barh(range(len(importances)), importances[indices], color='coral', alpha=0.7)
    ax7.set_yticks(range(len(importances)))
    ax7.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax7.set_xlabel('Importance', fontweight='bold')
    ax7.set_title('Feature Importance', fontweight='bold', fontsize=11)
    ax7.grid(True, alpha=0.3, axis='x')
else:
    ax7.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax7.set_title('Feature Importance', fontweight='bold', fontsize=11)
    ax7.axis('off')

# Plot 8: Dataset Comparison (Best Model per Dataset)
ax8 = plt.subplot(3, 3, 8)
best_per_dataset = df_results.loc[df_results.groupby('Dataset')['F1-Score'].idxmax()]
ax8.barh(best_per_dataset['Dataset'], best_per_dataset['F1-Score'], 
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'], alpha=0.8)
ax8.set_xlabel('Best F1-Score', fontweight='bold')
ax8.set_title('Best Model Performance per Dataset', fontweight='bold', fontsize=11)
ax8.grid(True, alpha=0.3, axis='x')

# Plot 9: Precision-Recall Trade-off
ax9 = plt.subplot(3, 3, 9)
baseline_metrics = df_results[df_results['Dataset'] == 'Baseline (24 feat)']
baseline_metrics = baseline_metrics.sort_values('Precision')
ax9.plot(baseline_metrics['Precision'], baseline_metrics['Recall'], 
        'o-', markersize=10, linewidth=2)
for idx, row in baseline_metrics.iterrows():
    ax9.annotate(row['Model'], (row['Precision'], row['Recall']), 
                fontsize=7, alpha=0.7)
ax9.set_xlabel('Precision', fontweight='bold')
ax9.set_ylabel('Recall', fontweight='bold')
ax9.set_title('Precision-Recall Trade-off (Baseline)', fontweight='bold', fontsize=11)
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation_results.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: model_evaluation_results.png")

# Save results
print("\n" + "=" * 80)
print("8. SAVING RESULTS")
print("=" * 80)

df_results.to_csv('model_results_all.csv', index=False)
print("[OK] Saved: model_results_all.csv")

df_tuned.to_csv('model_results_tuned.csv', index=False)
print("[OK] Saved: model_results_tuned.csv")

# Summary
print("\n" + "=" * 80)
print("9. FINAL SUMMARY")
print("=" * 80)

print(f"\nBest Model: {best_model_name}")
print(f"Dataset: Top-10 ANOVA Features")
print(f"Test Accuracy: {df_tuned['Accuracy'].max():.4f}")
print(f"Test F1-Score: {df_tuned['F1-Score'].max():.4f}")
print(f"\nModel trained on {len(X_train_t)} samples, tested on {len(X_test_t)} samples")
print(f"Class distribution - Training: {np.bincount(y_train_t)}, Test: {np.bincount(y_test_t)}")

print("\n" + "=" * 80)
print("MODEL TRAINING & EVALUATION COMPLETED!")
print("=" * 80)

print("\nGenerated Files:")
print("  1. model_results_all.csv - All models on all datasets")
print("  2. model_results_tuned.csv - Hyperparameter-tuned models")
print("  3. model_evaluation_results.png - Comprehensive visualizations")

print("\nRecommendations:")
print(f"  - Use {best_model_name} for production deployment")
print("  - Top-10 ANOVA features provide best balance of performance & simplicity")
print("  - Consider feature selection based on feature importance plot")

Pistachio Variety Classification using GLCM Texture Analysis
End-to-end machine learning pipeline for automated classification of two pistachio varieties (Kirmizi and Siirt) using GLCM texture feature extraction.
 Project Overview
This project achieves 91.4% accuracy in classifying pistachio varieties using texture analysis, feature engineering, and ensemble learning methods.
 Dataset
- **Source:** [Pistachio Image Dataset - Kaggle](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset/data)
- **Samples:** 2,148 images (1,232 Kirmizi + 916 Siirt)
- **Resolution:** 600x600 RGB images
 Tech Stack
- **Language:** Python
- **ML/DL:** Scikit-learn, Scikit-image, OpenCV
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
 Key Features
- Extracted 24 GLCM texture features (4 angles × 6 properties)
- 6 feature engineering strategies (ANOVA, PCA, Hybrid, etc.)
- 7 ML algorithms evaluated across 6 datasets
- Best model: Random Forest (91.4% accuracy, 90.0% F1-score)
 Results
| Metric | Score |
|--------|-------|
| Accuracy | 91.4% |
| F1-Score | 90.0% |
| ROC-AUC | 0.9705 |
| CV Accuracy | 91.5% ± 1.5% |
 Project Structure
pistachio-image-mining/
├── pistcahio-glcm.ipynb       # Original feature extraction
├── 01_eda.py                  # Exploratory data analysis
├── 02_feature_engineering.py  # Feature engineering strategies
├── 03_modelling.py            # Model training & evaluation
├── model_results_all.csv      # Complete model results
├── *.png                      # Analysis visualizations
└── README.md                  # This file
## Best Model
- **Algorithm:** Random Forest
- **Features:** 15 selected → 8 PCA components (Hybrid approach)
- **Training Samples:** 1,718
- **Test Samples:** 430
## Methodology
### 1. Feature Extraction
- Applied GLCM (Gray-Level Co-occurrence Matrix) algorithm
- Extracted 6 texture properties: contrast, dissimilarity, homogeneity, energy, correlation, ASM
- Calculated at 4 angles: 0°, 45°, 90°, 135°
- Total: 24 features per image
### 2. Feature Engineering
- Implemented 6 different strategies:
  - Baseline (all 24 features)
  - ANOVA-based selection (top 10 features)
  - PCA dimensionality reduction (5 components, 97.7% variance)
  - Hybrid approach (15 features → 8 PCA components)
  - Robust scaling (outlier handling)
  - Feature interactions
### 3. Model Training
- Evaluated 7 algorithms:
  - SVM (RBF & Linear kernels)
  - Random Forest
  - K-Nearest Neighbors
  - Logistic Regression
  - Gradient Boosting
  - Naive Bayes
- Used 5-fold stratified cross-validation
- Performed hyperparameter tuning with GridSearchCV
### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC analysis
- Confusion Matrix
- Cross-validation scores
## Visualizations
The project includes 8 visualization files:
1. `1_feature_distributions.png` - Feature distribution by class
2. `2_feature_boxplots.png` - Box plots for feature comparison
3. `3_correlation_matrix.png` - Correlation matrix (0° angle features)
4. `4_pca_visualization.png` - 2D PCA visualization
5. `5_feature_importance.png` - Top 15 discriminative features
6. `6_pairplot.png` - Pair plot of top features
7. `feature_engineering_summary.png` - Feature engineering strategies comparison
8. `model_evaluation_results.png` - Comprehensive model evaluation
## Author
**Muhammad Dawam Amali**  
Data Mining & Machine Learning Engineer
## License
This project is open source and available for educational purposes.

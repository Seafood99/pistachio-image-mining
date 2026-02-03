Pistachio Variety Classification using GLCM Texture Analysis
End-to-end machine learning pipeline for automated classification of two pistachio varieties (Kirmizi and Siirt) using GLCM texture feature extraction.
Project Overview
This project achieves **91.4% accuracy** in classifying pistachio varieties using texture analysis, feature engineering, and ensemble learning methods.
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
## Author
**Muhammad Dawam Amali**  
Data Mining & Machine Learning Engineer
## License
This project is open source and available for educational purposes.
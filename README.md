# 🥜 GLCM Feature Extraction for Pistachio Image Dataset

This project extracts **texture features** from pistachio images using **GLCM (Gray-Level Co-occurrence Matrix)** and saves them into a structured CSV file.

You can download the dataset in kaggel 
"https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset/data"

## 📂 Dataset
The dataset comes from the `Pistachio_Image_Dataset` and contains categorized pistachio images such as:
- `Kirmizi_Pistachio`
- `Siirt_Pistachio`

## 🧪 Extracted Features
Using GLCM properties at 4 angles (0°, 45°, 90°, 135°):
- `contrast`
- `dissimilarity`
- `homogeneity`
- `energy`
- `correlation`
- `ASM`

Each property is extracted at each angle, resulting in 24 features per image.

## 🛠️ Dependencies
```bash
pip install numpy pandas scikit-image opencv-python tqdm

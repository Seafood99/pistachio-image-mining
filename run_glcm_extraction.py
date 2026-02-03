import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
from skimage.color import rgb2gray
import os
from tqdm import tqdm
from skimage.io import imread

# Path dataset (sesuaikan dengan path Anda)
dataset_path = 'Pistachio_Image_Dataset-20260203T175131Z-3-001/Pistachio_Image_Dataset/Pistachio_Image_Dataset'
output_path = 'hasil_glcm.csv'

print(f"Dataset path: {os.path.abspath(dataset_path)}")
print(f"Output path: {os.path.abspath(output_path)}")
print("="*70)

def load_and_preprocess(image_path):
    """Load image dan preprocess ke grayscale 224x224"""
    img = imread(image_path)
    if img.ndim == 3:  # Jika RGB
        img = rgb2gray(img)
    img_resized = resize(img, (224, 224), anti_aliasing=True)
    img_uint8 = (img_resized * 255).astype(np.uint8)
    return img_uint8

def extract_glcm_features(img_gray):
    """Ekstrak fitur GLCM dari gambar grayscale"""
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
    distances = [1]

    glcm = graycomatrix(img_gray, distances=distances, angles=angles,
                         symmetric=True, normed=True)

    features = {}
    for angle_idx, angle in enumerate(["0", "45", "90", "135"]):
        features[f'contrast_{angle}'] = graycoprops(glcm, 'contrast')[0, angle_idx]
        features[f'dissimilarity_{angle}'] = graycoprops(glcm, 'dissimilarity')[0, angle_idx]
        features[f'homogeneity_{angle}'] = graycoprops(glcm, 'homogeneity')[0, angle_idx]
        features[f'energy_{angle}'] = graycoprops(glcm, 'energy')[0, angle_idx]
        features[f'correlation_{angle}'] = graycoprops(glcm, 'correlation')[0, angle_idx]
        features[f'ASM_{angle}'] = graycoprops(glcm, 'ASM')[0, angle_idx]

    return features

# Main execution
results = []
error_count = 0
processed_count = 0

print("\nScanning dataset structure...")
for root, dirs, files in os.walk(dataset_path):
    class_name = os.path.basename(root)
    if class_name in ['Kirmizi_Pistachio', 'Siirt_Pistachio']:
        print(f"\nProcessing class: {class_name}")
        print(f"  Files found: {len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])}")
        
for root, dirs, files in os.walk(dataset_path):
    for file in tqdm(files, desc="Processing images"):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, file)
            try:
                img = load_and_preprocess(path)
                feats = extract_glcm_features(img)
                feats['filename'] = os.path.basename(path)
                # Tambahkan label kelas berdasarkan nama folder
                class_name = os.path.basename(root)
                feats['class'] = class_name
                results.append(feats)
                processed_count += 1
            except Exception as e:
                print(f"\nError di {file}: {e}")
                error_count += 1

# Simpan ke CSV
print("\n" + "="*70)
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Hasil disimpan di: {os.path.abspath(output_path)}")
    print(f"✓ Total gambar yang diproses: {len(results)}")
    print(f"✓ Total error: {error_count}")
    print(f"\n✓ Data frame shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns[:5])}... (total {len(df.columns)} columns)")
    
    # Show class distribution
    print("\n✓ Class distribution:")
    print(df['class'].value_counts())
    
    # Show sample features
    print("\n✓ Sample extracted features (first row):")
    print(df.iloc[0])
    
else:
    print("Tidak ada hasil yang diperoleh. Cek path dataset Anda.")

print("\n" + "="*70)

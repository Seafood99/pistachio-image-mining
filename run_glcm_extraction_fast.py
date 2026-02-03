import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imread
import os
from tqdm import tqdm
import time

# Path dataset
dataset_path = 'Pistachio_Image_Dataset-20260203T175131Z-3-001/Pistachio_Image_Dataset/Pistachio_Image_Dataset'
output_path = 'hasil_glcm.csv'

print("="*70)
print(f"Dataset path: {os.path.abspath(dataset_path)}")
print(f"Output path: {os.path.abspath(output_path)}")
print("="*70)

def load_and_preprocess(image_path):
    """Load image and preprocess to grayscale 224x224"""
    img = imread(image_path)
    if img.ndim == 3:  # If RGB
        img = rgb2gray(img)
    img_resized = resize(img, (224, 224), anti_aliasing=True)
    img_uint8 = (img_resized * 255).astype(np.uint8)
    return img_uint8

def extract_glcm_features(img_gray):
    """Extract GLCM features from grayscale image"""
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

# First, collect all image paths
print("\nScanning dataset structure...")
image_paths = []
class_counts = {}

for root, dirs, files in os.walk(dataset_path):
    class_name = os.path.basename(root)
    if class_name in ['Kirmizi_Pistachio', 'Siirt_Pistachio']:
        jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(jpg_files)
        for file in jpg_files:
            image_paths.append((os.path.join(root, file), class_name))

print(f"\nTotal images found: {len(image_paths)}")
print(f"Class distribution:")
for cls, count in class_counts.items():
    print(f"  {cls}: {count}")

# Process images
print("\n" + "="*70)
print("Processing images and extracting GLCM features...")
print("="*70)

results = []
error_count = 0
start_time = time.time()

for idx, (img_path, class_name) in enumerate(tqdm(image_paths, desc="Progress")):
    try:
        img = load_and_preprocess(img_path)
        feats = extract_glcm_features(img)
        feats['filename'] = os.path.basename(img_path)
        feats['class'] = class_name
        results.append(feats)
        
        # Save checkpoint every 100 images
        if len(results) % 100 == 0:
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(output_path + '.tmp', index=False)
            
    except Exception as e:
        error_count += 1
        if error_count <= 5:  # Only show first 5 errors
            print(f"\nError processing {os.path.basename(img_path)}: {e}")

elapsed_time = time.time() - start_time

# Save final results
print("\n" + "="*70)
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    # Remove temp file if exists
    if os.path.exists(output_path + '.tmp'):
        os.remove(output_path + '.tmp')
    
    print(f"\n✓ Results saved to: {os.path.abspath(output_path)}")
    print(f"✓ Total images processed: {len(results)}")
    print(f"✓ Total errors: {error_count}")
    print(f"✓ Processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"✓ Average time per image: {elapsed_time/len(results):.3f} seconds")
    print(f"\n✓ DataFrame shape: {df.shape} (rows, columns)")
    print(f"✓ Features per image: {len(df.columns) - 2} (excluding filename and class)")
    
    print("\n✓ Column names:")
    for i, col in enumerate(df.columns):
        if i < 10:
            print(f"  {col}")
        elif i == 10:
            print(f"  ... and {len(df.columns) - 10} more columns")
            break
    
    print("\n✓ Class distribution:")
    print(df['class'].value_counts())
    
    print("\n✓ Sample features (first 3 rows, first 8 columns):")
    print(df.iloc[:3, :8].to_string())
    
    print("\n✓ Feature statistics (first 8 features):")
    print(df.iloc[:, 2:10].describe())
    
else:
    print("No results obtained. Please check your dataset path.")

print("="*70)

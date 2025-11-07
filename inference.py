##########################################
#
##INFERENCE - Process all images and calculate root lengths
#
####################################################
from os import listdir
from xml.etree import ElementTree as ET
import numpy as np
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from PIL import Image
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from matplotlib.patches import Rectangle
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import skimage.io
from skimage.transform import resize
from skimage.measure import find_contours
import warnings
import json
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

ROOT_DIR = os.path.abspath("./")

class PredictionConfig(Config):
    NAME = "roots_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

def calculate_root_length(mask):
    """Calculate root length from mask using skeleton/contour"""
    if mask.sum() == 0:
        return 0.0
    
    # Find contours
    contours = find_contours(mask, 0.5)
    if len(contours) == 0:
        return 0.0
    
    # Use the longest contour
    max_length = 0
    for contour in contours:
        # Calculate length along the contour
        length = 0
        for i in range(len(contour) - 1):
            p1 = contour[i]
            p2 = contour[i + 1]
            length += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        max_length = max(max_length, length)
    
    return max_length

def process_image(model, image_path, output_dir, image_name):
    """Process a single image and return results"""
    print(f"  Processing: {image_name}")
    
    # Load image
    image = skimage.io.imread(image_path)
    
    # Run detection
    results = model.detect([image])[0]
    
    # Extract results
    masks = results['masks']
    boxes = results['rois']
    class_ids = results['class_ids']
    scores = results['scores']
    
    num_roots = len(class_ids)
    print(f"    Detected {num_roots} roots")
    
    # Calculate lengths for each detected root
    root_lengths = []
    for i in range(num_roots):
        mask = masks[:, :, i]
        length = calculate_root_length(mask)
        root_lengths.append(length)
        print(f"      Root {i+1}: length = {length:.2f} pixels, confidence = {scores[i]:.3f}")
    
    # Save visualization
    fig, ax = pyplot.subplots(1, figsize=(12, 12))
    CLASS_NAMES = ['BG', 'primary_root']
    
    display_instances(image, boxes, masks, class_ids, CLASS_NAMES,
                     scores, title=f"{image_name} - {num_roots} roots detected",
                     figAx=(fig, ax))
    
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_result.png")
    pyplot.savefig(output_path, bbox_inches='tight', dpi=150)
    pyplot.close()
    
    # Return results
    return {
        'image_name': image_name,
        'image_path': image_path,
        'num_roots_detected': num_roots,
        'root_lengths_pixels': root_lengths,
        'total_length_pixels': sum(root_lengths),
        'mean_length_pixels': np.mean(root_lengths) if root_lengths else 0,
        'confidence_scores': scores.tolist(),
        'mean_confidence': float(np.mean(scores)) if len(scores) > 0 else 0
    }

# Main execution
print("="*80)
print("ROOT PHENOTYPING INFERENCE")
print("="*80)

# Create config
cfg = PredictionConfig()

# Define and load model
print("\nLoading model...")
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)

model_path = os.path.join(ROOT_DIR, 'root_mask_rcnn_trained.h5')
if not os.path.exists(model_path):
    print(f"ERROR: Model weights not found at {model_path}")
    exit()

model.load_weights(model_path, by_name=True)
print("Model loaded successfully!")

# Process test directories
test_root = os.path.join(ROOT_DIR, 'MaskRoot/Test_files/Root_files')
print(f"\nScanning test directory: {test_root}")

if not os.path.exists(test_root):
    print(f"ERROR: Test directory not found: {test_root}")
    exit()

# Create output directory
output_root = os.path.join(ROOT_DIR, 'inference_results')
os.makedirs(output_root, exist_ok=True)

# Collect all results
all_results = []
summary_by_directory = {}

# Process each subdirectory
subdirs = sorted([d for d in listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
print(f"Found {len(subdirs)} subdirectories to process\n")

for subdir in subdirs:
    subdir_path = os.path.join(test_root, subdir)
    print(f"\nProcessing directory: {subdir}")
    
    # Create output directory for this subdir
    output_dir = os.path.join(output_root, subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files (both .jpg and .JPG)
    files = listdir(subdir_path)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"  No image files found in {subdir}")
        continue
    
    print(f"  Found {len(image_files)} images")
    
    # Process each image
    dir_results = []
    for img_file in sorted(image_files):
        img_path = os.path.join(subdir_path, img_file)
        
        try:
            result = process_image(model, img_path, output_dir, img_file)
            result['directory'] = subdir
            dir_results.append(result)
            all_results.append(result)
        except Exception as e:
            print(f"    ERROR processing {img_file}: {str(e)}")
            continue
    
    # Save directory summary
    if dir_results:
        summary_by_directory[subdir] = {
            'num_images': len(dir_results),
            'total_roots_detected': sum(r['num_roots_detected'] for r in dir_results),
            'mean_roots_per_image': np.mean([r['num_roots_detected'] for r in dir_results]),
            'total_length_pixels': sum(r['total_length_pixels'] for r in dir_results),
            'mean_confidence': np.mean([r['mean_confidence'] for r in dir_results])
        }
        
        # Save detailed results for this directory
        df_dir = pd.DataFrame(dir_results)
        csv_path = os.path.join(output_dir, f'{subdir}_results.csv')
        df_dir.to_csv(csv_path, index=False)
        print(f"  Results saved to: {csv_path}")

# Save overall results
print("\n" + "="*80)
print("SAVING OVERALL RESULTS")
print("="*80)

# Save all results to CSV
if all_results:
    df_all = pd.DataFrame(all_results)
    csv_all_path = os.path.join(output_root, 'all_results.csv')
    df_all.to_csv(csv_all_path, index=False)
    print(f"\nAll results saved to: {csv_all_path}")
    
    # Save summary
    summary_path = os.path.join(output_root, 'summary_by_directory.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_by_directory, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # Print overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total directories processed: {len(summary_by_directory)}")
    print(f"Total images processed: {len(all_results)}")
    print(f"Total roots detected: {sum(r['num_roots_detected'] for r in all_results)}")
    print(f"Mean roots per image: {np.mean([r['num_roots_detected'] for r in all_results]):.2f}")
    print(f"Total root length: {sum(r['total_length_pixels'] for r in all_results):.2f} pixels")
    print(f"Mean confidence: {np.mean([r['mean_confidence'] for r in all_results]):.3f}")
    
    # Print per-directory summary
    print("\n" + "="*80)
    print("PER-DIRECTORY SUMMARY")
    print("="*80)
    for dir_name, summary in summary_by_directory.items():
        print(f"\n{dir_name}:")
        print(f"  Images: {summary['num_images']}")
        print(f"  Total roots: {summary['total_roots_detected']}")
        print(f"  Mean roots/image: {summary['mean_roots_per_image']:.2f}")
        print(f"  Total length: {summary['total_length_pixels']:.2f} pixels")
        print(f"  Mean confidence: {summary['mean_confidence']:.3f}")

print(f"\n{'='*80}")
print(f"All results saved to: {output_root}")
print(f"{'='*80}")

# Root Detection and Segmentation using Mask R-CNN (TF2.0 + Keras 2.2.8)

This project is part of my **Bachelor's Research Thesis**, aiming to detect and segment *primary roots* in plant images using a customized version of the [Mask R-CNN](https://arxiv.org/abs/1703.06870) model adapted for **TensorFlow 2.0** and **Keras 2.2.8**. The original codebase from [Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN) was modified for compatibility and to support training and inference on annotated root datasets.

## 🚀 Highlights
- ✅ Adapted for TensorFlow 2.0 and Keras 2.2.8
- ✅ XML-based polygon annotation parsing for roots
- ✅ Mask generation and bounding box extraction
- ✅ Transfer learning on COCO pre-trained weights
- ✅ Evaluation using mAP, Precision, Recall

## 🛠️ Setup and Installation

### 1. Create Virtual Environment
```bash
conda create -n root_detection python=3.7
conda activate root_detection
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Download Root Dataset
```bash
# Download the root dataset
wget https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip
# Unzip and place in Root Images folder
unzip TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip -d "Root Images"
```

### 4. Download COCO Weights
```bash
# Download COCO pre-trained weights
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
# Move to mrcnn directory
mv mask_rcnn_coco.h5 mrcnn/
```

### 5. Run Training
```bash
python Training.py
```

---

## 📁 Directory Structure
```
Root_Detection_Project/
│
├── mrcnn/                        # Modified MRCNN (TF2-compatible)
│   └── mask_rcnn_coco.h5        # Pre-trained COCO weights
├── Root Images/                  # Root dataset directory
├── Training.py                   # Custom training pipeline
├── Inference.py                  # Inference and evaluation code
├── root_mask_rcnn_trained.h5     # Saved trained weights
└── requirements.txt              # Dependencies
```
---

## 🧠 Dataset Format

### Annotations
- XML format
- Multiple plants per image
- Each plant can have multiple roots annotated with `<point x="..." y="..."/>`

### Classes
```python
CLASS_NAMES = ['BG', 'primary_root']
```

###  Training Pipeline (Training.py)
Key Components:
* RootsDataset: Parses XML, builds masks and bounding boxes
* RootsConfig: Custom training config for 1 root class
* Mask Generation: Uses polygon fill logic for segmenting roots

### 🔍 Inference Pipeline (Inference.py)
Performs:
* Detection using trained weights
* Visualization of:
    * Bounding boxes
    * Masks
    * Class labels and confidence
* Evaluation using mAP

### 📝 Citation
If this work helped you, please cite:
```bibtex
@misc{vyas_rootdetection_2023,
  title={Root Detection and Segmentation using Mask R-CNN in TensorFlow 2.0},
  author={Mayank Vyas},
  year={2023},
  journal={Bachelor's Thesis},
  howpublished={\url{https://www.overleaf.com/read/hmwjvyyqhhrx}},
}
```
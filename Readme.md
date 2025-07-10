# Root Detection and Segmentation using Mask R-CNN (TF2.0 + Keras 2.2.8)

This project is part of my **Bachelor's Research Thesis**, aiming to detect and segment *primary roots* in plant images using a customized version of the [Mask R-CNN](https://arxiv.org/abs/1703.06870) model adapted for **TensorFlow 2.0** and **Keras 2.2.8**. The original codebase from [Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN) was modified for compatibility and to support training and inference on annotated root datasets.

## 🚀 Highlights
- ✅ Adapted for TensorFlow 2.0 and Keras 2.2.8
- ✅ XML-based polygon annotation parsing for roots
- ✅ Mask generation and bounding box extraction
- ✅ Transfer learning on COCO pre-trained weights
- ✅ Evaluation using mAP, Precision, Recall

---

## 📁 Directory Structure
```
Root_Detection_Project/
│
├── mrcnn/                        # Modified MRCNN (TF2-compatible)
├── Training.py                   # Custom training pipeline
├── Inference.py                  # Inference and evaluation code
├── root_mask_rcnn_trained.h5     # Saved trained weights
├── mask_rcnn_coco.h5             # Pre-trained COCO weights
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

To Train:
1. Set your image/annotation path:
```python 
 dataset_dir = 'path/to/your/Root_dataset' 
 ```

2. Run training:
```python 
Training.py
```
* Loads COCO weights
* Fine-tunes heads on primary root
* Saves final weights to: `root_mask_rcnn_trained.h5`

### 🔍 Inference Pipeline (Inference.py)
Performs:
* Detection using trained weights
* Visualization of:
    * Bounding boxes
    * Masks
    * Class labels and confidence
* Evaluation using mAP

To Run: `Inference.py`

* Output is visualized using matplotlib
* `compute_ap()` is used to calculate:
    * Mean Average Precision (mAP)
    * Precision
    * Recall
    * Overlaps (IoU)


Install the requirements via: `pip install -r requirements.txt`

### 📝 Citation
If this work helped you, please cite:
```bibtex
Copy
Edit
@misc{vyas_rootdetection_2023,
  title={Root Detection and Segmentation using Mask R-CNN in TensorFlow 2.0},
  author={Mayank Vyas},
  year={2023},
  journal={Bachelor's Thesis},
  howpublished={\url{https://www.overleaf.com/read/hmwjvyyqhhrx}},
}
```
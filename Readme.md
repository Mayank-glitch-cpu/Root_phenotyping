# Root Detection and Segmentation using Mask R-CNN

This project is part of a **Bachelor's Research Thesis**, aiming to detect and segment *primary roots* in plant images using a customized version of the [Mask R-CNN](https://arxiv.org/abs/1703.06870) model adapted for **TensorFlow 2.0** and **Keras 2.2.8**. The original codebase from [Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN) was modified for compatibility and to support training and inference on annotated root datasets.

## 🚀 Highlights
- ✅ Adapted for TensorFlow 2.0 and Keras 2.2.8
- ✅ **Dockerized architecture** for easy deployment and automation
- ✅ Separate containers for training and inference
- ✅ Automated dataset download and preprocessing
- ✅ **Makefile** for simplified command-line operations
- ✅ XML-based polygon annotation parsing for roots
- ✅ Mask generation and bounding box extraction
- ✅ Transfer learning on COCO pre-trained weights
- ✅ **GPU support** for accelerated training (highly recommended)
- ✅ Comprehensive evaluation using mAP, Precision, Recall
- ✅ Generalized inference on any image directory

---

## 📋 Table of Contents
- [Quick Start (Docker)](#quick-start-docker)
- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
  - [Method 1: Docker (Recommended)](#method-1-docker-recommended)
  - [Method 2: Manual Setup](#method-2-manual-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Makefile Commands](#makefile-commands)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Performance Notes](#performance-notes)
- [Citation](#citation)

---

## 🚀 Quick Start (Docker)

The fastest way to get started with training and inference:

```bash
# 1. Check system requirements
make check

# 2. Build Docker images
make build-all

# 3. Train the model (requires GPU for reasonable training time)
make train

# 4. Run inference on your test images
# Place your images in ./test_images/ directory
make inference
```

That's it! Results will be available in `./inference_results/`

---

## 📦 Prerequisites

### For Docker Setup (Recommended)
- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 1.29 or higher)
- **NVIDIA Docker runtime** (for GPU support)
- **GPU**: NVIDIA GPU with CUDA support (highly recommended for training)
- **Disk Space**: At least 20 GB free space
- **RAM**: At least 16 GB (32 GB recommended for training)

### For Manual Setup
- **Python 3.7**
- **Conda** or **virtualenv**
- **CUDA 11.2** and **cuDNN 8** (for GPU support)
- **Git**

---

## 🛠️ Installation Methods

### Method 1: Docker (Recommended)

Docker provides an isolated, reproducible environment and is the easiest way to get started.

#### Step 1: Install Docker and NVIDIA Docker Runtime

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Docker runtime for GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Verify Installation:**
```bash
make check
# or manually:
docker run --rm --gpus all nvidia/cuda:11.2.2-base nvidia-smi
```

#### Step 2: Clone Repository

```bash
git clone https://github.com/Mayank-glitch-cpu/Root_phenotyping.git
cd Root_phenotyping
```

#### Step 3: Setup Directories

```bash
make setup
```

This creates the following directories:
- `./models/` - For trained model weights
- `./logs/` - For training logs and checkpoints
- `./inference_results/` - For inference outputs
- `./test_images/` - For your test images

---

### Method 2: Manual Setup

If you prefer not to use Docker:

#### 1. Create Virtual Environment
```bash
conda create -n root_detection python=3.7
conda activate root_detection
```

#### 2. Install Requirements
```bash
pip install -r requirements.txt
```

#### 3. Download Root Dataset
```bash
bash download_dataset.sh
# Or manually:
wget https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip
unzip TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip -d "Root Images"
```

#### 4. Download COCO Weights
```bash
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
mv mask_rcnn_coco.h5 mrcnn/
```

---

## 🎯 Usage

### Training

#### Using Docker (Recommended):

```bash
# Full automated training with dataset download
make train
```

This will:
1. Build the training Docker image if not already built
2. Download the dataset automatically (if not present)
3. Start training with GPU acceleration
4. Save the trained model to `./models/root_mask_rcnn_trained.h5`
5. Save training logs to `./logs/`

**⚠️ Important**: Training requires significant computational resources:
- **GPU**: Highly recommended (NVIDIA GPU with at least 8 GB VRAM)
- **Time**: 3-4 hours with GPU, 2-3 days on CPU
- **Disk**: ~10 GB for dataset and logs

#### Using Manual Setup:

```bash
conda activate root_detection
python Training.py
```

#### Monitor Training:

```bash
# View recent logs
make logs

# Or view TensorBoard logs
tensorboard --logdir=./logs
```

---

### Inference

#### Using Docker:

**On default test images:**
```bash
# Place your test images in ./test_images/
make inference
```

**On custom directory:**
```bash
make inference TEST_DIR=./my_custom_images
```

**Quick test:**
```bash
make test-inference
```

#### Using Manual Setup:

```bash
conda activate root_detection

# Default usage (processes MaskRoot/Test_files/Root_files/)
python inference.py

# Custom directory
python inference.py --test_dir ./my_images --output_dir ./my_results

# Process flat directory structure
python inference.py --test_dir ./images --output_dir ./results

# Process with recursive subdirectories
python inference.py --test_dir ./images --output_dir ./results --recursive
```

#### Inference Options:

```bash
python inference.py --help

Options:
  --test_dir      Directory containing test images (default: MaskRoot/Test_files/Root_files)
  --model_path    Path to trained model (default: root_mask_rcnn_trained.h5)
  --output_dir    Output directory for results (default: inference_results)
  --recursive     Process subdirectories recursively
```

---

## 🔧 Makefile Commands

The Makefile provides convenient commands for common operations:

### Building Images
```bash
make build-train        # Build training Docker image
make build-inference    # Build inference Docker image
make build-all          # Build both images
```

### Running Pipelines
```bash
make train              # Run training pipeline
make inference          # Run inference on test images
make test-inference     # Quick inference test
make full-pipeline      # Run complete training + inference
```

### Utilities
```bash
make setup              # Create required directories
make download-dataset   # Download dataset only
make logs               # View training logs
make check              # Check system requirements
```

### Development
```bash
make shell-train        # Open interactive shell in training container
make shell-inference    # Open interactive shell in inference container
```

### Cleanup
```bash
make clean              # Remove containers
make clean-results      # Remove inference results
make clean-all          # Remove everything (images, containers, volumes)
```

### Getting Help
```bash
make help               # Show all available commands
```

---

## 📁 Project Structure

```
Root_phenotyping/
├── Dockerfile.train              # Docker image for training
├── Dockerfile.inference          # Docker image for inference
├── docker-compose.yml            # Docker Compose configuration
├── Makefile                      # Automation commands
├── download_dataset.sh           # Dataset download script
├── requirements.txt              # Python dependencies
├── Training.py                   # Training script
├── inference.py                  # Inference script (generalized)
├── Readme.md                     # This file
│
├── mrcnn/                        # Modified Mask R-CNN library
│   ├── config.py
│   ├── model.py
│   ├── utils.py
│   ├── visualize.py
│   └── mask_rcnn_coco.h5        # Pre-trained COCO weights
│
├── configs/                      # Configuration files
│   └── root_train.yml
│
├── Root Images/                  # Training dataset (auto-downloaded)
│   ├── 0000/
│   ├── 0001/
│   └── ...
│
├── models/                       # Trained model weights
│   └── root_mask_rcnn_trained.h5
│
├── logs/                         # Training logs and checkpoints
│
├── test_images/                  # Your test images go here
│
└── inference_results/            # Inference outputs
    ├── all_results.csv
    ├── summary_by_directory.json
    └── [visualizations]
```

---

## 🎓 Advanced Usage

### Custom Training Configuration

Edit the `RootsConfig` class in `Training.py` to customize training:

```python
class RootsConfig(Config):
    NAME = "Roots_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 16          # Adjust based on your GPU memory
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # ... other parameters
```

### Running Inference on Different Image Types

The inference script supports various image formats and directory structures:

**Flat directory with images:**
```bash
python inference.py --test_dir ./all_images --output_dir ./results
```

**Hierarchical directory structure:**
```bash
python inference.py --test_dir ./root_dir --output_dir ./results --recursive
```

**Using different model:**
```bash
python inference.py --model_path ./my_model.h5 --test_dir ./images
```

### Docker Compose Advanced Usage

**Training with custom settings:**
```bash
docker-compose run --rm train python Training.py
```

**Inference with GPU:**
```bash
docker-compose run --rm --gpus all inference \
  python inference.py --test_dir test_images --output_dir inference_results
```

**Interactive debugging:**
```bash
docker-compose run --rm train /bin/bash
# Inside container:
python Training.py
```

---

## ⚡ Performance Notes

### Training Performance

| Hardware | Batch Size | Time per Epoch | Total Training Time |
|----------|-----------|----------------|---------------------|
| NVIDIA A100 (40GB) | 16 | ~10 min | ~3-4 hours (20 epochs) |
| NVIDIA V100 (16GB) | 8 | ~15 min | ~5-6 hours (20 epochs) |
| NVIDIA RTX 3090 | 8 | ~18 min | ~6-7 hours (20 epochs) |
| NVIDIA GTX 1080 Ti | 4 | ~25 min | ~8-10 hours (20 epochs) |
| CPU (32 cores) | 2 | ~3-4 hours | 2-3 days (20 epochs) |

**Recommendations:**
- **GPU Training**: Highly recommended. Training on CPU is extremely slow (2-3 days).
- **VRAM**: Minimum 8 GB for batch size 4-8
- **Storage**: SSD recommended for faster data loading
- **RAM**: Minimum 16 GB, 32 GB recommended

### Inference Performance

Inference is much faster:
- **GPU**: ~1-2 seconds per image
- **CPU**: ~10-15 seconds per image

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory Error**
```bash
# Solution: Reduce batch size in Training.py
IMAGES_PER_GPU = 4  # Reduce from 16 to 4 or 2
```

**2. Docker GPU not detected**
```bash
# Check NVIDIA Docker installation
docker run --rm --gpus all nvidia/cuda:11.2.2-base nvidia-smi

# Reinstall nvidia-docker2 if needed
sudo apt-get install --reinstall nvidia-docker2
sudo systemctl restart docker
```

**3. Dataset download fails**
```bash
# Download manually
wget https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip
unzip TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip -d "Root Images"
```

**4. Permission denied errors in Docker**
```bash
# Fix permissions
sudo chown -R $USER:$USER ./logs ./models ./inference_results
```

---

## 🧠 Dataset Format

### Annotations
- **Format**: XML (RSML format)
- **Structure**: Multiple plants per image, each with multiple roots
- **Coordinates**: Polyline points with `<point x="..." y="..."/>`

### Classes
```python
CLASS_NAMES = ['BG', 'primary_root']
```

### Training/Validation Split
- **Training**: Images 0000-3795 (3796 images)
- **Validation**: Images 3796+ (remaining images)

---

## 📊 Output Files

### Training Outputs
- `./models/root_mask_rcnn_trained.h5` - Trained model weights
- `./logs/` - Training checkpoints and TensorBoard logs

### Inference Outputs
- `inference_results/all_results.csv` - Detailed results for all images
- `inference_results/summary_by_directory.json` - Summary statistics
- `inference_results/[subdir]/` - Visualizations with bounding boxes and masks

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is part of academic research. Please cite if you use this work.

---

## 📝 Citation

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

---

## 🙏 Acknowledgments

- Original [Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN) by Matterport
- Dataset from [University of Nottingham Plant Images Database](https://plantimages.nottingham.ac.uk/)
- TensorFlow and Keras teams for the deep learning frameworks

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: Mayank Vyas

---

**Note**: GPU training is highly recommended. Training on CPU may take 2-3 days compared to 3-6 hours on a modern GPU.

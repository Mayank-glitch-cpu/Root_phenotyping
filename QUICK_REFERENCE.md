# Root Phenotyping - Quick Reference Guide

## 🚀 Quick Commands

### First Time Setup
```bash
# Check system
make check

# Create directories
make setup

# Build images
make build-all
```

### Training
```bash
# Train model (downloads dataset automatically)
make train

# Monitor training
make logs
```

### Inference
```bash
# Run inference on test_images/
make inference

# Run inference on custom directory
make inference TEST_DIR=./my_images

# Quick test
make test-inference
```

### Cleanup
```bash
# Remove containers
make clean

# Remove all Docker images
make clean-all

# Remove inference results
make clean-results
```

---

## 📂 Directory Layout

```
./models/                   # Place trained models here
./test_images/              # Place your test images here
./inference_results/        # Results appear here
./logs/                     # Training logs
```

---

## 🔧 Inference Script Options

```bash
# Basic usage
python inference.py

# Custom directory
python inference.py --test_dir ./images

# Custom output
python inference.py --output_dir ./results

# Recursive processing
python inference.py --test_dir ./images --recursive

# Custom model
python inference.py --model_path ./my_model.h5
```

---

## 🐳 Docker Commands

### Training
```bash
# Build training image
docker build -f Dockerfile.train -t root-phenotyping:train .

# Run training
docker-compose up train

# Interactive shell
make shell-train
```

### Inference
```bash
# Build inference image
docker build -f Dockerfile.inference -t root-phenotyping:inference .

# Run inference
docker-compose up inference

# Interactive shell
make shell-inference
```

---

## 📊 Expected Outputs

### Training
- `models/root_mask_rcnn_trained.h5` - Trained model (~245 MB)
- `logs/roots_cfg<timestamp>/` - Training checkpoints

### Inference
- `inference_results/all_results.csv` - Detailed metrics
- `inference_results/summary_by_directory.json` - Summary
- `inference_results/<dir>/<image>_result.png` - Visualizations

---

## ⚠️ Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size in `Training.py`
```python
IMAGES_PER_GPU = 4  # Reduce from 16
```

### Issue: Docker GPU not found
**Solution**: Check NVIDIA Docker
```bash
docker run --rm --gpus all nvidia/cuda:11.2.2-base nvidia-smi
```

### Issue: Permission denied on volumes
**Solution**: Fix permissions
```bash
sudo chown -R $USER:$USER ./logs ./models ./inference_results
```

### Issue: Dataset download fails
**Solution**: Download manually
```bash
wget https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip
unzip TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip -d "Root Images"
```

---

## 🎯 Performance Tips

1. **Use GPU for training** - 10-20x faster than CPU
2. **SSD storage** - Faster data loading
3. **Batch size** - Adjust based on GPU memory:
   - 8GB VRAM: batch size 4
   - 16GB VRAM: batch size 8
   - 32GB+ VRAM: batch size 16
4. **Monitor GPU usage**: `nvidia-smi -l 1`

---

## 📦 Requirements

- Docker 20.10+
- Docker Compose 1.29+
- NVIDIA Docker runtime
- GPU with 8GB+ VRAM (recommended)
- 20GB+ disk space
- 16GB+ RAM

---

## 🔗 Useful Links

- [Original Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN)
- [Dataset Source](https://plantimages.nottingham.ac.uk/)
- [Project Repository](https://github.com/Mayank-glitch-cpu/Root_phenotyping)

---

## 📞 Support

For issues or questions:
1. Check the main README.md
2. Review troubleshooting section
3. Open an issue on GitHub
4. Check logs: `make logs`

---

**Last Updated**: November 2025

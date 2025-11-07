# Docker Usage Guide - Root Phenotyping

This guide provides detailed instructions for using the Dockerized root phenotyping system.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Building Images](#building-images)
3. [Training Workflow](#training-workflow)
4. [Inference Workflow](#inference-workflow)
5. [Volume Mounts](#volume-mounts)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- Docker 20.10 or higher
- Docker Compose 1.29 or higher
- NVIDIA Docker runtime (for GPU support)

### Hardware Requirements
- **For Training**: NVIDIA GPU with 8+ GB VRAM (recommended)
- **For Inference**: GPU optional but recommended
- **Disk Space**: 20+ GB free
- **RAM**: 16+ GB (32 GB recommended for training)

### Verify Installation
```bash
# Check Docker
docker --version

# Check Docker Compose
docker-compose --version

# Check GPU support
docker run --rm --gpus all nvidia/cuda:11.2.2-base nvidia-smi
```

---

## Building Images

### Build Training Image
```bash
# Using Makefile (recommended)
make build-train

# Using Docker directly
docker build -f Dockerfile.train -t root-phenotyping:train .
```

### Build Inference Image
```bash
# Using Makefile (recommended)
make build-inference

# Using Docker directly
docker build -f Dockerfile.inference -t root-phenotyping:inference .
```

### Build Both Images
```bash
make build-all
```

### Image Sizes
- Training image: ~8-10 GB
- Inference image: ~7-9 GB

---

## Training Workflow

### Option 1: Using Makefile (Easiest)
```bash
# Complete training pipeline
make train
```

This will:
1. Create necessary directories
2. Download dataset automatically (if not present)
3. Start training with GPU
4. Save model to `./models/`
5. Save logs to `./logs/`

### Option 2: Using Docker Compose
```bash
# Run training service
docker-compose up train

# Run in background
docker-compose up -d train

# View logs
docker-compose logs -f train
```

### Option 3: Using Docker Run
```bash
docker run --gpus all \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/"Root Images":/workspace/"Root Images" \
  -v $(pwd)/models:/workspace/models \
  root-phenotyping:train
```

### Training Options

**Skip dataset download** (if you already have it):
```bash
docker run --gpus all \
  -v $(pwd)/"Root Images":/workspace/"Root Images" \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/models:/workspace/models \
  root-phenotyping:train \
  python Training.py
```

**Resume from checkpoint**:
Training automatically resumes from the last checkpoint in `./logs/`.

**Monitor training**:
```bash
# View logs
make logs

# Or use TensorBoard
tensorboard --logdir=./logs

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Expected Training Time
| GPU Model | Batch Size | Time per Epoch | Total (20 epochs) |
|-----------|-----------|----------------|-------------------|
| A100 40GB | 16 | ~10 min | ~3-4 hours |
| V100 16GB | 8 | ~15 min | ~5-6 hours |
| RTX 3090 | 8 | ~18 min | ~6-7 hours |
| GTX 1080 Ti | 4 | ~25 min | ~8-10 hours |

---

## Inference Workflow

### Prepare Test Images
```bash
# Create test directory
mkdir -p test_images

# Copy your images
cp /path/to/your/images/*.jpg test_images/
```

### Option 1: Using Makefile (Easiest)
```bash
# Run inference on ./test_images/
make inference

# Run on custom directory
make inference TEST_DIR=./my_images

# Quick test
make test-inference
```

### Option 2: Using Docker Compose
```bash
# Run inference service
docker-compose up inference
```

### Option 3: Using Docker Run
```bash
docker run --rm --gpus all \
  -v $(pwd)/test_images:/workspace/test_images \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/inference_results:/workspace/inference_results \
  root-phenotyping:inference \
  python inference.py --test_dir test_images --output_dir inference_results
```

### Option 4: Using Wrapper Script
```bash
# Run on default directory
./run_inference.sh

# Run on custom directory
./run_inference.sh -t ./my_images -o ./my_results

# See all options
./run_inference.sh --help
```

### Inference Options

**Flat directory structure** (all images in one folder):
```bash
docker run --rm --gpus all \
  -v $(pwd)/my_images:/workspace/test_images \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/results:/workspace/inference_results \
  root-phenotyping:inference \
  python inference.py --test_dir test_images --output_dir inference_results
```

**Recursive subdirectories**:
```bash
docker run --rm --gpus all \
  -v $(pwd)/my_images:/workspace/test_images \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/results:/workspace/inference_results \
  root-phenotyping:inference \
  python inference.py --test_dir test_images --output_dir inference_results --recursive
```

### Expected Inference Time
- **With GPU**: ~1-2 seconds per image
- **Without GPU**: ~10-15 seconds per image

---

## Volume Mounts

### Training Container Volumes
| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./logs` | `/workspace/logs` | Training logs and checkpoints |
| `./Root Images` | `/workspace/Root Images` | Training dataset |
| `./models` | `/workspace/models` | Trained model output |

### Inference Container Volumes
| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./test_images` | `/workspace/test_images` | Input test images |
| `./models` | `/workspace/models` | Trained model weights |
| `./inference_results` | `/workspace/inference_results` | Output results |

### Custom Volume Mounts
```bash
docker run --rm --gpus all \
  -v /path/on/host:/path/in/container \
  root-phenotyping:inference \
  python inference.py --test_dir /path/in/container
```

---

## Advanced Usage

### Interactive Shell

**Training container**:
```bash
make shell-train
# or
docker run --rm -it --gpus all \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/"Root Images":/workspace/"Root Images" \
  root-phenotyping:train /bin/bash
```

**Inference container**:
```bash
make shell-inference
# or
docker run --rm -it --gpus all \
  -v $(pwd)/test_images:/workspace/test_images \
  -v $(pwd)/models:/workspace/models \
  root-phenotyping:inference /bin/bash
```

### Custom Training Parameters

Edit `Training.py` before building, or mount a custom version:
```bash
docker run --gpus all \
  -v $(pwd)/custom_training.py:/workspace/Training.py \
  -v $(pwd)/logs:/workspace/logs \
  root-phenotyping:train \
  python Training.py
```

### CPU-Only Mode

Remove `--gpus all` flag:
```bash
docker run --rm \
  -v $(pwd)/test_images:/workspace/test_images \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/inference_results:/workspace/inference_results \
  root-phenotyping:inference \
  python inference.py --test_dir test_images
```

### Multi-GPU Training

Specify GPU IDs:
```bash
docker run --gpus '"device=0,1"' \
  -v $(pwd)/logs:/workspace/logs \
  root-phenotyping:train
```

### Background Execution

```bash
# Start training in background
docker-compose up -d train

# Check logs
docker-compose logs -f train

# Stop training
docker-compose down
```

---

## Troubleshooting

### Issue: Cannot find GPU

**Solution 1**: Check NVIDIA Docker runtime
```bash
docker run --rm --gpus all nvidia/cuda:11.2.2-base nvidia-smi
```

**Solution 2**: Reinstall NVIDIA Docker
```bash
sudo apt-get install --reinstall nvidia-docker2
sudo systemctl restart docker
```

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size in `Training.py`:
```python
class RootsConfig(Config):
    IMAGES_PER_GPU = 4  # Reduce from 16 to 4 or 2
```

Rebuild the image:
```bash
make build-train
```

### Issue: Permission Denied on Volumes

**Solution**: Fix permissions
```bash
sudo chown -R $USER:$USER ./logs ./models ./inference_results
chmod -R 755 ./logs ./models ./inference_results
```

### Issue: Container Exits Immediately

**Solution 1**: Check logs
```bash
docker-compose logs train
# or
docker logs root-phenotyping-train
```

**Solution 2**: Run interactively
```bash
make shell-train
# Then run commands manually inside container
```

### Issue: Model Not Found

**Solution**: Ensure model is in correct location
```bash
# Check model exists
ls -lh ./models/root_mask_rcnn_trained.h5

# Copy from root if needed
cp ./root_mask_rcnn_trained.h5 ./models/
```

### Issue: Dataset Download Fails

**Solution**: Download manually
```bash
wget https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip
unzip TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip -d "Root Images"
```

### Issue: Port Already in Use (if using web services)

**Solution**: Change port in `docker-compose.yml`:
```yaml
ports:
  - "8080:8080"  # Change 8080 to another port
```

---

## Best Practices

1. **Always use GPU for training** - CPU training is extremely slow
2. **Monitor GPU usage**: `watch -n 1 nvidia-smi`
3. **Use SSD for dataset** - Significantly faster I/O
4. **Clean up regularly**: `make clean` or `make clean-all`
5. **Backup models**: Copy `./models/` to safe location
6. **Version your images**: Tag Docker images with versions
7. **Check logs regularly** during training
8. **Test inference** on small batch before processing large datasets

---

## Resource Management

### Check Disk Usage
```bash
# Check Docker disk usage
docker system df

# Check workspace disk usage
du -sh .
```

### Clean Up Docker
```bash
# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Remove unused volumes
docker volume prune -f

# Clean everything (use with caution!)
docker system prune -a --volumes -f
```

### Clean Up Workspace
```bash
# Using Makefile
make clean          # Containers only
make clean-results  # Inference results
make clean-all      # Everything

# Manual cleanup
rm -rf ./inference_results/*
rm -rf ./logs/roots_cfg*/  # Keep only latest
```

---

## Production Deployment

### Using Docker Compose in Production

1. **Create production compose file**:
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  inference:
    image: root-phenotyping:inference
    restart: unless-stopped
    runtime: nvidia
    volumes:
      - /data/input:/workspace/test_images:ro
      - /data/output:/workspace/inference_results
      - /data/models:/workspace/models:ro
```

2. **Run production service**:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

3. **Monitor**:
```bash
docker-compose -f docker-compose.prod.yml logs -f
```

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Main README](./Readme.md)
- [Quick Reference](./QUICK_REFERENCE.md)

---

**Last Updated**: November 2025

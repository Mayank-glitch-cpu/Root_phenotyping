.PHONY: help build-train build-inference build-all train inference clean clean-all setup test-inference

# Default target
help:
	@echo "Root Phenotyping - Docker Automation"
	@echo "===================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make setup              - Create required directories"
	@echo "  make build-train        - Build training Docker image"
	@echo "  make build-inference    - Build inference Docker image"
	@echo "  make build-all          - Build both Docker images"
	@echo ""
	@echo "  make train              - Run training in Docker container"
	@echo "  make inference          - Run inference on test images"
	@echo "  make test-inference     - Run inference on sample images"
	@echo ""
	@echo "  make download-dataset   - Download root phenotyping dataset"
	@echo "  make logs               - View training logs"
	@echo "  make shell-train        - Open shell in training container"
	@echo "  make shell-inference    - Open shell in inference container"
	@echo ""
	@echo "  make clean              - Remove containers and temporary files"
	@echo "  make clean-all          - Remove everything (images, containers, volumes)"
	@echo "  make clean-results      - Remove inference results"
	@echo ""
	@echo "Advanced usage:"
	@echo "  make inference TEST_DIR=./my_images  - Run inference on custom directory"
	@echo ""

# Variables
TRAIN_IMAGE = root-phenotyping:train
INFERENCE_IMAGE = root-phenotyping:inference
MODELS_DIR = ./models
LOGS_DIR = ./logs
RESULTS_DIR = ./inference_results
TEST_DIR ?= ./test_images
DATASET_DIR = ./Root Images

# Setup required directories
setup:
	@echo "Creating required directories..."
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(LOGS_DIR)
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p $(TEST_DIR)
	@echo "Directories created successfully!"

# Build Docker images
build-train:
	@echo "Building training Docker image..."
	docker build -f Dockerfile.train -t $(TRAIN_IMAGE) .
	@echo "Training image built successfully!"

build-inference:
	@echo "Building inference Docker image..."
	docker build -f Dockerfile.inference -t $(INFERENCE_IMAGE) .
	@echo "Inference image built successfully!"

build-all: build-train build-inference
	@echo "All Docker images built successfully!"

# Download dataset
download-dataset:
	@echo "Downloading dataset..."
	@bash download_dataset.sh
	@echo "Dataset downloaded successfully!"

# Training
train: setup build-train
	@echo "Starting training..."
	@echo "Note: This will take several hours. GPU is highly recommended."
	docker-compose up train
	@echo "Training completed! Model saved to $(MODELS_DIR)/"

# Inference
inference: setup build-inference
	@echo "Starting inference on $(TEST_DIR)..."
	@if [ ! -f "$(MODELS_DIR)/root_mask_rcnn_trained.h5" ] && [ ! -f "./root_mask_rcnn_trained.h5" ]; then \
		echo "ERROR: Trained model not found!"; \
		echo "Please run 'make train' first or place root_mask_rcnn_trained.h5 in $(MODELS_DIR)/"; \
		exit 1; \
	fi
	@if [ ! -f "$(MODELS_DIR)/root_mask_rcnn_trained.h5" ] && [ -f "./root_mask_rcnn_trained.h5" ]; then \
		cp ./root_mask_rcnn_trained.h5 $(MODELS_DIR)/; \
		echo "Copied model to $(MODELS_DIR)/"; \
	fi
	docker-compose up inference
	@echo "Inference completed! Results saved to $(RESULTS_DIR)/"

# Test inference with sample
test-inference: setup build-inference
	@echo "Running test inference..."
	@echo "Make sure you have test images in $(TEST_DIR)/"
	@if [ ! -d "$(TEST_DIR)" ] || [ -z "$$(ls -A $(TEST_DIR) 2>/dev/null)" ]; then \
		echo "ERROR: Test directory is empty or doesn't exist!"; \
		echo "Please add some images to $(TEST_DIR)/"; \
		exit 1; \
	fi
	docker run --rm --gpus all \
		-v $(PWD)/$(TEST_DIR):/workspace/test_images \
		-v $(PWD)/$(MODELS_DIR):/workspace/models \
		-v $(PWD)/$(RESULTS_DIR):/workspace/inference_results \
		$(INFERENCE_IMAGE) \
		python inference.py --test_dir test_images --output_dir inference_results
	@echo "Test inference completed!"

# View logs
logs:
	@if [ -d "$(LOGS_DIR)" ]; then \
		echo "Training logs:"; \
		ls -lht $(LOGS_DIR) | head -10; \
	else \
		echo "No logs found. Run training first."; \
	fi

# Interactive shells
shell-train: build-train
	@echo "Opening shell in training container..."
	docker run --rm -it --gpus all \
		-v $(PWD)/logs:/workspace/logs \
		-v $(PWD)/$(DATASET_DIR):/workspace/Root\ Images \
		-v $(PWD)/$(MODELS_DIR):/workspace/models \
		$(TRAIN_IMAGE) /bin/bash

shell-inference: build-inference
	@echo "Opening shell in inference container..."
	docker run --rm -it --gpus all \
		-v $(PWD)/$(TEST_DIR):/workspace/test_images \
		-v $(PWD)/$(MODELS_DIR):/workspace/models \
		-v $(PWD)/$(RESULTS_DIR):/workspace/inference_results \
		$(INFERENCE_IMAGE) /bin/bash

# Cleanup
clean:
	@echo "Cleaning up containers and temporary files..."
	docker-compose down
	@echo "Cleanup completed!"

clean-results:
	@echo "Removing inference results..."
	rm -rf $(RESULTS_DIR)/*
	@echo "Results cleaned!"

clean-all: clean
	@echo "Removing all Docker images and volumes..."
	docker rmi $(TRAIN_IMAGE) $(INFERENCE_IMAGE) 2>/dev/null || true
	docker volume prune -f
	@echo "All cleaned up!"

# Full pipeline
full-pipeline: build-all train inference
	@echo "Full pipeline completed!"
	@echo "Training logs: $(LOGS_DIR)/"
	@echo "Trained model: $(MODELS_DIR)/root_mask_rcnn_trained.h5"
	@echo "Inference results: $(RESULTS_DIR)/"

# Check system
check:
	@echo "Checking system requirements..."
	@echo ""
	@echo "Docker version:"
	@docker --version || echo "ERROR: Docker not found!"
	@echo ""
	@echo "Docker Compose version:"
	@docker-compose --version || echo "ERROR: Docker Compose not found!"
	@echo ""
	@echo "NVIDIA Docker runtime:"
	@docker run --rm --gpus all nvidia/cuda:11.2.2-base nvidia-smi || echo "WARNING: GPU support not available"
	@echo ""
	@echo "Disk space:"
	@df -h . | tail -1
	@echo ""

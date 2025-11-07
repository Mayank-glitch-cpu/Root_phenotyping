#!/bin/bash
# Complete example workflow for Root Phenotyping

set -e

echo "=================================================="
echo "Root Phenotyping - Complete Example Workflow"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Check prerequisites
echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"
make check
echo -e "${GREEN}✓ Prerequisites check complete${NC}"
echo ""

# Step 2: Setup directories
echo -e "${BLUE}Step 2: Setting up directories...${NC}"
make setup
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 3: Build Docker images
echo -e "${BLUE}Step 3: Building Docker images...${NC}"
echo -e "${YELLOW}This may take 10-15 minutes...${NC}"
make build-all
echo -e "${GREEN}✓ Docker images built${NC}"
echo ""

# Step 4: Optional - Download dataset only (if you want to check it first)
echo -e "${BLUE}Step 4: Do you want to download the dataset now?${NC}"
echo "The dataset is ~5-8 GB and will be downloaded during training if not present."
read -p "Download now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    make download-dataset
    echo -e "${GREEN}✓ Dataset downloaded${NC}"
else
    echo -e "${YELLOW}Skipping dataset download (will download during training)${NC}"
fi
echo ""

# Step 5: Training
echo -e "${BLUE}Step 5: Training the model${NC}"
echo -e "${YELLOW}WARNING: Training will take 3-6 hours with GPU (or 2-3 days on CPU)${NC}"
read -p "Start training now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Starting training...${NC}"
    make train
    echo -e "${GREEN}✓ Training complete!${NC}"
    echo -e "${GREEN}Model saved to: ./models/root_mask_rcnn_trained.h5${NC}"
else
    echo -e "${YELLOW}Skipping training. You can run 'make train' later.${NC}"
fi
echo ""

# Step 6: Prepare test images
echo -e "${BLUE}Step 6: Preparing test images${NC}"
if [ ! -d "./test_images" ] || [ -z "$(ls -A ./test_images 2>/dev/null)" ]; then
    echo "Please add your test images to the ./test_images/ directory"
    echo "Example:"
    echo "  cp /path/to/your/images/*.jpg ./test_images/"
    echo ""
    read -p "Press Enter when ready to continue..."
else
    IMAGE_COUNT=$(find ./test_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    echo -e "${GREEN}Found $IMAGE_COUNT images in ./test_images/${NC}"
fi
echo ""

# Step 7: Run inference
echo -e "${BLUE}Step 7: Running inference${NC}"
if [ -f "./models/root_mask_rcnn_trained.h5" ] || [ -f "./root_mask_rcnn_trained.h5" ]; then
    read -p "Run inference now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Starting inference...${NC}"
        make inference
        echo -e "${GREEN}✓ Inference complete!${NC}"
        echo -e "${GREEN}Results saved to: ./inference_results/${NC}"
    else
        echo -e "${YELLOW}Skipping inference. You can run 'make inference' later.${NC}"
    fi
else
    echo -e "${YELLOW}Model not found. Please run training first with 'make train'${NC}"
fi
echo ""

# Step 8: View results
echo -e "${BLUE}Step 8: Viewing results${NC}"
if [ -d "./inference_results" ] && [ "$(ls -A ./inference_results 2>/dev/null)" ]; then
    echo "Results are available in:"
    echo "  - ./inference_results/all_results.csv (detailed results)"
    echo "  - ./inference_results/summary_by_directory.json (summary)"
    echo "  - ./inference_results/*_result.png (visualizations)"
    echo ""
    echo "To view summary:"
    if [ -f "./inference_results/all_results.csv" ]; then
        echo ""
        echo "=== Sample Results ==="
        head -5 ./inference_results/all_results.csv
        echo "..."
        echo "======================"
    fi
fi
echo ""

# Summary
echo "=================================================="
echo -e "${GREEN}Workflow Complete!${NC}"
echo "=================================================="
echo ""
echo "What's been set up:"
echo "  ✓ Docker images built"
echo "  ✓ Directory structure created"
if [ -f "./models/root_mask_rcnn_trained.h5" ]; then
    echo "  ✓ Model trained"
fi
if [ -d "./inference_results" ] && [ "$(ls -A ./inference_results 2>/dev/null)" ]; then
    echo "  ✓ Inference results generated"
fi
echo ""
echo "Next steps:"
echo "  - View results in ./inference_results/"
echo "  - Run more inference: make inference TEST_DIR=./my_images"
echo "  - Check logs: make logs"
echo "  - Clean up: make clean"
echo ""
echo "For help:"
echo "  - make help"
echo "  - See README.md"
echo "  - See QUICK_REFERENCE.md"
echo "  - See DOCKER_GUIDE.md"
echo ""
echo "=================================================="

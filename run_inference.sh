#!/bin/bash
# Simple wrapper script for running inference on any directory

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
TEST_DIR="./test_images"
OUTPUT_DIR="./inference_results"
MODEL_PATH="./models/root_mask_rcnn_trained.h5"
USE_DOCKER=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-dir)
            TEST_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --no-docker)
            USE_DOCKER=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --test-dir DIR     Directory with test images (default: ./test_images)"
            echo "  -o, --output-dir DIR   Output directory (default: ./inference_results)"
            echo "  -m, --model PATH       Model path (default: ./models/root_mask_rcnn_trained.h5)"
            echo "  --no-docker            Run without Docker (requires conda env)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -t ./my_images -o ./my_results"
            echo "  $0 --test-dir /path/to/images --no-docker"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Root Phenotyping Inference${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}ERROR: Test directory not found: $TEST_DIR${NC}"
    exit 1
fi

# Check if directory has images
IMAGE_COUNT=$(find "$TEST_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}ERROR: No images found in $TEST_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Found $IMAGE_COUNT images in $TEST_DIR${NC}"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ] && [ ! -f "./root_mask_rcnn_trained.h5" ]; then
    echo -e "${RED}ERROR: Model not found!${NC}"
    echo "Please ensure model exists at:"
    echo "  - $MODEL_PATH, or"
    echo "  - ./root_mask_rcnn_trained.h5, or"
    echo "  - ./models/root_mask_rcnn_trained.h5"
    echo ""
    echo "Run training first: make train"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

if [ "$USE_DOCKER" = true ]; then
    echo -e "${BLUE}Running inference using Docker...${NC}"
    echo ""
    
    # Get absolute paths
    ABS_TEST_DIR=$(realpath "$TEST_DIR")
    ABS_OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
    ABS_MODEL_DIR=$(dirname $(realpath "$MODEL_PATH" 2>/dev/null || echo "./models/root_mask_rcnn_trained.h5"))
    
    # Run Docker
    docker run --rm --gpus all \
        -v "$ABS_TEST_DIR:/workspace/test_images" \
        -v "$ABS_OUTPUT_DIR:/workspace/inference_results" \
        -v "$ABS_MODEL_DIR:/workspace/models" \
        root-phenotyping:inference \
        python inference.py --test_dir test_images --output_dir inference_results
else
    echo -e "${BLUE}Running inference without Docker...${NC}"
    echo ""
    
    # Check if conda environment exists
    if ! conda env list | grep -q "root_detection"; then
        echo -e "${RED}ERROR: Conda environment 'root_detection' not found${NC}"
        echo "Please create it first:"
        echo "  conda create -n root_detection python=3.7"
        echo "  conda activate root_detection"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
    
    # Run with conda
    conda run -n root_detection python inference.py \
        --test_dir "$TEST_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH"
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Inference completed successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - all_results.csv         (detailed results)"
echo "  - summary_by_directory.json (summary statistics)"
echo "  - *_result.png            (visualizations)"
echo ""

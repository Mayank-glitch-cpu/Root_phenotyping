#!/bin/bash
# Download and extract root phenotyping dataset

set -e

echo "========================================"
echo "Root Phenotyping Dataset Download"
echo "========================================"

DATASET_URL="https://plantimages.nottingham.ac.uk/datasets/TwMTc5BnBEcjUh2TLk4ESjFSyMe7eQc9wfsyxhrs.zip"
DATASET_DIR="Root Images"
TMP_ZIP="/tmp/root_dataset.zip"

# Check if dataset already exists
if [ -d "$DATASET_DIR" ] && [ "$(ls -A $DATASET_DIR)" ]; then
    echo "Dataset directory already exists and is not empty."
    echo "Skipping download..."
    exit 0
fi

# Create dataset directory
mkdir -p "$DATASET_DIR"

# Download dataset
echo "Downloading dataset from:"
echo "$DATASET_URL"
echo ""
wget -q --show-progress "$DATASET_URL" -O "$TMP_ZIP"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download dataset"
    exit 1
fi

echo ""
echo "Extracting dataset..."
unzip -q "$TMP_ZIP" -d "$DATASET_DIR"

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract dataset"
    rm -f "$TMP_ZIP"
    exit 1
fi

# Clean up
rm -f "$TMP_ZIP"

echo ""
echo "========================================"
echo "Dataset downloaded and extracted successfully!"
echo "Location: $DATASET_DIR"
echo "========================================"

# Count number of subdirectories
NUM_DIRS=$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Number of image directories: $NUM_DIRS"

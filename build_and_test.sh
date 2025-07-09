#!/bin/bash

# Build and test script for YOLO segmentation

set -e

echo "=== YOLO Segmentation Build and Test Script ==="

# Check if ONNX model exists
if [ ! -f "chpt/best.onnx" ]; then
    echo "ONNX model not found. Checking if we need to export from PyTorch model..."
    
    if [ -f "chpt/best.pt" ]; then
        echo "Found PyTorch model, exporting to ONNX..."
        python3 export_segmentation_onnx.py
    else
        echo "Error: Neither best.onnx nor best.pt found in chpt directory"
        echo "Please place your model file in the chpt directory"
        exit 1
    fi
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ../CMakeLists_segmentation.txt ..

# Build
echo "Building..."
make -j$(nproc)

# Run test
echo "Running segmentation test..."
./test_segmentation_onnx

echo "Test completed!"
echo "Check test_segmentation_result.jpg for visualization" 
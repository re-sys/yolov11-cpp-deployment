#!/bin/bash

# Model Analysis Runner Script

set -e

echo "=== YOLO Model Analysis Script ==="

# Check if we're in the right directory
if [ ! -d "scripts" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check if models exist
if [ ! -f "chpt/best.pt" ]; then
    echo "Error: PyTorch model not found at chpt/best.pt"
    exit 1
fi

if [ ! -f "chpt/best.onnx" ]; then
    echo "Error: ONNX model not found at chpt/best.onnx"
    echo "Please export the model first using export_segmentation_onnx.py"
    exit 1
fi

# Create output directories
mkdir -p comparison_results
mkdir -p layer_analysis

echo "Starting model comparison..."

# Run basic comparison
cd scripts
python3 compare_pt_onnx.py

echo ""
echo "Starting detailed layer analysis..."

# Run detailed layer analysis
python3 analyze_model_layers.py

echo ""
echo "=== Analysis Complete ==="
echo "Results saved in:"
echo "  - comparison_results/ (basic comparison)"
echo "  - layer_analysis/ (detailed layer analysis)"
echo ""
echo "Check the following files:"
echo "  - comparison_results/comparison_report.txt"
echo "  - layer_analysis/analysis_summary.txt"
echo "  - layer_analysis/analysis_report.json"
echo "  - Various visualization images in both directories" 
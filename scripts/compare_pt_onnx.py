#!/usr/bin/env python3
"""
Compare PyTorch and ONNX model outputs to check for export errors
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import onnxruntime as ort
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import time

class ModelComparator:
    def __init__(self, pt_path: str, onnx_path: str, device: str = 'cpu'):
        """
        Initialize model comparator
        
        Args:
            pt_path: Path to PyTorch model (.pt file)
            onnx_path: Path to ONNX model (.onnx file)
            device: Device to run PyTorch model on ('cpu' or 'cuda')
        """
        self.pt_path = pt_path
        self.onnx_path = onnx_path
        self.device = device
        
        # Load models
        self.pt_model = self.load_pytorch_model()
        self.onnx_session = self.load_onnx_model()
        
        # Store intermediate outputs
        self.pt_intermediates = {}
        self.onnx_intermediates = {}
        
    def load_pytorch_model(self) -> nn.Module:
        """Load PyTorch model and register hooks for intermediate outputs"""
        print(f"Loading PyTorch model from: {self.pt_path}")
        
        # Load model with weights_only=False for compatibility
        try:
            model = torch.load(self.pt_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=False: {e}")
            print("Trying with weights_only=True...")
            model = torch.load(self.pt_path, map_location=self.device, weights_only=True)
        
        # Handle different model formats
        if isinstance(model, dict):
            if 'model' in model:
                model = model['model']
            elif 'ema' in model:
                model = model['ema']
            else:
                print("Warning: Unknown model format, trying to use the first key")
                model = list(model.values())[0]
        
        model.eval()
        model.to(self.device)
        
        # Register hooks to capture intermediate outputs
        self.register_hooks(model)
        
        print("PyTorch model loaded successfully")
        return model
    
    def register_hooks(self, model: nn.Module):
        """Register hooks to capture intermediate layer outputs"""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.pt_intermediates[name] = output.detach().cpu().numpy()
                elif isinstance(output, (list, tuple)):
                    self.pt_intermediates[name] = [o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output]
            return hook
        
        # Register hooks for key layers (adjust based on your model architecture)
        for name, module in model.named_modules():
            if any(keyword in name.lower() for keyword in ['backbone', 'neck', 'head', 'detect', 'seg']):
                module.register_forward_hook(hook_fn(name))
    
    def load_onnx_model(self) -> ort.InferenceSession:
        """Load ONNX model"""
        print(f"Loading ONNX model from: {self.onnx_path}")
        
        # Create ONNX session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(self.onnx_path, providers=providers)
        
        print("ONNX model loaded successfully")
        return session
    
    def preprocess_image(self, image_path: str, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for both PyTorch and ONNX models
        
        Args:
            image_path: Path to input image
            input_size: Target input size (width, height)
            
        Returns:
            Tuple of (original_image, preprocessed_image)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        original_image = image.copy()
        
        # Preprocess
        resized = cv2.resize(image, input_size)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Convert to CHW format
        chw_image = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched_image = np.expand_dims(chw_image, axis=0)
        
        return original_image, batched_image
    
    def run_pytorch_inference(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """Run PyTorch model inference and capture intermediate outputs"""
        print("Running PyTorch inference...")
        
        # Clear previous intermediate outputs
        self.pt_intermediates.clear()
        
        # Convert to PyTorch tensor and match model precision
        pt_input = torch.from_numpy(input_tensor).to(self.device)
        
        # Check model precision and convert input if needed
        if hasattr(self.pt_model, 'dtype'):
            pt_input = pt_input.to(self.pt_model.dtype)
        else:
            # Try to infer precision from model parameters
            for param in self.pt_model.parameters():
                if param.dtype != torch.float32:
                    pt_input = pt_input.to(param.dtype)
                    break
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            pt_outputs = self.pt_model(pt_input)
        pt_time = time.time() - start_time
        
        print(f"PyTorch inference time: {pt_time:.3f}s")
        
        # Convert outputs to numpy
        if isinstance(pt_outputs, torch.Tensor):
            pt_outputs_np = pt_outputs.cpu().numpy()
        elif isinstance(pt_outputs, (list, tuple)):
            pt_outputs_np = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in pt_outputs]
        else:
            pt_outputs_np = pt_outputs
        
        return {
            'outputs': pt_outputs_np,
            'intermediates': self.pt_intermediates.copy(),
            'time': pt_time
        }
    
    def run_onnx_inference(self, input_tensor: np.ndarray) -> Dict[str, Any]:
        """Run ONNX model inference"""
        print("Running ONNX inference...")
        
        # Get input name
        input_name = self.onnx_session.get_inputs()[0].name
        output_names = [output.name for output in self.onnx_session.get_outputs()]
        
        # Run inference
        start_time = time.time()
        onnx_outputs = self.onnx_session.run(output_names, {input_name: input_tensor})
        onnx_time = time.time() - start_time
        
        print(f"ONNX inference time: {onnx_time:.3f}s")
        
        return {
            'outputs': onnx_outputs,
            'time': onnx_time
        }
    
    def compare_outputs(self, pt_results: Dict, onnx_results: Dict, tolerance: float = 1e-5) -> Dict[str, Any]:
        """
        Compare PyTorch and ONNX outputs
        
        Args:
            pt_results: PyTorch inference results
            onnx_results: ONNX inference results
            tolerance: Tolerance for numerical comparison
            
        Returns:
            Comparison results
        """
        print("\n=== Comparing Model Outputs ===")
        
        comparison_results = {
            'outputs_match': True,
            'max_diff': 0.0,
            'mean_diff': 0.0,
            'relative_diff': 0.0,
            'details': {}
        }
        
        pt_outputs = pt_results['outputs']
        onnx_outputs = onnx_results['outputs']
        
        # Compare main outputs
        if isinstance(pt_outputs, list) and isinstance(onnx_outputs, list):
            for i, (pt_out, onnx_out) in enumerate(zip(pt_outputs, onnx_outputs)):
                diff = np.abs(pt_out - onnx_out)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                relative_diff = np.mean(np.abs(diff / (pt_out + 1e-8)))
                
                comparison_results['details'][f'output_{i}'] = {
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'relative_diff': relative_diff,
                    'shapes': {'pt': pt_out.shape, 'onnx': onnx_out.shape},
                    'match': max_diff < tolerance
                }
                
                if max_diff >= tolerance:
                    comparison_results['outputs_match'] = False
                
                comparison_results['max_diff'] = max(comparison_results['max_diff'], max_diff)
                comparison_results['mean_diff'] = max(comparison_results['mean_diff'], mean_diff)
                comparison_results['relative_diff'] = max(comparison_results['relative_diff'], relative_diff)
                
                print(f"Output {i}:")
                print(f"  Shapes: PT {pt_out.shape} vs ONNX {onnx_out.shape}")
                print(f"  Max diff: {max_diff:.6f}")
                print(f"  Mean diff: {mean_diff:.6f}")
                print(f"  Relative diff: {relative_diff:.6f}")
                print(f"  Match: {'✓' if max_diff < tolerance else '✗'}")
        
        else:
            # Single output
            pt_out = pt_outputs
            onnx_out = onnx_outputs
            
            diff = np.abs(pt_out - onnx_out)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            relative_diff = np.mean(np.abs(diff / (pt_out + 1e-8)))
            
            comparison_results['details']['output_0'] = {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'relative_diff': relative_diff,
                'shapes': {'pt': pt_out.shape, 'onnx': onnx_out.shape},
                'match': max_diff < tolerance
            }
            
            comparison_results['max_diff'] = max_diff
            comparison_results['mean_diff'] = mean_diff
            comparison_results['relative_diff'] = relative_diff
            comparison_results['outputs_match'] = max_diff < tolerance
            
            print(f"Output shapes: PT {pt_out.shape} vs ONNX {onnx_out.shape}")
            print(f"Max diff: {max_diff:.6f}")
            print(f"Mean diff: {mean_diff:.6f}")
            print(f"Relative diff: {relative_diff:.6f}")
            print(f"Match: {'✓' if max_diff < tolerance else '✗'}")
        
        return comparison_results
    
    def analyze_intermediate_outputs(self) -> Dict[str, Any]:
        """Analyze intermediate layer outputs if available"""
        print("\n=== Analyzing Intermediate Outputs ===")
        
        analysis = {
            'pt_intermediates': len(self.pt_intermediates),
            'onnx_intermediates': len(self.onnx_intermediates),
            'details': {}
        }
        
        if self.pt_intermediates:
            print(f"PyTorch intermediate layers captured: {len(self.pt_intermediates)}")
            for name, output in self.pt_intermediates.items():
                if isinstance(output, np.ndarray):
                    print(f"  {name}: {output.shape}")
                elif isinstance(output, list):
                    print(f"  {name}: {[o.shape if isinstance(o, np.ndarray) else type(o) for o in output]}")
        
        if self.onnx_intermediates:
            print(f"ONNX intermediate layers captured: {len(self.onnx_intermediates)}")
            for name, output in self.onnx_intermediates.items():
                print(f"  {name}: {output.shape}")
        
        return analysis
    
    def generate_test_image(self, size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Generate a test image with simple shapes"""
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Add some shapes
        cv2.circle(image, (size[0]//2, size[1]//2), 100, (255, 0, 0), -1)
        cv2.rectangle(image, (100, 100), (300, 200), (0, 255, 0), -1)
        cv2.ellipse(image, (500, 400), (80, 60), 45, 0, 360, (0, 0, 255), -1)
        
        return image
    
    def save_comparison_visualization(self, pt_results: Dict, onnx_results: Dict, 
                                    comparison_results: Dict, output_dir: str = "comparison_results"):
        """Save comparison visualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison report
        report_path = os.path.join(output_dir, "comparison_report.txt")
        with open(report_path, 'w') as f:
            f.write("PyTorch vs ONNX Model Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"PyTorch inference time: {pt_results['time']:.3f}s\n")
            f.write(f"ONNX inference time: {onnx_results['time']:.3f}s\n")
            f.write(f"Speedup: {pt_results['time'] / onnx_results['time']:.2f}x\n\n")
            
            f.write(f"Outputs match: {comparison_results['outputs_match']}\n")
            f.write(f"Max difference: {comparison_results['max_diff']:.6f}\n")
            f.write(f"Mean difference: {comparison_results['mean_diff']:.6f}\n")
            f.write(f"Relative difference: {comparison_results['relative_diff']:.6f}\n\n")
            
            f.write("Detailed comparison:\n")
            for name, details in comparison_results['details'].items():
                f.write(f"{name}:\n")
                f.write(f"  Shapes: PT {details['shapes']['pt']} vs ONNX {details['shapes']['onnx']}\n")
                f.write(f"  Max diff: {details['max_diff']:.6f}\n")
                f.write(f"  Mean diff: {details['mean_diff']:.6f}\n")
                f.write(f"  Relative diff: {details['relative_diff']:.6f}\n")
                f.write(f"  Match: {details['match']}\n\n")
        
        print(f"Comparison report saved to: {report_path}")
        
        # Save output visualizations if possible
        try:
            pt_outputs = pt_results['outputs']
            onnx_outputs = onnx_results['outputs']
            
            if isinstance(pt_outputs, list):
                for i, (pt_out, onnx_out) in enumerate(zip(pt_outputs, onnx_outputs)):
                    if len(pt_out.shape) == 4 and pt_out.shape[1] <= 3:  # Image-like output
                        self.save_output_visualization(pt_out, onnx_out, i, output_dir)
            else:
                if len(pt_outputs.shape) == 4 and pt_outputs.shape[1] <= 3:
                    self.save_output_visualization(pt_outputs, onnx_outputs, 0, output_dir)
        except Exception as e:
            print(f"Warning: Could not save output visualizations: {e}")
    
    def save_output_visualization(self, pt_output: np.ndarray, onnx_output: np.ndarray, 
                                output_idx: int, output_dir: str):
        """Save output visualization"""
        # Take first batch and first channel
        pt_viz = pt_output[0, 0] if pt_output.shape[1] == 1 else pt_output[0, :3].transpose(1, 2, 0)
        onnx_viz = onnx_output[0, 0] if onnx_output.shape[1] == 1 else onnx_output[0, :3].transpose(1, 2, 0)
        
        # Normalize for visualization
        pt_viz = (pt_viz - pt_viz.min()) / (pt_viz.max() - pt_viz.min() + 1e-8)
        onnx_viz = (onnx_viz - onnx_viz.min()) / (onnx_viz.max() - onnx_viz.min() + 1e-8)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(pt_viz, cmap='viridis')
        axes[0].set_title('PyTorch Output')
        axes[0].axis('off')
        
        axes[1].imshow(onnx_viz, cmap='viridis')
        axes[1].set_title('ONNX Output')
        axes[1].axis('off')
        
        diff = np.abs(pt_viz - onnx_viz)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(f'Difference (max: {diff.max():.4f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'output_{output_idx}_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Output {output_idx} visualization saved")

def main():
    # Configuration
    pt_path = "../chpt/best.pt"
    onnx_path = "../chpt/best.onnx"
    device = 'cpu'  # Change to 'cuda' if GPU available
    
    # Check if files exist
    if not os.path.exists(pt_path):
        print(f"Error: PyTorch model not found at {pt_path}")
        return 1
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        return 1
    
    # Create comparator
    comparator = ModelComparator(pt_path, onnx_path, device)
    
    # Generate test image
    test_image = comparator.generate_test_image()
    cv2.imwrite("test_image.jpg", test_image)
    print("Generated test image: test_image.jpg")
    
    # Preprocess image
    original_image, preprocessed_image = comparator.preprocess_image("test_image.jpg")
    
    # Run inference
    pt_results = comparator.run_pytorch_inference(preprocessed_image)
    onnx_results = comparator.run_onnx_inference(preprocessed_image)
    
    # Compare outputs
    comparison_results = comparator.compare_outputs(pt_results, onnx_results)
    
    # Analyze intermediate outputs
    intermediate_analysis = comparator.analyze_intermediate_outputs()
    
    # Save results
    comparator.save_comparison_visualization(pt_results, onnx_results, comparison_results)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"PyTorch time: {pt_results['time']:.3f}s")
    print(f"ONNX time: {onnx_results['time']:.3f}s")
    print(f"Speedup: {pt_results['time'] / onnx_results['time']:.2f}x")
    print(f"Outputs match: {comparison_results['outputs_match']}")
    print(f"Max difference: {comparison_results['max_diff']:.6f}")
    
    if comparison_results['outputs_match']:
        print("✓ Models are consistent!")
    else:
        print("✗ Models have significant differences!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
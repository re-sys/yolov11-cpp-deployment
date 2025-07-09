#!/usr/bin/env python3
"""
Analyze model intermediate layers and outputs in detail
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
import json

class ModelAnalyzer:
    def __init__(self, pt_path: str, onnx_path: str, device: str = 'cpu'):
        """
        Initialize model analyzer
        
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
        self.layer_names = []
        
    def load_pytorch_model(self) -> nn.Module:
        """Load PyTorch model and register hooks for all layers"""
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
        
        # Register hooks for all layers
        self.register_comprehensive_hooks(model)
        
        print("PyTorch model loaded successfully")
        return model
    
    def register_comprehensive_hooks(self, model: nn.Module):
        """Register hooks for comprehensive layer analysis"""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.pt_intermediates[name] = {
                        'output': output.detach().cpu().numpy(),
                        'shape': output.shape,
                        'dtype': str(output.dtype),
                        'device': str(output.device)
                    }
                elif isinstance(output, (list, tuple)):
                    outputs = []
                    for i, o in enumerate(output):
                        if isinstance(o, torch.Tensor):
                            outputs.append({
                                'output': o.detach().cpu().numpy(),
                                'shape': o.shape,
                                'dtype': str(o.dtype),
                                'device': str(o.device)
                            })
                        else:
                            outputs.append({'type': type(o).__name__, 'value': str(o)})
                    self.pt_intermediates[name] = outputs
            return hook
        
        # Register hooks for all named modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                module.register_forward_hook(hook_fn(name))
                self.layer_names.append(name)
        
        print(f"Registered hooks for {len(self.layer_names)} layers")
    
    def load_onnx_model(self) -> ort.InferenceSession:
        """Load ONNX model and analyze its structure"""
        print(f"Loading ONNX model from: {self.onnx_path}")
        
        # Create ONNX session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(self.onnx_path, providers=providers)
        
        # Analyze ONNX model structure
        self.analyze_onnx_structure(session)
        
        print("ONNX model loaded successfully")
        return session
    
    def analyze_onnx_structure(self, session: ort.InferenceSession):
        """Analyze ONNX model structure"""
        print("\n=== ONNX Model Structure ===")
        
        # Input information
        inputs = session.get_inputs()
        print(f"Number of inputs: {len(inputs)}")
        for i, input_info in enumerate(inputs):
            print(f"Input {i}: {input_info.name}")
            print(f"  Shape: {input_info.shape}")
            print(f"  Type: {input_info.type}")
        
        # Output information
        outputs = session.get_outputs()
        print(f"\nNumber of outputs: {len(outputs)}")
        for i, output_info in enumerate(outputs):
            print(f"Output {i}: {output_info.name}")
            print(f"  Shape: {output_info.shape}")
            print(f"  Type: {output_info.type}")
        
        # Get model metadata
        try:
            model_meta = session.get_modelmeta()
            print(f"\nModel metadata:")
            print(f"  Description: {model_meta.description}")
            print(f"  Version: {model_meta.version}")
            print(f"  Producer: {model_meta.producer_name}")
        except:
            print("Could not retrieve model metadata")
    
    def generate_test_image(self, size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Generate a test image with various features"""
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Add different shapes and patterns
        cv2.circle(image, (size[0]//2, size[1]//2), 100, (255, 0, 0), -1)
        cv2.rectangle(image, (100, 100), (300, 200), (0, 255, 0), -1)
        cv2.ellipse(image, (500, 400), (80, 60), 45, 0, 360, (0, 0, 255), -1)
        
        # Add some lines
        cv2.line(image, (50, 50), (200, 150), (255, 255, 0), 5)
        cv2.line(image, (400, 100), (600, 300), (255, 0, 255), 3)
        
        # Add text
        cv2.putText(image, "Test", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        return image
    
    def preprocess_image(self, image_path: str, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image for model input"""
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
    
    def run_pytorch_analysis(self, input_tensor: np.ndarray) -> Dict[str, Any]:
        """Run PyTorch model and capture all intermediate outputs"""
        print("\n=== Running PyTorch Analysis ===")
        
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
        print(f"Captured {len(self.pt_intermediates)} intermediate layers")
        
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
            'time': pt_time,
            'layer_count': len(self.pt_intermediates)
        }
    
    def run_onnx_analysis(self, input_tensor: np.ndarray) -> Dict[str, Any]:
        """Run ONNX model analysis"""
        print("\n=== Running ONNX Analysis ===")
        
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
    
    def analyze_layer_statistics(self) -> Dict[str, Any]:
        """Analyze statistics of intermediate layers"""
        print("\n=== Layer Statistics Analysis ===")
        
        layer_stats = {}
        
        for name, data in self.pt_intermediates.items():
            if isinstance(data, dict) and 'output' in data:
                output = data['output']
                
                stats = {
                    'shape': output.shape,
                    'dtype': str(output.dtype),
                    'min': float(np.min(output)),
                    'max': float(np.max(output)),
                    'mean': float(np.mean(output)),
                    'std': float(np.std(output)),
                    'zeros': int(np.sum(output == 0)),
                    'total_elements': int(output.size),
                    'sparsity': float(np.sum(output == 0) / output.size)
                }
                
                layer_stats[name] = stats
                
                print(f"{name}:")
                print(f"  Shape: {stats['shape']}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                print(f"  Sparsity: {stats['sparsity']:.2%}")
        
        return layer_stats
    
    def find_key_layers(self, layer_stats: Dict[str, Any]) -> Dict[str, List[str]]:
        """Find key layers based on various criteria"""
        print("\n=== Key Layer Analysis ===")
        
        key_layers = {
            'high_activation': [],
            'low_activation': [],
            'high_sparsity': [],
            'large_output': [],
            'small_output': []
        }
        
        for name, stats in layer_stats.items():
            # High activation layers
            if stats['max'] > 10.0:
                key_layers['high_activation'].append(name)
            
            # Low activation layers
            if stats['max'] < 0.1:
                key_layers['low_activation'].append(name)
            
            # High sparsity layers
            if stats['sparsity'] > 0.8:
                key_layers['high_sparsity'].append(name)
            
            # Large output layers
            if stats['total_elements'] > 1000000:
                key_layers['large_output'].append(name)
            
            # Small output layers
            if stats['total_elements'] < 1000:
                key_layers['small_output'].append(name)
        
        # Print key layers
        for category, layers in key_layers.items():
            if layers:
                print(f"{category.replace('_', ' ').title()}:")
                for layer in layers[:5]:  # Show first 5
                    print(f"  {layer}")
                if len(layers) > 5:
                    print(f"  ... and {len(layers) - 5} more")
        
        return key_layers
    
    def visualize_layer_outputs(self, layer_stats: Dict[str, Any], output_dir: str = "layer_analysis"):
        """Visualize layer outputs"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Saving Layer Visualizations to {output_dir} ===")
        
        # Select layers to visualize (focus on key layers)
        layers_to_viz = []
        for name, stats in layer_stats.items():
            # Select layers with reasonable shapes for visualization
            if len(stats['shape']) == 4 and stats['shape'][1] <= 64:  # Feature maps
                layers_to_viz.append(name)
            elif len(stats['shape']) == 2 and stats['shape'][1] <= 100:  # Linear layers
                layers_to_viz.append(name)
        
        # Limit number of visualizations
        layers_to_viz = layers_to_viz[:20]  # Max 20 layers
        
        for i, layer_name in enumerate(layers_to_viz):
            try:
                data = self.pt_intermediates[layer_name]
                if isinstance(data, dict) and 'output' in data:
                    output = data['output']
                    
                    # Create visualization
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    fig.suptitle(f'Layer: {layer_name}', fontsize=14)
                    
                    # Output distribution
                    axes[0, 0].hist(output.flatten(), bins=50, alpha=0.7)
                    axes[0, 0].set_title('Output Distribution')
                    axes[0, 0].set_xlabel('Value')
                    axes[0, 0].set_ylabel('Frequency')
                    
                    # Output statistics
                    stats_text = f"Shape: {output.shape}\n"
                    stats_text += f"Min: {np.min(output):.4f}\n"
                    stats_text += f"Max: {np.max(output):.4f}\n"
                    stats_text += f"Mean: {np.mean(output):.4f}\n"
                    stats_text += f"Std: {np.std(output):.4f}"
                    
                    axes[0, 1].text(0.1, 0.5, stats_text, transform=axes[0, 1].transAxes, 
                                   fontsize=10, verticalalignment='center')
                    axes[0, 1].set_title('Statistics')
                    axes[0, 1].axis('off')
                    
                    # Feature map visualization (if applicable)
                    if len(output.shape) == 4:
                        # Show first few channels
                        n_channels = min(8, output.shape[1])
                        for j in range(n_channels):
                            row = j // 4
                            col = j % 4
                            if row < 2 and col < 2:
                                feature_map = output[0, j]
                                im = axes[row, col].imshow(feature_map, cmap='viridis')
                                axes[row, col].set_title(f'Channel {j}')
                                axes[row, col].axis('off')
                                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'layer_{i:02d}_{layer_name.replace("/", "_")}.png'), 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                    
            except Exception as e:
                print(f"Warning: Could not visualize layer {layer_name}: {e}")
        
        print(f"Saved {len(layers_to_viz)} layer visualizations")
    
    def save_analysis_report(self, pt_results: Dict, onnx_results: Dict, 
                           layer_stats: Dict, key_layers: Dict, output_dir: str = "layer_analysis"):
        """Save comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "analysis_report.json")
        
        report = {
            'model_info': {
                'pt_path': self.pt_path,
                'onnx_path': self.onnx_path,
                'device': self.device
            },
            'performance': {
                'pt_inference_time': pt_results['time'],
                'onnx_inference_time': onnx_results['time'],
                'speedup': pt_results['time'] / onnx_results['time']
            },
            'layer_analysis': {
                'total_layers': pt_results['layer_count'],
                'layer_statistics': layer_stats,
                'key_layers': key_layers
            },
            'output_shapes': {
                'pt_outputs': [str(o.shape) if hasattr(o, 'shape') else str(type(o)) for o in (pt_results['outputs'] if isinstance(pt_results['outputs'], list) else [pt_results['outputs']])],
                'onnx_outputs': [str(o.shape) for o in onnx_results['outputs']]
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved to: {report_path}")
        
        # Also save a human-readable summary
        summary_path = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Model Layer Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"PyTorch model: {self.pt_path}\n")
            f.write(f"ONNX model: {self.onnx_path}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("Performance:\n")
            f.write(f"  PyTorch inference time: {pt_results['time']:.3f}s\n")
            f.write(f"  ONNX inference time: {onnx_results['time']:.3f}s\n")
            f.write(f"  Speedup: {pt_results['time'] / onnx_results['time']:.2f}x\n\n")
            
            f.write(f"Layer Analysis:\n")
            f.write(f"  Total layers analyzed: {pt_results['layer_count']}\n")
            f.write(f"  Layers with high activation: {len(key_layers['high_activation'])}\n")
            f.write(f"  Layers with high sparsity: {len(key_layers['high_sparsity'])}\n")
            f.write(f"  Large output layers: {len(key_layers['large_output'])}\n\n")
            
            f.write("Output Shapes:\n")
            for i, shape in enumerate(report['output_shapes']['pt_outputs']):
                f.write(f"  PyTorch output {i}: {shape}\n")
            for i, shape in enumerate(report['output_shapes']['onnx_outputs']):
                f.write(f"  ONNX output {i}: {shape}\n")
        
        print(f"Analysis summary saved to: {summary_path}")

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
    
    # Create analyzer
    analyzer = ModelAnalyzer(pt_path, onnx_path, device)
    
    # Generate test image
    test_image = analyzer.generate_test_image()
    cv2.imwrite("test_image.jpg", test_image)
    print("Generated test image: test_image.jpg")
    
    # Preprocess image
    original_image, preprocessed_image = analyzer.preprocess_image("test_image.jpg")
    
    # Run analysis
    pt_results = analyzer.run_pytorch_analysis(preprocessed_image)
    onnx_results = analyzer.run_onnx_analysis(preprocessed_image)
    
    # Analyze layer statistics
    layer_stats = analyzer.analyze_layer_statistics()
    
    # Find key layers
    key_layers = analyzer.find_key_layers(layer_stats)
    
    # Visualize layer outputs
    analyzer.visualize_layer_outputs(layer_stats)
    
    # Save analysis report
    analyzer.save_analysis_report(pt_results, onnx_results, layer_stats, key_layers)
    
    # Print summary
    print("\n=== Analysis Summary ===")
    print(f"Total layers analyzed: {pt_results['layer_count']}")
    print(f"PyTorch time: {pt_results['time']:.3f}s")
    print(f"ONNX time: {onnx_results['time']:.3f}s")
    print(f"Speedup: {pt_results['time'] / onnx_results['time']:.2f}x")
    print(f"Key layers found: {sum(len(layers) for layers in key_layers.values())}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
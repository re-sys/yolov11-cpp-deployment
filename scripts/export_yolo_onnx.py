#!/usr/bin/env python3
"""
高精度导出YOLO分割模型为ONNX，并可选测试ONNX与PT输出差异。
"""
import os
import sys
import torch
import numpy as np

# 优先尝试ultralytics官方API
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

def export_with_ultralytics(pt_path, onnx_path, img_size=(640, 640), opset=12):
    print(f"[ultralytics] Exporting {pt_path} to {onnx_path} ...")
    model = YOLO(pt_path)
    model.export(format='onnx', dynamic=False, opset=opset, simplify=True, imgsz=img_size)
    print(f"[ultralytics] Exported to {onnx_path}")

def export_with_torch(pt_path, onnx_path, img_size=(640, 640), opset=12):
    print(f"[torch] Exporting {pt_path} to {onnx_path} ...")
    model = torch.load(pt_path, map_location='cpu')
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
        elif 'ema' in model:
            model = model['ema']
        else:
            model = list(model.values())[0]
    model.eval()
    model.float()  # 强制float32
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output0', 'output1'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'output0': {0: 'batch_size'},
            'output1': {0: 'batch_size'}
        }
    )
    print(f"[torch] Exported to {onnx_path}")

def test_diff(pt_path, onnx_path, img_size=(640, 640)):
    import onnxruntime as ort
    import cv2
    # 加载模型
    model = torch.load(pt_path, map_location='cpu')
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
        elif 'ema' in model:
            model = model['ema']
        else:
            model = list(model.values())[0]
    model.eval()
    model.float()
    # 随机图片
    img = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[None]
    torch_input = torch.from_numpy(img_input).float()
    # pt输出
    with torch.no_grad():
        pt_out = model(torch_input)
    if isinstance(pt_out, (list, tuple)):
        pt_out = [o.cpu().numpy() if hasattr(o, 'cpu') else o for o in pt_out]
    else:
        pt_out = [pt_out.cpu().numpy()]
    # onnx输出
    ort_sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = ort_sess.run(None, {'images': img_input.astype(np.float32)})
    # 对比
    print("\n=== PT vs ONNX ===")
    for i, (a, b) in enumerate(zip(pt_out, ort_out)):
        print(f"Output {i}: shape {a.shape} vs {b.shape}")
        diff = np.abs(a - b)
        print(f"  max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
        if diff.max() > 1e-4:
            print("  ⚠️  WARNING: Large diff detected!")
        else:
            print("  ✓ Outputs are very close.")

def main():
    pt_path = '../chpt/best.pt'
    onnx_path = '../chpt/best.onnx'
    img_size = (640, 640)
    opset = 12
    # 导出
    if ULTRALYTICS_AVAILABLE:
        export_with_ultralytics(pt_path, onnx_path, img_size, opset)
    else:
        export_with_torch(pt_path, onnx_path, img_size, opset)
    # 测试差异
    test_diff(pt_path, onnx_path, img_size)

if __name__ == '__main__':
    main() 
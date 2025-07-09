#!/usr/bin/env python3
"""
重新导出YOLOv8 PT模型为ONNX格式，使用固定输入尺寸
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def re_export_onnx(pt_path, onnx_path, imgsz=640):
    """
    重新导出YOLOv8 PT模型为ONNX格式，使用固定输入尺寸
    """
    print(f"正在重新导出 {pt_path} 为 ONNX 格式...")
    print(f"输入尺寸: {imgsz}x{imgsz}")
    print(f"输出文件: {onnx_path}")
    
    try:
        # 加载模型
        model = YOLO(pt_path)
        
        # 检查模型类型
        model_type = model.task
        print(f"模型类型: {model_type}")
        
        # 导出为ONNX - 使用固定尺寸，不使用动态形状
        success = model.export(format='onnx', 
                              imgsz=imgsz, 
                              dynamic=False,  # 固定尺寸
                              simplify=True,
                              opset=12)
        
        if success:
            print(f"✅ ONNX模型重新导出成功: {onnx_path}")
            
            # 验证导出的模型
            print("正在验证导出的模型...")
            try:
                test_model = YOLO(onnx_path, task=model_type)
                print("✅ 模型验证成功")
                
                # 检查模型输入信息
                if hasattr(test_model, 'model') and hasattr(test_model.model, 'input'):
                    input_shape = test_model.model.input[0].shape
                    print(f"模型输入形状: {input_shape}")
                    
                    # 检查是否为固定尺寸
                    if -1 in input_shape:
                        print("⚠️  警告: 模型仍然包含动态尺寸")
                    else:
                        print("✅ 模型使用固定输入尺寸")
                        
            except Exception as e:
                print(f"⚠️  模型验证时出现警告: {e}")
            
            return True
        else:
            print("❌ ONNX模型重新导出失败")
            return False
            
    except Exception as e:
        print(f"❌ 重新导出过程中出错: {e}")
        return False

def main():
    # 默认参数
    pt_path = 'chpt/best.pt'
    onnx_path = 'best_fixed.onnx'
    imgsz = 640
    
    # 检查PT文件是否存在
    if not os.path.exists(pt_path):
        print(f"❌ PT模型文件不存在: {pt_path}")
        print("请确保 chpt/best.pt 文件存在")
        return
    
    # 检查输出目录
    output_dir = Path(onnx_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 重新导出ONNX
    if re_export_onnx(pt_path, onnx_path, imgsz):
        print(f"\n🎉 重新导出完成!")
        print(f"新的ONNX文件: {onnx_path}")
        print(f"输入尺寸: {imgsz}x{imgsz}")
        print("\n现在可以使用这个新的ONNX文件进行C++推理了")
    else:
        print("\n❌ 重新导出失败")

if __name__ == "__main__":
    main() 
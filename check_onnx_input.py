#!/usr/bin/env python3
"""
检查ONNX模型的输入形状
"""

import onnx
import numpy as np

def check_onnx_input(onnx_path):
    """
    检查ONNX模型的输入形状
    """
    print(f"检查ONNX模型: {onnx_path}")
    
    try:
        # 加载ONNX模型
        model = onnx.load(onnx_path)
        
        # 检查输入
        print("\n模型输入信息:")
        for i, input_info in enumerate(model.graph.input):
            print(f"输入 {i}: {input_info.name}")
            
            # 获取输入形状
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)  # 动态维度
                else:
                    shape.append(dim.dim_value)  # 固定维度
            
            print(f"  形状: {shape}")
            
            # 检查是否为固定尺寸
            if -1 in shape or '?' in shape or any(isinstance(x, str) for x in shape):
                print("  ⚠️  包含动态尺寸")
            else:
                print("  ✅ 使用固定尺寸")
        
        # 检查输出
        print("\n模型输出信息:")
        for i, output_info in enumerate(model.graph.output):
            print(f"输出 {i}: {output_info.name}")
            
            # 获取输出形状
            shape = []
            for dim in output_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            
            print(f"  形状: {shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查模型时出错: {e}")
        return False

def main():
    # 检查两个ONNX文件
    files_to_check = ['best.onnx', 'best_fixed.onnx']
    
    for onnx_file in files_to_check:
        try:
            check_onnx_input(onnx_file)
            print("\n" + "="*50 + "\n")
        except FileNotFoundError:
            print(f"文件不存在: {onnx_file}")
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 
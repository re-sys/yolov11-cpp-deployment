# YOLOv11 C++ 高性能部署

本项目是YOLOv11的C++高性能实现，基于ONNX Runtime进行推理，支持CPU和GPU加速，实现了相比Python API 8-11倍的性能提升。

## 🚀 主要特性

- **高性能推理**: GPU加速下达到88-121 FPS，相比CPU提升8-11倍
- **低延迟**: GPU推理延迟仅4-5ms，相比CPU的80ms提升16倍
- **灵活配置**: 支持命令行参数配置模型路径、置信度阈值等
- **多输入支持**: 支持图片、视频文件和摄像头实时推理
- **可视化输出**: 实时显示检测结果和性能指标
- **内存优化**: 高效的内存管理和GPU资源利用
- **分割支持**: 支持YOLO分割模型，提供像素级分割结果

## 🎥 演示视频

### YOLOv11 分割推理演示

https://github.com/your-username/yolocpp/assets/your-user-id/resource/test_yolov11_seg_gpu.mp4

*使用YOLOv11分割模型进行实时视频分割推理，GPU加速下达到稳定的推理性能*

### YOLOv8 分割推理演示

![YOLOv8 分割推理](resource/test_gpu_seg.gif)
*YOLOv8分割模型推理效果（前8秒，GIF动态展示）*

## 📊 性能对比

| 平台 | FPS | 推理时间 | 总时间(363帧) | 性能提升 |
|------|-----|----------|---------------|----------|
| CPU  | 11  | 80-83ms  | 33秒          | 基准     |
| GPU  | 88-121 | 4-5ms | 6秒        | 8-11倍   |

## 📁 项目结构

```
yolocpp/
├── CMakeLists.txt          # CMake构建配置
├── classes.txt             # 类别名称文件
├── src/
│   ├── main.cpp           # 主程序入口
│   └── ia/
│       ├── YOLO11.hpp     # YOLO11类头文件
│       ├── YOLO11.cpp     # YOLO11类实现
│       └── tools/
│           ├── Config.hpp # 配置常量定义
│           └── Config.cpp # 配置工具
├── chpt/
│   └── best.pt           # 训练好的PyTorch模型
├── build/                # 编译输出目录
└── models/               # ONNX模型存放目录
```

## 🛠️ 依赖安装

### 1. 基础依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake pkg-config
sudo apt install libopencv-dev
```

### 2. ONNX Runtime 安装

#### CPU版本
```bash
# 下载ONNX Runtime CPU版本
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
tar -xzf onnxruntime-linux-x64-1.17.1.tgz
sudo mv onnxruntime-linux-x64-1.17.1 /opt/onnxruntime-cpu

# 设置环境变量
echo 'export ONNXRUNTIME_ROOT_PATH=/opt/onnxruntime-cpu' >> ~/.bashrc
source ~/.bashrc
```

#### GPU版本（推荐）
```bash
# 下载ONNX Runtime GPU版本
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.17.1.tgz
sudo mv onnxruntime-linux-x64-gpu-1.17.1 /opt/onnxruntime-gpu

# 设置环境变量（GPU版本）
echo 'export ONNXRUNTIME_ROOT_PATH=/opt/onnxruntime-gpu' >> ~/.bashrc
source ~/.bashrc
```

### 3. CUDA和cuDNN安装（GPU加速必需）

#### 检查CUDA版本
```bash
nvcc --version
nvidia-smi
```

#### 安装cuDNN
```bash
# 下载cuDNN 8.9.7 for CUDA 12.x
# 从NVIDIA官网下载对应版本的cuDNN
# https://developer.nvidia.com/cudnn

# 解压并安装
tar -xzf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h /usr/local/cuda/include/
sudo cp -P cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 4. 验证安装

```bash
# 验证OpenCV
pkg-config --modversion opencv4

# 验证CUDA
nvcc --version

# 验证cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# 验证ONNX Runtime
ls $ONNXRUNTIME_ROOT_PATH/lib/
```

## 🔄 从.pt文件开始的完整流程

### 1. 模型转换（.pt → .onnx）

```bash
# 方法1: 使用YOLOv11官方脚本
pip install ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('chpt/best.pt')
model.export(format='onnx', imgsz=640, opset=11)
"

# 方法2: 使用命令行
yolo export model=chpt/best.pt format=onnx imgsz=640 opset=11

# 生成的ONNX文件会保存为 best.onnx
```

### 2. 分析模型结构

使用提供的模型分析工具：

```bash
cd yolocpp
g++ -o model_info model_info.cpp -lonnxruntime -I$ONNXRUNTIME_ROOT_PATH/include -L$ONNXRUNTIME_ROOT_PATH/lib
./model_info best.onnx
```

输出示例：
```
Model Analysis:
Input: images [1, 3, 640, 640]
Output: output0 [1, 5, 8400]  # [batch, bbox_attrs, anchors]
Class count: 1 (single object detection)
```

### 3. 修改类别文件

根据您的模型修改 `classes.txt`:

```bash
# 单类别模型
echo "object" > classes.txt

# 多类别模型（例如COCO）
echo -e "person\ncar\ndog\ncat" > classes.txt
```

### 4. 适配源码（如果需要）

如果您的模型输出格式不同，需要修改 `src/ia/YOLO11.cpp`:

#### 修改类别数量
```cpp
// 在YOLO11.cpp中找到并修改
const int num_classes = 1;  // 改为您的类别数量
```

#### 修改输出解析（根据模型输出格式）
```cpp
// 如果输出格式为 [1, 84, 8400] (COCO格式)
// 修改postProcess函数中的解析逻辑
float confidence = output_data[4 * num_anchors + i];  // 调整索引
for (int j = 0; j < num_classes; j++) {
    float class_score = output_data[(5 + j) * num_anchors + i];
    // 处理多类别逻辑
}
```

## 📋 ONNX导出参数变化后的源码修改指南

根据ultralytics最新官方文档，ONNX导出时可以使用多种参数来优化模型。当这些参数改变时，需要相应修改C++源码以确保兼容性。

### 🔧 导出参数详解

#### 1. 图像尺寸参数 (`imgsz`)

**导出时的设置：**
```python
# 正方形输入
model.export(format='onnx', imgsz=640)          # 640x640
model.export(format='onnx', imgsz=832)          # 832x832

# 矩形输入  
model.export(format='onnx', imgsz=(480, 640))   # 高度480, 宽度640
model.export(format='onnx', imgsz=[384, 672])   # 高度384, 宽度672
```

**需要修改的源码位置：**

1. **修改 `src/ia/tools/Config.hpp`:**
```cpp
// 根据你的导出设置修改这些值
#define DEFAULT_INPUT_WIDTH 640   // 改为你的宽度
#define DEFAULT_INPUT_HEIGHT 640  // 改为你的高度

// 例如，如果导出时使用 imgsz=(480, 640)
#define DEFAULT_INPUT_WIDTH 640
#define DEFAULT_INPUT_HEIGHT 480
```

2. **修改 `src/ia/YOLO11.cpp` 中的锚点数量计算:**
```cpp
// 在postprocess函数中，锚点数量需要根据输入尺寸调整
// 标准公式: num_anchors = (width/8)*(height/8) + (width/16)*(height/16) + (width/32)*(height/32)

// 对于640x640: 8400 = 80*80 + 40*40 + 20*20
// 对于832x832: 14756 = 104*104 + 52*52 + 26*26  
// 对于480x640: 6300 = 60*80 + 30*40 + 15*20

int num_detections = 8400;  // 根据你的输入尺寸修改这个值

// 计算公式示例：
// int num_detections = (input_width_/8)*(input_height_/8) + 
//                      (input_width_/16)*(input_height_/16) + 
//                      (input_width_/32)*(input_height_/32);
```

#### 2. 批处理大小参数 (`batch`)

**导出时的设置：**
```python
model.export(format='onnx', batch=1)   # 默认值
model.export(format='onnx', batch=4)   # 批处理4张图片
```

**需要修改的源码位置：**

如果导出时设置了 `batch > 1`，需要修改推理代码：

```cpp
// 在YOLO11.cpp的detect函数中
std::vector<int64_t> input_shape = {batch_size, 3, input_height_, input_width_};

// 输出也需要相应调整
// 输出形状会变为 [batch_size, elements_per_detection, num_anchors]
```

#### 3. 动态输入参数 (`dynamic`)

**导出时的设置：**
```python
model.export(format='onnx', dynamic=True)   # 支持动态输入尺寸
```

**需要修改的源码位置：**

如果启用了动态输入，推理时可以处理不同尺寸的图片，但需要修改预处理逻辑：

```cpp
// 在YOLO11.cpp中添加动态尺寸支持
cv::Mat YOLO11::preprocess(const cv::Mat& image, int target_width, int target_height) {
    cv::Mat resized, normalized;
    
    // 动态调整目标尺寸
    cv::resize(image, resized, cv::Size(target_width, target_height));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    return normalized;
}
```

#### 4. 半精度参数 (`half`)

**导出时的设置：**
```python
model.export(format='onnx', half=True)   # 启用FP16
```

**需要修改的源码位置：**

启用FP16后，模型权重精度会降低但速度更快。通常不需要修改C++代码，但如果遇到精度问题可以调整：

```cpp
// 在YOLO11.cpp的initialize函数中
if (use_half_precision) {
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 可能需要调整置信度阈值
    // 因为FP16可能会略微影响精度
}
```

#### 5. 简化参数 (`simplify`)

**导出时的设置：**
```python
model.export(format='onnx', simplify=True)   # 默认值，简化模型图
model.export(format='onnx', simplify=False)  # 保持原始模型结构
```

通常不需要修改C++代码，但如果 `simplify=False`
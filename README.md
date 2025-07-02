# YOLOv11 C++ 高性能部署

本项目是YOLOv11的C++高性能实现，基于ONNX Runtime进行推理，支持CPU和GPU加速，实现了相比Python API 8-11倍的性能提升。

## 🚀 主要特性

- **高性能推理**: GPU加速下达到88-121 FPS，相比CPU提升8-11倍
- **低延迟**: GPU推理延迟仅4-5ms，相比CPU的80ms提升16倍
- **灵活配置**: 支持命令行参数配置模型路径、置信度阈值等
- **多输入支持**: 支持图片、视频文件和摄像头实时推理
- **可视化输出**: 实时显示检测结果和性能指标
- **内存优化**: 高效的内存管理和GPU资源利用

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

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake pkg-config opencv-devel
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

## 🏗️ 编译和运行

### 1. 编译项目

```bash
cd yolocpp
mkdir -p build
cd build

# 配置编译
cmake ..

# 编译
make -j$(nproc)
```

### 2. 运行推理

#### 基本使用
```bash
# 图片推理
./yolo_detector --model ../best.onnx --input image.jpg --output result.jpg

# 视频推理
./yolo_detector --model ../best.onnx --input video.mp4 --output result.mp4

# 摄像头推理
./yolo_detector --model ../best.onnx --camera 0
```

#### 高级参数
```bash
./yolo_detector \
    --model ../best.onnx \
    --input video.mp4 \
    --output result.mp4 \
    --gpu \                    # 启用GPU加速
    --conf-threshold 0.5 \     # 置信度阈值
    --iou-threshold 0.4 \      # NMS IoU阈值
    --debug \                  # 显示调试信息
    --timing                   # 显示性能指标
```

## ⚙️ GPU加速配置

### 1. 编译时启用GPU支持

在 `CMakeLists.txt` 中确保GPU支持已启用：

```cmake
# 查找CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    add_definitions(-DUSE_CUDA)
endif()
```

### 2. 运行时启用GPU

```bash
# 方法1: 命令行参数
./yolo_detector --model ../best.onnx --input video.mp4 --gpu

# 方法2: 环境变量
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
./yolo_detector --model ../best.onnx --input video.mp4
```

### 3. GPU性能优化

```cpp
// 在YOLO11.cpp中的构造函数添加GPU优化选项
if (use_gpu) {
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    sessionOptions.EnableMemPattern();
    sessionOptions.EnableCpuMemArena();
}
```

## 🔧 故障排除

### 常见问题

1. **编译错误**: `fatal error: onnxruntime/core/session/onnxruntime_cxx_api.h: No such file`
   ```bash
   # 检查ONNX Runtime路径
   echo $ONNXRUNTIME_ROOT_PATH
   ls $ONNXRUNTIME_ROOT_PATH/include/onnxruntime/core/session/
   ```

2. **运行时错误**: `libonnxruntime.so: cannot open shared object file`
   ```bash
   # 添加库路径
   export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT_PATH/lib:$LD_LIBRARY_PATH
   ```

3. **GPU推理失败**: `CUDA error` 或性能没有提升
   ```bash
   # 检查CUDA环境
   nvidia-smi
   # 检查cuDNN版本兼容性
   cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
   ```

4. **内存不足**: 大视频处理时内存溢出
   ```bash
   # 减少批处理大小或使用内存映射
   ulimit -v 8388608  # 限制虚拟内存
   ```

### 性能调优

1. **CPU优化**:
   ```bash
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```

2. **GPU优化**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   nvidia-smi -pm 1  # 启用持久化模式
   ```

## 📈 部署建议

### 生产环境部署

1. **Docker容器化**:
   ```dockerfile
   FROM nvidia/cuda:12.1-devel-ubuntu20.04
   # 安装依赖和复制程序
   COPY yolo_detector /app/
   COPY models/ /app/models/
   ```

2. **服务化部署**:
   ```bash
   # 创建systemd服务
   sudo nano /etc/systemd/system/yolo_detector.service
   sudo systemctl enable yolo_detector
   sudo systemctl start yolo_detector
   ```

3. **负载均衡**:
   - 使用多个GPU实例
   - 实现请求队列管理
   - 添加健康检查端点

### 模型优化

1. **量化**:
   ```python
   # 导出时启用INT8量化
   model.export(format='onnx', int8=True)
   ```

2. **TensorRT优化** (NVIDIA GPU):
   ```bash
   trtexec --onnx=best.onnx --saveEngine=best.trt --fp16
   ```

## 📞 技术支持

如果遇到问题，请检查：
1. 系统环境和依赖版本
2. 模型文件完整性
3. 硬件兼容性（GPU驱动、CUDA版本）
4. 运行日志和错误信息

---

## 🎯 快速开始示例

```bash
# 1. 克隆或准备项目文件
cd yolocpp

# 2. 安装依赖（参考上述安装指南）

# 3. 转换模型
python -c "from ultralytics import YOLO; YOLO('chpt/best.pt').export(format='onnx')"

# 4. 编译项目
mkdir build && cd build && cmake .. && make

# 5. 运行测试
./yolo_detector --model ../best.onnx --input ../test_image --gpu --debug
```

享受高性能的YOLOv11 C++推理体验！🚀
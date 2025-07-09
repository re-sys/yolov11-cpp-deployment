# YOLO Segmentation ONNX Implementation

这个项目实现了基于ONNX Runtime的YOLO分割模型推理，支持实例分割功能。

## 功能特性

- 基于ONNX Runtime的高效推理
- 支持实例分割（Instance Segmentation）
- 自动处理模型输入输出
- 可视化分割结果
- 支持CPU和GPU推理

## 文件结构

```
src/ia/
├── YOLOSegmentation.hpp    # 分割推理类头文件
├── YOLOSegmentation.cpp    # 分割推理类实现
└── tools/
    └── Config.hpp          # 配置常量

test_segmentation_onnx.cpp  # 测试程序
CMakeLists_segmentation.txt # CMake构建文件
export_segmentation_onnx.py # PyTorch到ONNX导出脚本
build_and_test.sh          # 构建和测试脚本
```

## 依赖要求

- OpenCV 4.x
- ONNX Runtime 1.x
- CMake 3.10+
- C++17 编译器

## 安装依赖

### Ubuntu/Debian
```bash
# 安装OpenCV
sudo apt update
sudo apt install libopencv-dev

# 安装ONNX Runtime
# 方法1: 从官方下载
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz
tar -xzf onnxruntime-linux-x64-1.15.1.tgz
sudo cp -r onnxruntime-linux-x64-1.15.1/include/* /usr/local/include/
sudo cp onnxruntime-linux-x64-1.15.1/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig

# 方法2: 使用包管理器（如果可用）
sudo apt install libonnxruntime-dev
```

### CentOS/RHEL
```bash
# 安装OpenCV
sudo yum install opencv-devel

# 安装ONNX Runtime（类似Ubuntu方法）
```

## 使用方法

### 1. 准备模型文件

将你的YOLO分割模型放在 `chpt/` 目录下：
- `chpt/best.pt` - PyTorch模型文件
- `chpt/best.onnx` - ONNX模型文件（如果已有）

### 2. 准备类别文件

确保 `classes.txt` 文件存在，包含类别名称：
```
object
```

### 3. 构建和测试

使用提供的脚本自动构建和测试：
```bash
./build_and_test.sh
```

或者手动构建：
```bash
# 如果需要从PyTorch导出ONNX
python3 export_segmentation_onnx.py

# 构建
mkdir -p build
cd build
cmake -f ../CMakeLists_segmentation.txt ..
make -j$(nproc)

# 运行测试
./test_segmentation_onnx
```

### 4. 在代码中使用

```cpp
#include "src/ia/YOLOSegmentation.hpp"

int main() {
    // 创建分割模型实例
    YOLOSegmentation seg_model;
    
    // 初始化模型
    if (!seg_model.initialize("chpt/best.onnx", "classes.txt", false)) {
        std::cerr << "Failed to initialize model" << std::endl;
        return -1;
    }
    
    // 加载图像
    cv::Mat image = cv::imread("test_image.jpg");
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }
    
    // 执行分割
    auto results = seg_model.detect(image, 0.25f, 0.45f);
    
    // 绘制结果
    cv::Mat result_image = image.clone();
    seg_model.draw_segmentation(result_image, results);
    
    // 保存结果
    cv::imwrite("result.jpg", result_image);
    
    return 0;
}
```

## API 参考

### YOLOSegmentation 类

#### 构造函数
```cpp
YOLOSegmentation();
```

#### 初始化
```cpp
bool initialize(const std::string& model_path, 
                const std::string& classes_path, 
                bool use_gpu = false);
```

#### 推理
```cpp
std::vector<SegmentationResult> detect(const cv::Mat& image, 
                                      float conf_threshold = 0.25f, 
                                      float iou_threshold = 0.45f);
```

#### 可视化
```cpp
void draw_segmentation(cv::Mat& image, 
                      const std::vector<SegmentationResult>& results);
```

### SegmentationResult 结构

```cpp
struct SegmentationResult {
    cv::Rect bbox;           // 边界框
    float confidence;        // 置信度
    int class_id;           // 类别ID
    std::string class_name; // 类别名称
    cv::Mat mask;           // 分割掩码
};
```

## 模型要求

- 输入: `[1, 3, 640, 640]` (RGB图像，归一化到[0,1])
- 输出0: 检测结果，包含边界框、置信度和掩码系数
- 输出1: 原型掩码，用于生成最终分割掩码

## 性能优化

1. **GPU加速**: 设置 `use_gpu = true` 启用CUDA加速
2. **批处理**: 支持批量推理以提高吞吐量
3. **内存优化**: 使用智能指针管理内存

## 故障排除

### 常见问题

1. **ONNX Runtime未找到**
   ```
   CMake Error: ONNX Runtime not found
   ```
   解决: 确保正确安装ONNX Runtime并设置环境变量

2. **模型加载失败**
   ```
   Failed to initialize YOLOSegmentation
   ```
   解决: 检查模型文件路径和格式

3. **内存不足**
   ```
   std::bad_alloc
   ```
   解决: 减少批处理大小或使用更小的输入尺寸

### 调试模式

编译时启用DEBUG标志以获取详细日志：
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -f ../CMakeLists_segmentation.txt ..
```

## 示例输出

测试程序会输出类似以下信息：
```
=== ONNX Model Information ===
--- Input Information ---
Number of inputs: 1
Input 0: images
  Shape: [1, 3, 640, 640]

--- Output Information ---
Number of outputs: 2
Output 0: output0
  Shape: [1, 37, 8400]
  Total elements: 310800
Output 1: output1
  Shape: [1, 32, 160, 160]
  Total elements: 819200

=== Testing Model Inference ===
Inference time: 45 ms

=== Testing YOLOSegmentation Class ===
Segmentation model initialized successfully!
Detection completed in 52 ms
Found 2 objects
Result saved as test_segmentation_result.jpg
```

## 许可证

本项目基于MIT许可证开源。 
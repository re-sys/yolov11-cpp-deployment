# YOLO Model Analysis Scripts

这个目录包含了用于分析PyTorch和ONNX模型差异的Python脚本。

## 文件说明

### 核心脚本

- `compare_pt_onnx.py` - 基础模型比较脚本
  - 比较PyTorch和ONNX模型的输出差异
  - 生成测试图像并运行推理
  - 输出详细的比较报告和可视化

- `analyze_model_layers.py` - 详细层分析脚本
  - 分析模型的所有中间层输出
  - 统计每层的激活值分布
  - 识别关键层（高激活、高稀疏性等）
  - 生成层可视化图表

- `run_model_analysis.sh` - 一键运行脚本
  - 自动执行所有分析步骤
  - 生成完整的分析报告

### 配置文件

- `requirements.txt` - Python依赖包列表
- `README.md` - 本说明文件

## 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 或者使用conda
conda install pytorch torchvision onnxruntime opencv numpy matplotlib pillow
```

## 使用方法

### 1. 一键分析（推荐）

```bash
# 从项目根目录运行
./scripts/run_model_analysis.sh
```

### 2. 单独运行脚本

```bash
# 进入scripts目录
cd scripts

# 基础比较
python3 compare_pt_onnx.py

# 详细层分析
python3 analyze_model_layers.py
```

### 3. 手动运行

```bash
# 确保模型文件存在
ls chpt/best.pt chpt/best.onnx

# 运行比较
python3 scripts/compare_pt_onnx.py

# 运行层分析
python3 scripts/analyze_model_layers.py
```

## 输出结果

### 基础比较结果 (`comparison_results/`)

- `comparison_report.txt` - 详细的比较报告
- `output_*_comparison.png` - 输出可视化对比图
- `test_image.jpg` - 生成的测试图像

### 详细层分析结果 (`layer_analysis/`)

- `analysis_report.json` - 完整的分析报告（JSON格式）
- `analysis_summary.txt` - 人类可读的分析摘要
- `layer_*_*.png` - 各层的可视化图表
- `test_image.jpg` - 生成的测试图像

## 分析内容

### 基础比较分析

1. **模型结构对比**
   - 输入输出形状
   - 模型元数据
   - 推理时间对比

2. **输出差异分析**
   - 最大差异值
   - 平均差异值
   - 相对差异百分比
   - 数值一致性检查

3. **性能对比**
   - PyTorch推理时间
   - ONNX推理时间
   - 加速比计算

### 详细层分析

1. **中间层统计**
   - 每层的输出形状
   - 激活值分布（最小值、最大值、均值、标准差）
   - 稀疏性分析
   - 零值比例

2. **关键层识别**
   - 高激活层（激活值 > 10）
   - 低激活层（激活值 < 0.1）
   - 高稀疏层（稀疏性 > 80%）
   - 大输出层（元素数 > 1M）
   - 小输出层（元素数 < 1K）

3. **层可视化**
   - 输出分布直方图
   - 特征图可视化（适用于卷积层）
   - 统计信息展示

## 配置选项

### 修改设备

在脚本中修改 `device` 参数：

```python
# CPU模式
device = 'cpu'

# GPU模式（需要CUDA支持）
device = 'cuda'
```

### 修改测试图像

可以修改 `generate_test_image()` 函数来生成不同的测试图像：

```python
def generate_test_image(self, size=(640, 640)):
    # 自定义测试图像生成逻辑
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # 添加你想要的形状和图案
    return image
```

### 修改容差

在比较脚本中修改 `tolerance` 参数：

```python
# 更严格的容差
comparison_results = comparator.compare_outputs(pt_results, onnx_results, tolerance=1e-6)

# 更宽松的容差
comparison_results = comparator.compare_outputs(pt_results, onnx_results, tolerance=1e-4)
```

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   Error: PyTorch model not found at ../chpt/best.pt
   ```
   解决：确保模型文件存在于正确路径

2. **依赖包缺失**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   解决：安装依赖包 `pip install -r requirements.txt`

3. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：使用CPU模式或减少批处理大小

4. **可视化失败**
   ```
   Warning: Could not visualize layer
   ```
   解决：检查matplotlib是否正确安装，或跳过可视化步骤

### 调试模式

启用详细输出：

```bash
# 设置环境变量
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0  # 指定GPU

# 运行分析
python3 -u scripts/analyze_model_layers.py
```

## 扩展功能

### 添加自定义分析

可以在 `ModelAnalyzer` 类中添加新的分析方法：

```python
def custom_analysis(self):
    """自定义分析逻辑"""
    # 你的分析代码
    pass
```

### 批量分析

可以修改脚本来分析多个模型：

```python
models = [
    ("model1.pt", "model1.onnx"),
    ("model2.pt", "model2.onnx"),
    # ...
]

for pt_path, onnx_path in models:
    analyzer = ModelAnalyzer(pt_path, onnx_path)
    # 运行分析
```

## 性能优化

1. **GPU加速**：使用CUDA设备可以显著提升PyTorch推理速度
2. **内存优化**：对于大模型，可以分批处理中间层输出
3. **并行处理**：可以并行运行多个分析任务

## 许可证

本项目基于MIT许可证开源。 
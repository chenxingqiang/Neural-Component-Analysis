# 快速入门指南

本指南将帮助您快速上手Neural Component Analysis项目，了解如何安装、配置和运行基本故障检测实验。

## 1. 安装项目

### 环境要求

- Python 3.7+
- PyTorch 1.7.0+
- NumPy, Matplotlib, scikit-learn, pandas等

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/username/neural-component-analysis.git
cd neural-component-analysis
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 开发模式安装（可选）：

```bash
pip install -e .
```

## 2. 目录结构说明

- `src/`: 核心源代码
- `scripts/`: 运行脚本
- `examples/`: 使用示例
- `data/`: 数据集目录
- `results/`: 结果输出目录
- `docs/`: 文档目录

## 3. 数据准备

项目支持SECOM半导体制造数据集和Tennessee Eastman (TE)数据集。您可以：

### 使用内置数据

在`data/`目录已包含处理好的数据集，可直接使用。

### 处理自定义数据

对于自定义数据，请参考以下格式要求：

```python
from src.utils.process_secom_data import preprocess_data

# 处理自定义数据
X_train, X_test, y_test = preprocess_data(
    train_data_path, 
    test_data_path,
    labels_path
)
```

## 4. 运行基础实验

### SECOM数据集故障检测

运行基本检测实验：

```bash
python scripts/run_secom_fault_detection.py
```

运行结果将保存在`results/`目录下：
- 模型文件：`results/models/`
- 图表：`results/plots/`
- 日志：`results/logs/`

### 比较不同检测方法

```bash
python scripts/secom_comparison_detection.py
```

### T²和SPE对比分析

```bash
python scripts/compare_t2_spe.py
```

## 5. 使用示例

### 基本使用流程

```python
from src.models import EnhancedTransformerAutoencoder
from src.detectors import create_fault_detector

# 1. 加载数据
from src.utils import load_secom_data, preprocess_secom_data
X_train, X_test, y_test = load_secom_data()
X_train_processed, X_test_processed = preprocess_secom_data(X_train, X_test)

# 2. 创建和训练检测器
detector = create_fault_detector('enhanced_transformer')
detector.fit(X_train_processed)

# 3. 检测故障
results = detector.detect(X_test_processed)
print(f"检测到的故障样本数: {results['fault_flags'].sum()}")

# 4. 可视化结果
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(results['metric_values'])
plt.axhline(y=detector.threshold, color='r', linestyle='--')
plt.title('故障检测结果')
plt.xlabel('样本')
plt.ylabel('统计量')
plt.show()
```

### 检测器类型

可用的检测器类型包括：

- `'enhanced_transformer'`: 增强型Transformer检测器
- `'t2_transformer'`: T²优化的Transformer检测器
- `'spe'`: SPE专用检测器
- `'two_stage'`: 两阶段检测器

## 6. 自定义模型

您可以通过继承基类来创建自定义检测器：

```python
from src.detectors.fault_detector_factory import FaultDetector

class MyCustomDetector(FaultDetector):
    def __init__(self, input_dim, **kwargs):
        # 初始化代码
        pass
        
    def train(self, train_data, **kwargs):
        # 训练逻辑
        pass
        
    def detect(self, test_data):
        # 检测逻辑
        pass
```

## 7. 故障排除

如果遇到问题，请检查：

1. Python和PyTorch版本是否兼容
2. 数据格式是否正确
3. GPU是否可用（如果使用）
4. `results/`目录是否存在并可写入

## 8. 下一步学习

- 阅读[算法详细介绍](algo_intro.md)了解模型原理
- 查看[方法比较](README_COMPARISON.md)了解各种方法的优缺点
- 参考`examples`目录中的示例代码
- 探索`scripts`目录中的分析脚本 
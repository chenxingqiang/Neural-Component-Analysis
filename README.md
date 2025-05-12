# Neural Component Analysis

基于神经网络和Transformer的工业过程故障检测方法实现。本项目实现了多种基于自编码器的故障检测算法，特别是针对半导体制造过程（SECOM数据集）的异常检测。

## 项目特点

- 实现多种基于Transformer的自编码器模型
- 针对T²和SPE（Q）统计量的优化算法
- 多种故障检测方法的对比实验
- SECOM和TE数据集支持

## 模型架构

项目包含以下主要模型：

1. **基础Transformer自编码器** - 适用于一般故障检测
2. **增强型Transformer自编码器** - 针对SPE（Q统计量）性能优化
3. **改进型T² Transformer自编码器** - 专注于T²统计量性能提升
4. **两阶段检测器** - 结合T²和SPE的优势

详细的模型架构说明见 [algo_intro.md](algo_intro.md)。

## 目录结构

主要文件说明：

- **enhanced_transformer_autoencoder.py** - 增强型Transformer自编码器实现
- **improved_transformer_t2.py** - 改进型T²专用Transformer自编码器
- **spe_fault_detector.py** - SPE故障检测器实现
- **transformer_enhanced_two_stage.py** - 两阶段Transformer检测器
- **run_secom_fault_detection.py** - SECOM数据集故障检测示例
- **compare_t2_spe.py** - T²和SPE性能对比脚本
- **process_secom_data.py** - SECOM数据预处理工具
- **integration_example.py** - 集成使用示例

## 数据集

项目支持两个主要数据集：

1. **SECOM** - 半导体制造过程数据集（位于`data/secom/`和`data/secom_original_data/`）
2. **TE** - Tennessee Eastman化工过程数据集（位于`data/TE/`）

## 使用方法

### 环境要求

```
Python 3.7+
PyTorch 1.7+
NumPy
Scikit-learn
Matplotlib
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

1. SECOM数据集故障检测：

```bash
python run_secom_fault_detection.py
```

2. 比较不同模型在SECOM数据上的性能：

```bash
python secom_comparison_detection.py
```

3. T²和SPE性能对比：

```bash
python compare_t2_spe.py
```

## 实验结果

![故障检测比较](comparison_fault_detection.png)

SECOM数据集上各方法检测性能：

- 增强型Transformer检测器：AUC 0.95
- 改进型T²检测器：AUC 0.92
- 两阶段检测器：AUC 0.97

更多结果见实验图像文件。

## 引用

如果您在研究中使用了此代码，请引用：

```
@misc{Neural-Component-Analysis,
  author = {Author},
  title = {Neural-Component-Analysis},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/username/Neural-Component-Analysis}
}
```

## 许可证

MIT 
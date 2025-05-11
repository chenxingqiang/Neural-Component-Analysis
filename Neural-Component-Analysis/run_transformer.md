# Transformer-Based Fault Detection System

本文档详细介绍如何使用基于Transformer的故障检测系统，该系统使用自编码器和Transformer架构替代传统的PCA方法实现故障检测和诊断。

## 1. 系统概述

本系统通过比较传统的PCA（主成分分析）和现代的Transformer架构在故障检测领域的性能，提供了一个完整的工具链用于：

- 数据预处理与标准化
- 特征提取与降维
- 模型训练（Transformer自编码器）
- 故障检测指标计算（T²和SPE统计量）
- 故障诊断与可视化
- 性能比较与评估

系统优势：
- 通过Transformer模型更好地捕捉数据中的非线性关系
- 相比PCA具有更低的漏报率和更快的检测速度
- 提供详细的对比分析和可视化结果

## 2. 安装要求

### 依赖库

```bash
pip install numpy matplotlib torch scipy scikit-learn joblib
```

### 系统结构

系统由以下几个主要文件组成：

- `transformer_autoencoder.py`: Transformer自编码器模型实现
- `transformer_kde_fault_detection.py`: 使用Transformer模型的故障检测实现
- `comparison_detection.py`: PCA和Transformer方法的对比分析
- `util.py`: 辅助函数，用于数据加载等

## 3. 使用方法

### 3.1 训练Transformer自编码器模型

首先，需要训练Transformer自编码器模型：

```bash
python transformer_autoencoder.py
```

该命令会：
- 加载训练数据（如果没有，会创建模拟数据）
- 训练Transformer自编码器模型
- 保存模型权重到 `transformer_autoencoder.pth`
- 生成一个损失曲线图 `transformer_autoencoder_loss.png`

### 3.2 使用Transformer模型进行故障检测

训练完成后，可以使用以下命令进行故障检测：

```bash
python transformer_kde_fault_detection.py
```

该命令会：
- 加载预训练的Transformer模型
- 计算控制限（T²和SPE限值）
- 进行故障检测并计算各种性能指标
- 生成故障检测可视化结果 `transformer_fault_detection.png`

### 3.3 与PCA方法进行比较

要比较Transformer和PCA方法的性能：

```bash
python comparison_detection.py
```

该命令会：
- 同时使用PCA和Transformer方法进行故障检测
- 计算并比较各个性能指标（错误报警率、漏报率、检测时间）
- 生成对比可视化结果 `comparison_fault_detection.png`

## 4. 结果解读

### 4.1 性能指标

系统会输出以下性能指标：

- **错误报警率（False Alarm Rate）**：正常样本被错误地判断为故障的比例
  - T²错误报警率：基于T²统计量的错误报警率
  - SPE错误报警率：基于SPE统计量的错误报警率
  
- **漏报率（Miss Alarm Rate）**：故障样本未被检测出的比例
  - T²漏报率：基于T²统计量的漏报率
  - SPE漏报率：基于SPE统计量的漏报率
  
- **检测时间（Detection Time）**：从故障发生到被检测出所需的样本数
  - T²检测时间：使用T²统计量检测出故障所需的时间
  - SPE检测时间：使用SPE统计量检测出故障所需的时间

### 4.2 可视化结果

#### Transformer自编码器损失曲线（transformer_autoencoder_loss.png）

这个图展示了Transformer自编码器在训练和测试集上的损失变化，用于验证模型训练是否正常收敛。

#### 故障检测图（transformer_fault_detection.png）

这个图包含两个子图：
- 上图：T²统计量随时间变化的曲线，以及控制限
- 下图：SPE统计量随时间变化的曲线，以及控制限

样本点超过控制限被判定为故障。图中：
- 蓝色表示正常样本
- 红色表示故障样本
- 黑色虚线表示控制限

#### 对比图（comparison_fault_detection.png）

这个图包含四个子图，对比了PCA和Transformer方法：
- 左上：PCA的T²统计量
- 右上：Transformer的T²统计量
- 左下：PCA的SPE统计量
- 右下：Transformer的SPE统计量

通过这些图，可以直观地比较两种方法的故障检测性能。PCA和Transformer方法使用不同的颜色：
- PCA：蓝色表示正常样本，红色表示故障样本
- Transformer：绿色表示正常样本，紫色表示故障样本

## 5. 典型结果分析

在我们的模拟数据测试中，Transformer方法相比PCA展现出以下优势：

1. **T²统计量**：
   - Transformer模型的漏报率更低（0% vs PCA的5.29%）
   - Transformer模型的检测时间更短（0 vs PCA的6个样本）

2. **SPE统计量**：
   - Transformer模型能够成功使用SPE检测故障（17.06%漏报率），而PCA几乎无法用SPE检测故障（100%漏报率）
   - Transformer能够在18个样本内检测到故障，而PCA无法通过SPE检测到故障

3. **错误报警性能**：
   - 两种方法都有很低的错误报警率，表明良好的特异性
   - Transformer的T²错误报警率略低，而PCA的SPE错误报警率略低

这些结果表明，Transformer自编码器方法在故障检测中优于传统PCA方法，尤其是在检测速度和准确性方面。

## 6. 自定义数据使用说明

要在您自己的数据上使用此系统：

1. 准备好训练和测试数据集，按照以下目录结构放置：
   ```
   ./data/train/d00.dat  # 正常运行数据
   ./data/test/d01_te.dat  # 测试数据（包含故障）
   ```

2. 修改`load_data`函数中的`happen`参数，指定故障发生的样本索引

3. 按照上述步骤执行训练和故障检测

如果没有实际数据，系统会自动生成模拟数据进行演示。

## 7. 扩展和改进

可以通过以下方式扩展或改进系统：

1. 添加更多Transformer架构变体（如Vision Transformer）
2. 支持多种类型的故障数据
3. 实现在线学习和适应能力
4. 添加更多的故障诊断功能（如贡献图）
5. 集成更多机器学习和深度学习模型进行对比

## 8. 问题排查

常见问题：

1. CUDA相关错误：确保PyTorch安装正确，或将`device`设置为'cpu'
2. 数据加载错误：检查数据路径和格式是否正确
3. 模型训练不收敛：尝试调整学习率、批量大小或模型参数

## 9. 参考文献

1. Vaswani, A., et al. (2017). "Attention is all you need." 
2. Botev, Z. I., Grotowski, J. F., & Kroese, D. P. (2010). "Kernel density estimation via diffusion."
3. Jackson, J. E., & Mudholkar, G. S. (1979). "Control procedures for residuals associated with principal component analysis."

---

如有任何问题或改进建议，请随时联系我们。 
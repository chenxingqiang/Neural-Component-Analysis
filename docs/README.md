# Neural Component Analysis Documentation

This directory contains documentation for the Neural Component Analysis project, which implements various Transformer-based methods for industrial process fault detection.

## Documentation Files

- [快速入门指南](getting_started.md) - 快速上手项目的指南，包含安装、配置和基本使用方法。
- [Algorithm Introduction](algo_intro.md) - Detailed introduction to the Transformer models used in this project.
- [Balanced Two-Stage Detection](balance-twostage.md) - Explanation of the balanced two-stage fault detection approach.
- [Comparative Analysis](README_COMPARISON.md) - Comparison of different fault detection methods implemented in this project.
- [SECOM Dataset Guide](README_SECOM.md) - Information about the SECOM semiconductor manufacturing dataset and how it's used in this project.

## Project Structure

The project is organized into the following modules:

### Models (`src/models/`)

- **EnhancedTransformerAutoencoder** - Specialized Transformer architecture optimized for SPE (Squared Prediction Error).
- **ImprovedTransformerAutoencoder** - Improved Transformer architecture with specific optimizations for T² statistic.
- **TransformerRefiner** - Two-stage Transformer architecture for enhancing detection results.

### Detectors (`src/detectors/`)

- **EnhancedTransformerDetector** - Detector implementation using the enhanced Transformer model.
- **SPEFaultDetector** - Dedicated SPE-based fault detection implementation.
- **FaultDetectorFactory** - Factory pattern for creating different fault detector types.

### Utilities (`src/utils/`)

- **Data Processing** - Functions for preprocessing SECOM and TE datasets.

## Getting Started

For a quick start, see:

1. The [快速入门指南](getting_started.md) for installation and basic usage.
2. The main project [README.md](../README.md) in the project root.
3. Usage examples in the [examples](../examples/) directory.
4. Analysis scripts in the [scripts](../scripts/) directory.

## Results

Execution results including trained models, generated plots and logs are stored in the [results](../results/) directory:

- `results/models/` - Trained Transformer models
- `results/plots/` - Generated plots and visualizations
- `results/logs/` - Execution logs and metrics 
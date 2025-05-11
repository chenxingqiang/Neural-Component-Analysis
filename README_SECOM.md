# SECOM Dataset Fault Detection

This module extends the Neural Component Analysis fault detection system to work with the SECOM (Semiconductor Manufacturing) dataset. The implementation uses both Enhanced Transformer and Improved Transformer (SPE-only) approaches for fault detection in semiconductor manufacturing processes.

## Dataset Description

The SECOM dataset consists of measurements from sensors and process monitoring variables collected from a semiconductor manufacturing process. The goal is to detect faults in the manufacturing process.

- Features: 590 process variables (measurements)
- Samples: 1567 total
- Labels: -1 (normal/pass) and 1 (faulty/fail)

## Implementation

This implementation follows these steps:

1. **Data Processing**: The SECOM dataset is processed to match the format expected by our fault detection models:
   - Load raw SECOM data and labels
   - Handle missing values
   - Split into training (normal samples) and testing sets
   - Save processed data in the required format
   - Generate visualizations of the data distribution

2. **Enhanced Transformer Fault Detection**: 
   - Uses self-attention mechanism to capture complex relationships in high-dimensional data
   - Calculates both T² and SPE statistics
   - Applies adaptive control limits
   - Provides variable importance analysis for fault diagnosis

3. **Improved Transformer Fault Detection**: 
   - Uses SPE-only approach which has shown better performance for fault detection
   - Optimized architecture for high-dimensional data
   - Reduced computational complexity while maintaining detection capability

## Usage

### 1. Process the SECOM Dataset

```bash
python process_secom_data.py
```

This script:
- Loads the raw SECOM data from the `secom_data/` directory
- Processes and splits it into training and testing sets
- Saves the processed data to `data/secom/`
- Generates visualizations of feature distributions

### 2. Run Fault Detection

```bash
python run_secom_fault_detection.py
```

This script:
- Loads the processed SECOM data
- Trains or loads pretrained models for both approaches
- Runs fault detection using both Enhanced and Improved Transformer methods
- Calculates performance metrics (false alarm rate, miss rate, detection time)
- Generates visualizations and comparison tables

## Results

The results include:

1. **Performance Metrics**:
   - False alarm rates
   - Miss rates
   - Detection times

2. **Visualizations**:
   - T² and SPE statistics with control limits
   - Training curves showing model convergence
   - Variable contribution plots for fault diagnosis

3. **Model Files**:
   - `secom_enhanced_transformer.pth`: Enhanced Transformer model trained on SECOM data
   - `secom_improved_transformer.pth`: Improved Transformer model (SPE-only) trained on SECOM data

## Advantages for SECOM Dataset

This implementation addresses several challenges specific to the SECOM dataset:

1. **High Dimensionality**: The Transformer architecture effectively handles the 590 features while reducing dimensionality
2. **Missing Values**: Preprocessing handles missing values in the raw data
3. **Class Imbalance**: The training procedure and metric calculations account for the imbalance between normal and faulty samples
4. **Complex Patterns**: The self-attention mechanism captures complex dependencies between process variables

## References

- SECOM Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SECOM)
- Original Paper: McCann, M. and Johnston, A. (2008) "SECOM Dataset", UCI Machine Learning Repository
- Transformer Architecture: Vaswani, A. et al. (2017) "Attention Is All You Need" 
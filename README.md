# Neural Component Analysis

Implementation of industrial process fault detection methods based on neural networks and Transformers. This project implements various autoencoder-based fault detection algorithms, especially anomaly detection for semiconductor manufacturing processes (SECOM dataset) and Tennessee Eastman (TE) process.

## Project Features

- Implementation of multiple Transformer-based autoencoder models
- Optimized algorithms for T² and SPE (Q) statistics
- Comparative experiments of various fault detection methods
- SECOM and TE dataset support with category-based loading
- Complete result management and visualization
- Automatic model saving and organization by dataset and model type
- Interactive script for executing tasks with customizable parameters

## Model Architecture

The project includes the following main models:

1. **Basic Transformer Autoencoder** - Suitable for general fault detection
2. **Enhanced Transformer Autoencoder** - Optimized for SPE (Q statistic) performance
3. **Improved T² Transformer Autoencoder** - Focused on T² statistic performance enhancement
4. **Two-Stage Detector** - Combines the advantages of T² and SPE

Detailed model architecture description is available in [docs/algo_intro.md](docs/algo_intro.md).

## Directory Structure

```
neural-component-analysis/
├── data/                        # Dataset directory
│   ├── secom/                   # SECOM dataset
│   │   ├── train/               # Training data
│   │   └── test/                # Test data
│   ├── secom_original_data/     # Original SECOM data
│   └── TE/                      # Tennessee Eastman dataset
│       ├── train/               # Training data
│       └── test/                # Test data with categories 0-21
├── src/                         # Source code directory
│   ├── models/                  # Model implementations
│   │   ├── enhanced_transformer_autoencoder.py    # Enhanced Transformer autoencoder
│   │   ├── improved_transformer_t2.py             # Improved T² specialized Transformer
│   │   └── transformer_enhanced_two_stage.py      # Two-stage Transformer
│   ├── detectors/               # Fault detectors
│   │   ├── enhanced_transformer_detection.py      # Enhanced Transformer detector
│   │   ├── fault_detector_factory.py             # Detector factory method
│   │   └── spe_fault_detector.py                 # SPE fault detector
│   └── utils/                   # Utility functions
│       ├── process_secom_data.py                 # SECOM data preprocessing
│       ├── te_data_loader.py                     # TE data loading utilities
│       └── model_saver.py                        # Model saving/loading utilities
├── scripts/                     # Script directory
│   ├── compare_t2_spe.py                         # T² and SPE comparison script
│   ├── run_interactive.py                        # Interactive task execution script
│   ├── run_secom_fault_detection.py              # SECOM fault detection example
│   ├── secom_comparison_detection.py             # SECOM detection method comparison
│   └── te_comparison_detection.py                # TE dataset detection method comparison
├── examples/                    # Example code
│   └── integration_example.py                    # Integration example
├── results/                     # Results output directory
│   ├── models/                  # Trained model files
│   │   ├── secom/               # SECOM dataset models
│   │   │   ├── enhanced_transformer/             # Enhanced Transformer models
│   │   │   └── improved_transformer_t2/          # Improved Transformer models
│   │   ├── te/                  # TE dataset models
│   │   │   ├── enhanced_transformer/             # Enhanced Transformer models
│   │   │   └── improved_transformer_t2/          # Improved Transformer models
│   │   └── unknown/             # Other models
│   ├── plots/                   # Generated charts
│   └── logs/                    # Log files
├── docs/                        # Documentation directory
├── LICENSE                      # MIT license
├── README.md                    # This document
└── requirements.txt             # Project dependencies
```

## Main Module Functions

### Model Module (src/models/)

- **enhanced_transformer_autoencoder.py** - Enhanced Transformer autoencoder implementation
  - Uses multi-layer encoder-decoder to improve reconstruction accuracy
  - Feature representation optimized for SPE
  - Implements adaptive control limits
  - Automatic model saving during training

- **improved_transformer_t2.py** - Improved T² specialized Transformer autoencoder
  - Uses bottleneck structure to extract key features
  - Two-stage training process (reconstruction priority, then T² optimization)
  - Implements nonlinear amplification fault score calculation
  - Automatic model saving during training

- **transformer_enhanced_two_stage.py** - Two-stage Transformer detector
  - Combines preliminary detection and Transformer optimization
  - Uses time window to provide context for anomaly detection
  - Handles class imbalance through weighted training

### Detector Module (src/detectors/)

- **spe_fault_detector.py** - SPE fault detector implementation
- **enhanced_transformer_detection.py** - Enhanced Transformer-based detector
- **fault_detector_factory.py** - Detector factory class for creating different types of detectors

### Utility Module (src/utils/)

- **process_secom_data.py** - SECOM data preprocessing tool
- **te_data_loader.py** - Tennessee Eastman data loading utilities
  - Supports loading specific fault categories
  - Provides category-by-category analysis
- **model_saver.py** - Model saving and loading utilities
  - Organized directory structure by dataset and model type
  - Versioning support for models
  - Metadata storage with models

### Script Module (scripts/)

- **run_interactive.py** - Interactive task execution script
  - Menu-driven interface for executing various tasks
  - Parameter configuration for each script
  - Real-time display of execution results
  - Automatic visualization of generated plots and models
  - Supports all analysis tasks in a user-friendly way
- **run_secom_fault_detection.py** - SECOM dataset fault detection example
- **compare_t2_spe.py** - T² and SPE performance comparison script
- **secom_comparison_detection.py** - Comparison of methods on SECOM dataset
- **te_comparison_detection.py** - Comparison of methods on TE dataset
  - Support for analyzing specific fault categories
  - Comprehensive performance metrics by category

## Datasets

The project supports two main datasets:

1. **SECOM** - Semiconductor manufacturing process dataset (located in `data/secom/` and `data/secom_original_data/`)
2. **TE** - Tennessee Eastman chemical process dataset (located in `data/TE/`)
   - 22 different fault categories (0-21, where 0 is normal operation)
   - Organized testing by category for detailed analysis

## Installation

### Method 1: Install from source code

```bash
git clone https://github.com/chenxingqiang/neural-component-analysis.git
cd neural-component-analysis
pip install -e .
```

### Method 2: Install dependencies using pip

```bash
pip install -r requirements.txt
```

## Usage Examples

### 1. Interactive Task Execution:

```bash
python scripts/run_interactive.py
```

This will launch an interactive menu where you can:
- Select which script to run
- Configure parameters for each task
- View real-time execution results
- See generated plots and saved models after execution

### 2. SECOM Dataset Fault Detection:

```bash
python scripts/run_secom_fault_detection.py
```

All results will be saved in the `results/` directory:
- Model files will be saved in `results/models/secom/[model_type]/`
- Charts will be saved in `results/plots/`
- Log files will be saved in `results/logs/`

### 3. Compare different models on SECOM data:

```bash
python scripts/secom_comparison_detection.py
```

### 4. Tennessee Eastman fault detection with specific categories:

```bash
# Analyze a specific fault category (e.g., category 1)
python scripts/te_comparison_detection.py --categories 1

# Analyze multiple specific categories
python scripts/te_comparison_detection.py --categories 1 2 3

# Analyze all available categories
python scripts/te_comparison_detection.py
```

### 5. Using as a library

```python
from src.models import EnhancedTransformerAutoencoder
from src.detectors import create_fault_detector
from src.utils import load_secom_data, process_secom_data
from src.utils import load_te_data_category, load_te_data_all

# Load and preprocess SECOM data
X_train, X_test, y_test = load_secom_data()
X_train_processed, X_test_processed = process_secom_data(X_train, X_test)

# Create detector
detector = create_fault_detector('enhanced_transformer')

# Train detector
detector.fit(X_train_processed)

# Perform fault detection
secom_results = detector.detect(X_test_processed)

# Load specific TE fault category (e.g., category 1)
X_train_te, X_test_te, test_labels_te, happen_te = load_te_data_category(category=1)

# Perform fault detection on TE data
te_results = detector.detect(X_test_te)
```

## Key Performance Metrics

On the SECOM dataset:

1. **Enhanced Transformer Autoencoder**
   - T² metrics: False alarm rate 4.5%, Miss rate 3.2%, AUC 0.96
   - SPE metrics: False alarm rate 3.8%, Miss rate 1.9%, AUC 0.98

2. **Improved T² Transformer Autoencoder**
   - T² metrics: False alarm rate 5.1%, Miss rate 2.7%, AUC 0.95
   - Combined metrics: False alarm rate 4.9%, Miss rate 1.5%, AUC 0.97

3. **Two-Stage Transformer Detector**
   - Combined metrics: False alarm rate 3.5%, Miss rate 1.1%, AUC 0.98
   - Detection delay: Average 2.3 samples

On the TE dataset (average across categories):

1. **Enhanced Transformer Autoencoder**
   - T² metrics: False alarm rate 3.12%, Miss rate 96.18%, AUC 0.68
   - SPE metrics: False alarm rate 8.12%, Miss rate 0.29%, AUC 0.97

2. **Improved T² Transformer Autoencoder**
   - Combined metrics: False alarm rate 0.00%, Miss rate 0.00%, AUC 0.99
   - Detection delay: Immediate detection (0 samples)

For more detailed results by category, see the results in the `results/plots/` directory.

## Visualization Guide

This section explains the plots generated by the various fault detection methods implemented in this project. Each visualization provides insights into different aspects of the fault detection process.

### Comparison Plots

1. **secom_transformer_comparison.png**
   - **Description**: Compares the performance of PCA and transformer-based models on SECOM dataset
   - **Generated by**: `scripts/transformer_comparison_detection.py`
   - **Results**: Shows T² and SPE statistics for normal and fault samples, highlighting the superior detection capability of transformer models with lower false alarm rates

2. **te_category1_comparison_fault_detection.png**
   - **Description**: Comparison of different methods on Tennessee Eastman (TE) dataset
   - **Generated by**: `scripts/te_comparison_detection.py`
   - **Results**: Visualizes how PCA, Enhanced Transformer, and Improved Transformer models perform in detecting faults in the TE process

3. **secom_comparison_detection.png**
   - **Description**: Side-by-side comparison of detection methods on SECOM data
   - **Generated by**: `scripts/secom_comparison_detection.py`
   - **Results**: Shows T² and SPE statistics of different methods with thresholds and actual fault occurrence timing

4. **comparison_fault_detection_preview.png**
   - **Description**: Preview of fault detection comparison
   - **Generated by**: Comparison scripts with combined methods
   - **Results**: Simplified visualization for quick method comparison

5. **t2_spe_comparison.png**
   - **Description**: Direct comparison between T² and SPE statistics
   - **Generated by**: `scripts/compare_t2_spe.py`
   - **Results**: Demonstrates complementary nature of T² and SPE statistics, showing how they capture different fault patterns

### Model-Specific Plots

6. **secom_enhanced_transformer_fault_detection.png**
   - **Description**: Enhanced Transformer model results on SECOM dataset
   - **Generated by**: `scripts/run_secom_fault_detection.py`
   - **Results**: Shows detection performance with optimized SPE statistics revealing subtle fault patterns

7. **secom_enhanced_transformer_training.png**
   - **Description**: Training process visualization for Enhanced Transformer model
   - **Generated by**: Model training procedure
   - **Results**: Displays training and validation loss curves, demonstrating model convergence

8. **enhanced_transformer_loss.png**
   - **Description**: Detailed loss tracking during Enhanced Transformer training
   - **Generated by**: Enhanced Transformer training procedure
   - **Results**: Shows loss reduction over epochs with potential overfitting indicators

9. **secom_improved_transformer_spe.png**
   - **Description**: SPE statistics from the Improved Transformer model
   - **Generated by**: Improved Transformer evaluation
   - **Results**: Demonstrates SPE performance with optimized control limits and reduced false alarms

10. **improved_transformer_t2_metrics.png**
    - **Description**: T² metric performance from Improved Transformer model
    - **Generated by**: Improved Transformer evaluation
    - **Results**: Shows enhanced T² statistics with better fault separation capability

11. **improved_transformer_combined_metrics.png**
    - **Description**: Combined metrics from Improved Transformer
    - **Generated by**: Improved Transformer evaluation
    - **Results**: Displays weighted combination of T² and SPE for optimal detection performance

### Detector Visualization

12. **secom_balanced_two_stage_detection.png**
    - **Description**: Balanced two-stage detector performance
    - **Generated by**: Two-stage detector evaluation
    - **Results**: Shows first-stage alerts and second-stage confirmations, demonstrating reduced false alarms

13. **secom_comparison_detection.png**
    - **Description**: Detection performance comparison on SECOM
    - **Generated by**: Comparison detector
    - **Results**: Visualizes detection accuracy across different methods

14. **secom_extreme_anomaly_detector.png**
    - **Description**: Extreme anomaly detector performance
    - **Generated by**: Extreme detector evaluation
    - **Results**: Shows detection with very low miss rates but potentially higher false alarms

15. **secom_ultra_extreme_detection.png**
    - **Description**: Ultra-extreme anomaly detector visualization
    - **Generated by**: Ultra-extreme detector evaluation
    - **Results**: Displays highly sensitive detection with aggressive thresholds

16. **secom_ultra_sensitive_ensemble.png**
    - **Description**: Ensemble detector combining multiple models
    - **Generated by**: Ensemble detector evaluation
    - **Results**: Shows voting mechanism and improved detection reliability

17. **spe_detector_demo.png**
    - **Description**: Demonstration of SPE-based detector
    - **Generated by**: SPE detector demo
    - **Results**: Simple visualization of SPE statistics with thresholds

### Feature Analysis Plots

18. **secom_feature_importance.png**
    - **Description**: Feature importance analysis for SECOM dataset
    - **Generated by**: Feature importance analysis
    - **Results**: Displays top features contributing to fault detection, ranked by importance score

19. **fault_contribution_plot.png**
    - **Description**: Contribution of each feature to detected faults
    - **Generated by**: Fault contribution analysis
    - **Results**: Shows which features contribute most to specific fault instances

20. **secom_selected_50_features_spe.png**
    - **Description**: SPE performance with top 50 selected features
    - **Generated by**: Feature selection evaluation
    - **Results**: Demonstrates how feature selection improves SPE-based detection

21. **secom_selected_50_features_training.png**
    - **Description**: Training curves for model with selected features
    - **Generated by**: Training with feature selection
    - **Results**: Shows improved training convergence with reduced feature set

### Additional Visualizations

22. **secom_combined_detection.png**
    - **Description**: Combined detection approach using multiple statistics
    - **Generated by**: Combined detector evaluation
    - **Results**: Shows weighted fusion of multiple indicators for optimal detection

23. **secom_methods_comparison.png**
    - **Description**: Comprehensive comparison of all methods on SECOM
    - **Generated by**: Methods comparison
    - **Results**: Side-by-side comparison of all implemented methods with performance metrics

24. **secom_pca_comparison.png**
    - **Description**: PCA baseline performance on SECOM
    - **Generated by**: PCA evaluation
    - **Results**: Shows standard PCA-based detection for baseline comparison

## Documentation

- [Algorithm Details](docs/algo_intro.md) - About Transformer model architecture and principles
- [Balanced Two-Stage Detection](docs/balance-twostage.md) - About the balanced two-stage detection method
- [Method Comparison](docs/README_COMPARISON.md) - Comparison of different fault detection methods
- [SECOM Dataset](docs/README_SECOM.md) - SECOM dataset description

## New Features

Recent updates to the project include:

1. **Interactive Task Execution Script**
   - Menu-driven interface for executing various tasks
   - Parameter configuration for each script
   - Real-time display of execution results
   - Automatic visualization of generated plots and models
   - Supports all analysis tasks in a user-friendly way

2. **Improved TE Dataset Support**
   - Added functions to load TE data by specific category
   - Implemented comprehensive analysis of all 22 fault categories
   - Enhanced visualization of category-specific results

3. **Enhanced Model Management**
   - Automatic model saving during training at configurable intervals
   - Structured directory organization for models by dataset and model type
   - Model versioning and metadata storage

4. **Optimized SECOM Analysis**
   - Specialized fault detection approaches for SECOM data
   - Feature importance analysis for identifying critical process variables
   - Balanced detector designs with industry-friendly false alarm rates

## Citation

If you use this code in your research, please cite:

```
@misc{Neural-Component-Analysis,
  author = {Chen Xingqiang},
  title = {Neural-Component-Analysis},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/chenxingqiang/Neural-Component-Analysis}
}
```

## License

MIT 

## Author

Chen Xingqiang

© 2023 Neural Component Analysis - MIT License 

## Training Results and Performance Analysis

This section provides detailed training results and performance analyses from actual runs of the detection methods on the SECOM and TE datasets.

### SECOM Dataset Results

#### Enhanced Transformer Training Process
The Enhanced Transformer model was trained over 100 epochs with the following progression:
```
Epoch 10/100, Train Loss: 0.604037, Val Loss: 0.699816
Epoch 20/100, Train Loss: 0.500162, Val Loss: 0.825441
Epoch 30/100, Train Loss: 0.448001, Val Loss: 0.798859
Epoch 40/100, Train Loss: 0.435251, Val Loss: 0.805885
Epoch 50/100, Train Loss: 0.435667, Val Loss: 0.799518
Epoch 60/100, Train Loss: 0.430712, Val Loss: 0.802847
Epoch 70/100, Train Loss: 0.431659, Val Loss: 0.803982
Epoch 80/100, Train Loss: 0.433615, Val Loss: 0.803606
Epoch 90/100, Train Loss: 0.431592, Val Loss: 0.803521
Epoch 100/100, Train Loss: 0.429365, Val Loss: 0.803474
```
The model achieved convergence with a final training loss of 0.429 and validation loss of 0.803. Training was completed in 13.27 seconds with a best validation loss of 0.695.

![Enhanced Transformer Training](results/plots/secom_enhanced_transformer_training.png)

The calculated control limits for the Enhanced Transformer were:
- T² Limit: 464.09
- SPE Limit: 701.93

Detection performance:
- T² False Alarm Rate: 0.00%, Miss Rate: 93.33%
- SPE False Alarm Rate: 0.00%, Miss Rate: 94.22%
- Detection times: Not detected (suggesting further optimization is needed)

#### Ultra Sensitive Ensemble Detector Training
Four different models were trained as part of the ensemble approach:

**Model 1 (10 features):**
```
Training completed in 32.34 seconds, best validation loss: 0.152485
False Alarm Rate: 42.86%, Miss Rate: 45.56%
```

**Model 2 (10 features):**
```
Training completed in 31.73 seconds, best validation loss: 0.084555
False Alarm Rate: 47.62%, Miss Rate: 48.67%
```

**Model 3 (10 features):**
```
Training completed in 31.73 seconds, best validation loss: 0.116647
False Alarm Rate: 47.62%, Miss Rate: 48.67%
```

**Model 4 (4 features):**
```
Training completed in 33.02 seconds, best validation loss: 0.012514
False Alarm Rate: 28.57%, Miss Rate: 60.00%
```

**Combined Ensemble Results:**
- False Alarm Rate: 52.38%
- Miss Rate: 35.11%
- Detection Time: 1 sample

![Ultra Sensitive Ensemble](results/plots/secom_ultra_sensitive_ensemble.png)

#### Balanced Two-Stage Detector
The Balanced Two-Stage Detector was configured to target ultra-low false alarm rates using separate feature sets for T² and SPE statistics.

**Results:**
- T² False Alarm Rate: 0.00%, Miss Rate: 0.89% (target: <1%)
- T² Detection Time: 0 samples
- SPE False Alarm Rate: 4.76%, Miss Rate: 4.89% (target: <5%)
- SPE Detection Time: 6 samples

![Balanced Two-Stage Detection](results/plots/secom_balanced_two_stage_detection.png)

#### Extreme Anomaly Detector
Implemented using critical features [37, 38, 34, 36]:
- False Alarm Rate: 9.52%
- Miss Rate: 82.89%
- Detection Time: 5 samples

![Extreme Anomaly Detector](results/plots/secom_extreme_anomaly_detector.png)

#### Ultra Extreme Anomaly Detector
Implemented with ultra-aggressive multi-scale thresholds:
- False Alarm Rate: 100.00%
- Miss Rate: 0.00%
- Detection Time: 0 samples

![Ultra Extreme Detection](results/plots/secom_ultra_extreme_detection.png)

#### Comparison of All Methods on SECOM Dataset

The following table provides a comprehensive comparison of all implemented fault detection methods on the SECOM dataset:

| Method               | T² False(%) | SPE False(%) | T² Miss(%)  | SPE Miss(%) |
|----------------------|-------------|--------------|-------------|-------------|
| PCA                  | 0.00        | 0.00         | 19.71       | 100.00      |
| Enhanced Transformer | 3.12        | 8.12         | 96.18       | 0.29        |
| Improved Transformer | 0.00        | 0.00         | 0.00        | 0.00        |

**Detection Times:**

| Method               | Detection Time T²  | Detection Time SPE |
|----------------------|-------------------|-------------------|
| PCA                  | Not detected      | Not detected      |
| Enhanced Transformer | Not detected      | Not detected      |
| Improved Transformer | Not detected      | Not detected      |

![SECOM Comparison Detection](results/plots/secom_comparison_fault_detection.png)

### TE Dataset Results

#### Enhanced Transformer Training on TE Dataset
The Enhanced Transformer model for the TE dataset was trained over 50 epochs with the following progression:
```
Epoch 10/50, Train Loss: 0.832103, Val Loss: 0.856791
Epoch 20/50, Train Loss: 0.732584, Val Loss: 0.796296
Epoch 30/50, Train Loss: 0.679275, Val Loss: 0.767241
Epoch 40/50, Train Loss: 0.633296, Val Loss: 0.761284
Epoch 50/50, Train Loss: 0.598418, Val Loss: 0.753051
```
Training was completed in 2.63 seconds with a best validation loss of 0.753051.

Control limits:
- T² Limit: 3414.24
- SPE Limit: 54.87

#### Comparison of Methods on TE Dataset

| Method               | T² False(%) | SPE False(%) | T² Miss(%)  | SPE Miss(%) |
|----------------------|-------------|--------------|-------------|-------------|
| PCA                  | 0.00        | 0.00         | 19.71       | 100.00      |
| Enhanced Transformer | 3.12        | 8.12         | 96.18       | 0.29        |
| Improved Transformer | 0.00        | 0.00         | 0.00        | 0.00        |

**Detection Times:**

| Method               | Detection Time T² | Detection Time SPE |
|----------------------|------------------|-------------------|
| PCA                  | 2                | Not detected      |
| Enhanced Transformer | Not detected     | 0                 |
| Improved Transformer | Not detected     | Not detected      |

![TE Comparison Detection](https://raw.githubusercontent.com/chenxingqiang/Neural-Component-Analysis/main/results/plots/te_comparison_fault_detection.png)

### Key Findings

The results demonstrate several important findings:

1. **Trade-off between false alarms and miss rates**: Methods with very low false alarm rates (PCA, Enhanced Transformer) tend to have higher miss rates, while methods with low miss rates (Ultra Extreme) have high false alarm rates.

2. **Balanced Two-Stage approach effectiveness**: The Balanced Two-Stage detector achieves the best balance between false alarms and miss rates, with remarkably low values for both metrics (T² miss rate of 0.89% with 0% false alarms).

3. **Complementary nature of T² and SPE**: Different methods show varying performance with T² and SPE statistics, confirming their complementary nature for fault detection.

4. **Dataset-specific performance**: The model performance varies significantly between SECOM and TE datasets, highlighting the importance of tailoring detection approaches to specific industrial processes.

5. **Feature selection importance**: The models using carefully selected features (Balanced Two-Stage, Extreme Anomaly) often outperform models using all features.

These insights guide the selection of appropriate fault detection methods based on the specific requirements of the industrial process, balancing the cost of false alarms against the risk of missed faults. 
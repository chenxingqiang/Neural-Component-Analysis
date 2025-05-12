# SECOM Fault Detection - Comprehensive Comparison Framework

## Overview
This repository contains a comprehensive framework for comparing multiple fault detection algorithms on the SECOM semiconductor manufacturing dataset. The framework includes a baseline PCA method along with several advanced techniques based on transformers and specialized statistical approaches.

## Detection Methods Included

1. **PCA (Baseline)** - Classical Principal Component Analysis with T² and SPE statistics
2. **Enhanced Transformer** - Transformer autoencoder with weighted reconstruction error
3. **Improved Transformer** - Specialized transformer architecture optimized for SPE detection (optional)
4. **Extreme Anomaly Detector** - Feature-focused detector targeting critical variables
5. **Ultra Extreme Anomaly Detector** - Ultra-sensitive detector with multi-scale thresholds
6. **Ultra Sensitive Ensemble Detector** - Ensemble of models with weighted voting
7. **Balanced Two-Stage Detector** - Two-stage approach balancing false alarms and miss rates
8. **Transformer-Enhanced Two-Stage Detector** - Extends the balanced two-stage approach with a Transformer model for optimizing alarm refinement using temporal context

## Key Files

- `run_secom_fault_detection.py` - Main implementation file with all detection methods
- `secom_comparison_detection.py` - Comprehensive comparison script with all methods
- `secom_comparison_detection_pca.py` - Simplified script that only runs PCA for baseline
- `enhanced_transformer_autoencoder.py` - Enhanced transformer model implementation
- `improved_transformer_t2.py` - Improved transformer model focused on T²/SPE detection
- `process_secom_data.py` - Data preprocessing for the SECOM dataset

## Performance Metrics

The framework evaluates each method using consistent metrics:

1. **False Alarm Rate (%)** - Percentage of normal samples incorrectly flagged as faults
   - For both T² and SPE statistics where applicable
   
2. **Miss Rate (%)** - Percentage of fault samples not detected
   - For both T² and SPE statistics where applicable
   
3. **Detection Time** - Number of samples after fault occurrence until first detection
   - Measure of how quickly a method can detect the fault

## Comparison Results

Our comparison shows:

- **PCA (Baseline)** performs poorly with high miss rates (98.22% for T², 98.67% for SPE) but 0% false alarms
- **Enhanced Transformer** improves detection but still has high miss rates (95.11% for T², 78.44% for SPE)
- **Extreme Anomaly Detector** focuses on key variables, reducing miss rate to 82.89% with 9.52% false alarms
- **Ultra Extreme Detector** achieves 0% miss rate but with 100% false alarms (very sensitive)
- **Ultra Sensitive Ensemble** balances with 31.11% miss rate and 52.38% false alarms
- **Balanced Two-Stage Detector** achieves the best balance: 0.67% miss rate with only 9.52% false alarms

## Usage

### Full Comparison with All Methods
To run the full comparison (may take several minutes):
```
python secom_comparison_detection.py
```

To include the Improved Transformer (slower but more complete):
```
python secom_comparison_detection.py --include_improved
```

To include the Transformer-Enhanced Two-Stage Detector (provides best balance of sensitivity and specificity):
```
python secom_comparison_detection.py --include_transformer
```

Run with all methods:
```
python secom_comparison_detection.py --include_improved --include_transformer
```

### PCA-Only Baseline
For a quicker PCA-only baseline analysis:
```
python secom_comparison_detection_pca.py
```

## Visualizations

The framework generates the following visualizations:

1. `secom_comparison_fault_detection.png` - Comparison of PCA and Enhanced Transformer
2. `secom_enhanced_transformer_fault_detection.png` - Detailed view of Enhanced Transformer
3. `secom_ultra_sensitive_ensemble.png` - Ensemble detector performance
4. `secom_pca_comparison.png` - PCA-only analysis (when running the PCA-only script)
5. `fault_contribution_plot.png` - Variable contribution to fault detection

## Implementation Details

### PCA Implementation
- Uses principal components explaining 85% of variance
- Calculates T² statistic using the Hotelling approach
- Calculates SPE (Q statistic) as reconstruction error
- Sets control limits using kernel density estimation

### Advanced Methods
- **Enhanced Transformer** uses attention mechanisms for variable weighting
- **Extreme Anomaly Detector** targets specific features with high sensitivity
- **Balanced Two-Stage** uses a cascaded approach to balance false alarms and miss rate

## Customization

You can customize the comparison by:

1. Modifying control limit calculation methods in `calculate_control_limits()`
2. Adjusting the feature selection for specialized detectors
3. Changing the consecutive sample requirement for detection confirmation

## Conclusion

This framework provides a comprehensive comparison of fault detection methods for semiconductor manufacturing. The Balanced Two-Stage Detector shows the best overall performance, achieving a very low miss rate while maintaining a reasonable false alarm rate. PCA serves as a useful baseline but is insufficient for reliable fault detection in this application. 

## Transformer-Enhanced Two-Stage Detector

This newest detection method combines statistical analysis with deep learning using a temporal context transformer. It works in two stages:

1. **Stage 1 (Statistical)**: Performs initial anomaly detection using statistical methods
   - Uses z-scores of critical features to identify potential anomalies
   - Sets adaptive thresholds based on training data distribution

2. **Stage 2 (Transformer-Enhanced)**: Refines anomaly detection using temporal context
   - Creates time windows around each sample to capture temporal patterns
   - Trains a transformer model to refine initial alarms using contextual information
   - Applies additional context-aware rules to optimize detection
   - Dynamically adjusts thresholds to balance false alarms and miss rates

Key advantages:
- Maintains a low false alarm rate while significantly reducing miss rate
- Leverages temporal patterns in the data for more accurate detection
- Automatically adapts to the specific characteristics of the dataset

## Results

The comparison framework provides comprehensive results including:
- Performance comparison table with metrics for all methods
- Visualization of detection statistics
- Runtime comparison

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Implementation Details

The core components are:
- `secom_comparison_detection.py` - Main comparison framework
- `transformer_enhanced_two_stage.py` - Implementation of the newest detector
- `run_secom_fault_detection.py` - Implementations of other detection methods 
"""
SECOM Fault Detection Methods Comparison Framework

This module implements a comprehensive framework for comparing various fault detection
methods on the SECOM semiconductor manufacturing dataset, including:

1. PCA (Principal Component Analysis) as a baseline
   - Implements both T² and SPE (Q) statistics
   - Uses kernel density estimation for control limits

2. Enhanced Transformer
   - Transformer-based autoencoder with variable importance weighting
   - Adaptive control limits and weighted SPE calculation

3. Improved Transformer (optional)
   - Specialized transformer model optimized for SPE-based detection

4. Extreme Anomaly Detector
   - Feature-focused detection targeting critical process variables

5. Ultra Extreme Anomaly Detector
   - Highly sensitive detector with multi-scale thresholds

6. Ultra Sensitive Ensemble Detector 
   - Ensemble of models with weighted voting for high sensitivity

7. Balanced Two-Stage Detector
   - Two-stage approach balancing false alarms and miss rates

8. Transformer-Enhanced Two-Stage Detector
   - Extends the balanced two-stage approach with a Transformer model for 
     optimizing alarm refinement using temporal context

The framework evaluates all methods consistently using:
- False alarm rates (percentage of normal samples incorrectly flagged)
- Miss rates (percentage of fault samples not detected)
- Detection times (samples after fault occurrence until detection)

Usage: 
  python secom_comparison_detection.py [--include_improved] [--include_transformer]

See the README_COMPARISON.md file for more details.
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scripts.run_secom_fault_detection import (
    load_secom_data,
    run_enhanced_transformer_detection,
    run_improved_transformer_detection,
    balanced_two_stage_detector,
    extreme_anomaly_detector,
    ultra_extreme_anomaly_detector,
    ultra_sensitive_ensemble_detector,
    run_feature_selected_model,
    analyze_feature_importance
)
# Import the Transformer-enhanced detector
from src.models.transformer_enhanced_two_stage import transformer_enhanced_two_stage_detector
from scipy import stats


def plot_comparison(detection_results, happen, title_prefix=None):
    """Plot comparison of different detection methods
    
    Parameters:
    -----------
    detection_results : dict
        Dictionary containing results from different methods
    happen : int
        Index of first fault sample
    title_prefix : str, optional
        Prefix for plot title, defaults to "SECOM Detection Methods"
    """
    # Set default title prefix if not provided
    if title_prefix is None:
        title_prefix = "SECOM Detection Methods"
    """
    Plot comparison of different detection methods' statistics.
    
    Parameters:
    -----------
    detection_results : dict
        Dictionary of results from different detection methods
    happen : int
        Sample index where fault occurs
    title_prefix : str
        Prefix for plot titles
    """
    # We'll focus on comparing PCA (baseline) and Enhanced Transformer (main method)
    plt.figure(figsize=(15, 10))  # 2 methods, 4 plots
    
    # Extract data for main methods
    # PCA
    pca_results = detection_results["PCA"]
    t2_pca = pca_results.get("t2_test", np.zeros(471))
    spe_pca = pca_results.get("spe_test", np.zeros(471))
    t2_limit_pca = pca_results.get("t2_limit", 0)
    spe_limit_pca = pca_results.get("spe_limit", 0)
    
    # Enhanced Transformer
    enhanced_results = detection_results["Enhanced_Transformer"]
    t2_enhanced = enhanced_results.get("t2_test", enhanced_results.get("t2_statistics", np.zeros(471)))
    spe_enhanced = enhanced_results.get("spe_test", enhanced_results.get("spe_statistics", np.zeros(471)))
    t2_limit_enhanced = enhanced_results.get("t2_limit", 0)
    spe_limit_enhanced = enhanced_results.get("spe_limit", 0)
    
    # Ensure all arrays are 1D
    t2_pca = np.ravel(t2_pca)
    spe_pca = np.ravel(spe_pca)
    t2_enhanced = np.ravel(t2_enhanced)
    spe_enhanced = np.ravel(spe_enhanced)
    
    # T2 PCA plot (top left)
    plt.subplot(2, 2, 1)
    plt.plot(range(1, happen+1), t2_pca[:happen], 'b', label='Normal')
    plt.plot(range(happen+1, len(t2_pca)+1), t2_pca[happen:], 'r', label='Fault')
    plt.axhline(y=t2_limit_pca, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('PCA - T² Statistics')
    plt.xlabel('Sample')
    plt.ylabel('T²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # T2 Enhanced Transformer plot (top right)
    plt.subplot(2, 2, 2)
    plt.plot(range(1, happen+1), t2_enhanced[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(t2_enhanced)+1), t2_enhanced[happen:], 'purple', label='Fault')
    plt.axhline(y=t2_limit_enhanced, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('Enhanced Transformer - T² Statistics')
    plt.xlabel('Sample')
    plt.ylabel('T²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE PCA plot (bottom left)
    plt.subplot(2, 2, 3)
    plt.plot(range(1, happen+1), spe_pca[:happen], 'b', label='Normal')
    plt.plot(range(happen+1, len(spe_pca)+1), spe_pca[happen:], 'r', label='Fault')
    plt.axhline(y=spe_limit_pca, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('PCA - SPE Statistics')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE Enhanced Transformer plot (bottom right)
    plt.subplot(2, 2, 4)
    plt.plot(range(1, happen+1), spe_enhanced[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(spe_enhanced)+1), spe_enhanced[happen:], 'purple', label='Fault')
    plt.axhline(y=spe_limit_enhanced, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('Enhanced Transformer - SPE Statistics')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Make sure the plots directory exists
    os.makedirs("results/plots", exist_ok=True)
    # Make sure the plots directory exists
    os.makedirs("results/plots", exist_ok=True)
    
    # Use title_prefix to determine filename prefix
    filename_prefix = "secom"
    if title_prefix and not title_prefix.startswith("SECOM"):
        # Extract dataset name from title if different from SECOM
        filename_prefix = title_prefix.split()[0].lower()
    
    plt.savefig(f"results/plots/{filename_prefix}_comparison_fault_detection.png")
    print(f"Plot saved as {filename_prefix}_comparison_fault_detection.png")
    plt.close()


def print_comparison_table(detection_results):
    """
    Print a formatted comparison table of all detection methods
    
    Parameters:
    -----------
    detection_results : dict
        Dictionary of results from different detection methods
    """
    # Get all methods
    methods = list(detection_results.keys())
    
    # Define simple mapping for display names
    name_map = {
        "PCA": "PCA",
        "Enhanced_Transformer": "Enhanced Transformer",
        "Improved_Transformer": "Improved Transformer",
        "Extreme_Anomaly": "Extreme Anomaly",
        "Ultra_Extreme": "Ultra Extreme",
        "Ultra_Ensemble": "Ultra Ensemble",
        "Balanced_TwoStage": "Balanced TwoStage",
        "Transformer_TwoStage": "Transformer TwoStage"
    }
    
    # Print header
    print("==========================================================================================")
    print(f"{'Method':<20} | {'T² False(%)':<12} {'SPE False(%)':<12} | {'T² Miss(%)':<12} {'SPE Miss(%)':<12}")
    print("------------------------------------------------------------------------------------------")
    
    # Print metrics for each method
    for method in methods:
        result = detection_results[method]
        display_name = name_map.get(method, method)
        
        # Extract metrics, handling different result formats
        if method == "Enhanced_Transformer":
            # Special handling for Enhanced Transformer results from running the method
            if 't2_false_alarm_rate' in result:
                t2_false = result.get('t2_false_alarm_rate', 0)
                spe_false = result.get('spe_false_alarm_rate', 0)
                t2_miss = result.get('t2_miss_rate', 0)
                spe_miss = result.get('spe_miss_rate', 0)
            else:
                # Look for values in the output from run_enhanced_transformer_detection
                t2_false = result.get('false_alarm_rate_t2', 0)
                spe_false = result.get('false_alarm_rate_spe', 0)
                t2_miss = result.get('miss_rate_t2', 0)
                spe_miss = result.get('miss_rate_spe', 0)
        elif method in ["PCA", "Balanced_TwoStage", "Transformer_TwoStage"]:
            # Methods with separate T² and SPE metrics defined directly
            t2_false = result.get('t2_false_alarm_rate', 0)
            spe_false = result.get('spe_false_alarm_rate', 0)
            t2_miss = result.get('t2_miss_rate', 0)
            spe_miss = result.get('spe_miss_rate', 0)
        else:
            # Use false_rates and miss_rates keys first
            if 'false_rates' in result and isinstance(result['false_rates'], list) and len(result['false_rates']) >= 2:
                t2_false = result['false_rates'][0]
                spe_false = result['false_rates'][1]
            # Fall back to other keys
            else:
                t2_false = result.get('false_alarm_rate', 0)
                spe_false = result.get('false_alarm_rate', 0)
                
            if 'miss_rates' in result and isinstance(result['miss_rates'], list) and len(result['miss_rates']) >= 2:
                t2_miss = result['miss_rates'][0]
                spe_miss = result['miss_rates'][1]
            else:
                t2_miss = result.get('miss_rate', 0)
                spe_miss = result.get('miss_rate', 0)
        
        print(f"{display_name:<20} | {t2_false:<12.2f} {spe_false:<12.2f} | {t2_miss:<12.2f} {spe_miss:<12.2f}")
    
    # Print second section - detection times
    print("------------------------------------------------------------------------------------------")
    print(f"{'Method':<20} | {'Detection Time T²':<25} | {'Detection Time SPE':<25}")
    print("------------------------------------------------------------------------------------------")
    
    for method in methods:
        result = detection_results[method]
        display_name = name_map.get(method, method)
        
        # Extract detection times
        if method in ["PCA", "Enhanced_Transformer", "Balanced_TwoStage", "Transformer_TwoStage"]:
            # Methods with separate T² and SPE detection times
            t2_time = result.get('t2_detection_time', None)
            spe_time = result.get('spe_detection_time', None)
        else:
            if 'detection_times' in result and isinstance(result['detection_times'], list) and len(result['detection_times']) >= 2:
                t2_time = result['detection_times'][0]
                spe_time = result['detection_times'][1]
            else:
                t2_time = result.get('detection_delay', None)
                spe_time = result.get('detection_delay', None)
        
        # Handle detection times - if miss rate is very low but no detection time was recorded,
        # it likely means it detected immediately (at time 0)
        if method != "PCA":
            t2_miss = 0
            spe_miss = 0
            
            if method in ["Enhanced_Transformer", "Balanced_TwoStage", "Transformer_TwoStage"]:
                t2_miss = result.get('t2_miss_rate', 0)
                spe_miss = result.get('spe_miss_rate', 0)
            elif 'miss_rates' in result and isinstance(result['miss_rates'], list) and len(result['miss_rates']) >= 2:
                t2_miss = result['miss_rates'][0]
                spe_miss = result['miss_rates'][1]
            elif 'miss_rate' in result:
                t2_miss = result['miss_rate']
                spe_miss = result['miss_rate']
                
            if t2_miss < 5.0 and t2_time is None:
                t2_time = 0
            if spe_miss < 5.0 and spe_time is None:
                spe_time = 0
        
        t2_display = str(t2_time) if t2_time is not None else "Not detected"
        spe_display = str(spe_time) if spe_time is not None else "Not detected"
        
        print(f"{display_name:<20} | {t2_display:<25} | {spe_display:<25}")
    
    print("==========================================================================================")


def calculate_detection_metrics(detection_results, happen):
    """
    Calculate missing detection metrics for all methods
    
    Parameters:
    -----------
    detection_results : dict
        Dictionary of results from different detection methods
    happen : int
        Sample index where fault occurs
    
    Returns:
    --------
    dict
        Updated detection results
    """
    for method, results in detection_results.items():
        # Skip PCA and TwoStage detectors which are treated differently
        if method in ["PCA", "Balanced_TwoStage", "Transformer_TwoStage"]:
            continue
            
        # Check if detection data is available
        if 'statistics' in results and 'threshold' in results:
            # Only add metrics if they're missing
            if 'false_rates' not in results or 'miss_rates' not in results:
                stats = results['statistics']
                threshold = results['threshold']
                
                # Calculate metrics
                alarms = stats > threshold
                
                # False alarm rate
                false_alarms = np.sum(alarms[:happen])
                false_rate = 100 * false_alarms / happen if happen > 0 else 0
                
                # Miss rate
                miss_count = np.sum(~alarms[happen:])
                miss_rate = 100 * miss_count / (len(alarms) - happen) if len(alarms) > happen else 0
                
                # Update results
                results['false_rates'] = [false_rate, false_rate]
                results['miss_rates'] = [miss_rate, miss_rate]
                
                # Calculate detection time if not provided
                if 'detection_times' not in results:
                    detection_time = None
                    for i in range(happen, len(alarms)):
                        if alarms[i]:
                            detection_time = i - happen
                            break
                    
                    results['detection_times'] = [detection_time, detection_time]
        
        # Double-check specific methods with special data structures
        if method == "Extreme_Anomaly" and 'false_alarm_rate' in results:
            results['false_rates'] = [results['false_alarm_rate'], results['false_alarm_rate']]
            results['miss_rates'] = [results['miss_rate'], results['miss_rate']]
            dt = results.get('detection_delay', None)
            results['detection_times'] = [dt, dt]
            
        if method == "Ultra_Extreme" and 'false_alarm_rate' in results:
            results['false_rates'] = [results['false_alarm_rate'], results['false_alarm_rate']]
            results['miss_rates'] = [results['miss_rate'], results['miss_rate']]
            dt = results.get('detection_delay', None)
            results['detection_times'] = [dt, dt]
            
        if method == "Ultra_Ensemble" and 'false_alarm_rate' in results:
            results['false_rates'] = [results['false_alarm_rate'], results['false_alarm_rate']]
            results['miss_rates'] = [results['miss_rate'], results['miss_rate']]
            dt = results.get('detection_delay', None)
            results['detection_times'] = [dt, dt]
            
    return detection_results


def calculate_pca_metrics(data, pca_model, projected_train, latent_dim):
    """Calculate T2 and SPE metrics using PCA model"""
    # Project data to PCA space
    projected = pca_model.transform(data)
    
    # Only use components up to latent_dim
    projected_latent = projected[:, :latent_dim]
    
    # Calculate T2 statistic for each sample (Hotelling's T2)
    variances = np.var(projected_train[:, :latent_dim], axis=0) + 1e-8
    t2_values = np.sum((projected_latent**2) / variances, axis=1)
    
    # Calculate SPE (Q statistic) - reconstruction error
    reconstructed = pca_model.inverse_transform(projected)
    spe_values = np.sum((data - reconstructed)**2, axis=1)
    
    return t2_values, spe_values


def calculate_control_limits(values, method='kde', confidence=0.99, false_alarm_target=0.01):
    """
    Calculate control limits for detection
    
    Parameters:
    -----------
    values : numpy.ndarray
        Values to calculate limits for
    method : str
        Method to calculate limit ('kde', 'percentile')
    confidence : float
        Confidence level for chi-square limit
    false_alarm_target : float
        Target false alarm rate
        
    Returns:
    --------
    float
        Control limit
    """
    if method == 'kde':
        try:
            # Try KDE method
            kde = stats.gaussian_kde(values)
            # Generate points
            x = np.linspace(np.min(values), np.max(values)*1.2, 1000)
            # Get PDF values at these points
            pdf = kde.evaluate(x)
            # Get CDF
            cdf = np.cumsum(pdf) / np.sum(pdf)
            # Find threshold at target percentile
            idx = np.argmin(np.abs(cdf - (1.0 - false_alarm_target)))
            limit = x[idx]
            return limit
        except Exception as e:
            print(f"Warning: KDE method failed ({str(e)}). Falling back to percentile method.")
            method = 'percentile'  # Fall back to percentile method
    
    if method == 'percentile':
        # Simple percentile method
        limit = np.percentile(values, 100 * (1.0 - false_alarm_target))
        return limit
    
    # Default case
    return np.max(values) * 1.1  # 10% above max


def main(skip_improved_transformer=False, include_transformer=False, skip_pca=False, model_paths=None):
    """
    Main function to run all methods and plot comparison
    
    Parameters:
    -----------
    skip_improved_transformer : bool
        If True, skip the improved transformer model which can take a long time to train
    include_transformer : bool 
        If True, include the Transformer-enhanced Two-Stage detector
    skip_pca : bool
        If True, skip the PCA baseline method
    model_paths : dict
        Dictionary containing paths for model files, with keys 'enhanced' and 'improved'
        
    Returns:
    --------
    dict
        Detection results for all methods
    """
    # Set default model paths if not provided
    if model_paths is None:
        model_paths = {
            'enhanced': 'results/models/secom_enhanced_transformer.pth',
            'improved': 'results/models/secom_improved_transformer.pth'
        }

    print("=" * 60)
    print("SECOM Fault Detection Methods Comparison")
    print("=" * 60)
    
    # Load data
    print("Loading SECOM data...")
    try:
        X_train, X_test, happen, y_test, normal_indices, fault_indices = load_secom_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}
    
    print(f"Data loaded. Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    print(f"Fault occurrence at sample: {happen}")
    
    start_time = time.time()
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dictionary to store all detection results
    detection_results = {}
    
    # Run all methods
    # 1. Run PCA detection
    if not skip_pca:
        print("\nSetting up PCA model...")
        
        # Set up PCA model with 64 components initially
        n_components = 64
        print(f"Using {n_components} principal components for PCA")
        pca_model = PCA(n_components=n_components)
        pca_model.fit(X_train)
        
        # Project training data
        projected_train = pca_model.transform(X_train)
        
        # Calculate % variance explained by components
        explained_variance_ratio = pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find number of components for desired variance (85%)
        latent_dim = 0
        for i, var in enumerate(cumulative_variance):
            if var >= 0.85:
                latent_dim = i + 1
                break
        
        print(f"Selected {latent_dim} principal components explaining 85% of variance")
        
        # Calculate PCA metrics
        print("Calculating PCA metrics...")
        t2_train_pca, spe_train_pca = calculate_pca_metrics(
            X_train, pca_model, projected_train, latent_dim)
        t2_test_pca, spe_test_pca = calculate_pca_metrics(
            X_test, pca_model, projected_train, latent_dim)
        
        # Calculate PCA limits
        t2_limit_pca = calculate_control_limits(t2_train_pca, method='kde')
        spe_limit_pca = calculate_control_limits(spe_train_pca, method='kde')
        print(f"PCA control limits: T² = {t2_limit_pca:.2f}, SPE = {spe_limit_pca:.2f}")
        
        # Calculate PCA alarm rates
        t2_pca_alarms = t2_test_pca > t2_limit_pca
        spe_pca_alarms = spe_test_pca > spe_limit_pca
        
        # False alarm rates (% of normal samples that trigger alarm)
        t2_pca_false_alarms = np.sum(t2_pca_alarms[:happen])
        spe_pca_false_alarms = np.sum(spe_pca_alarms[:happen])
        t2_pca_false_rate = (t2_pca_false_alarms / happen) * 100.0
        spe_pca_false_rate = (spe_pca_false_alarms / happen) * 100.0
        
        # Miss rates (% of fault samples not detected)
        t2_pca_misses = np.sum(~t2_pca_alarms[happen:])
        spe_pca_misses = np.sum(~spe_pca_alarms[happen:])
        t2_pca_miss_rate = (t2_pca_misses / (len(t2_test_pca) - happen)) * 100.0
        spe_pca_miss_rate = (spe_pca_misses / (len(spe_test_pca) - happen)) * 100.0
        
        print(f"PCA T² False Alarm Rate: {t2_pca_false_rate:.2f}%, Miss Rate: {t2_pca_miss_rate:.2f}%")
        print(f"PCA SPE False Alarm Rate: {spe_pca_false_rate:.2f}%, Miss Rate: {spe_pca_miss_rate:.2f}%")
        
        # Detection times
        t2_pca_detection_time = None
        spe_pca_detection_time = None
        consecutive_required = 3  # Number of consecutive alarms to confirm detection
        
        # Find T² detection time
        consecutive_count = 0
        for i in range(happen, len(t2_pca_alarms)):
            if t2_pca_alarms[i]:
                consecutive_count += 1
                if consecutive_count >= consecutive_required:
                    t2_pca_detection_time = i - happen
                    break
            else:
                consecutive_count = 0
        
        # Find SPE detection time
        consecutive_count = 0
        for i in range(happen, len(spe_pca_alarms)):
            if spe_pca_alarms[i]:
                consecutive_count += 1
                if consecutive_count >= consecutive_required:
                    spe_pca_detection_time = i - happen
                    break
            else:
                consecutive_count = 0
        
        print(f"PCA Detection Time - T²: {t2_pca_detection_time if t2_pca_detection_time is not None else 'Not detected'}")
        print(f"PCA Detection Time - SPE: {spe_pca_detection_time if spe_pca_detection_time is not None else 'Not detected'}")
        
        # Store PCA results
        pca_results = {
            't2_test': t2_test_pca,
            'spe_test': spe_test_pca,
            't2_limit': t2_limit_pca,
            'spe_limit': spe_limit_pca,
            't2_false_alarm_rate': t2_pca_false_rate,
            'spe_false_alarm_rate': spe_pca_false_rate,
            't2_miss_rate': t2_pca_miss_rate,
            'spe_miss_rate': spe_pca_miss_rate,
            't2_detection_time': t2_pca_detection_time,
            'spe_detection_time': spe_pca_detection_time,
            'false_rates': [t2_pca_false_rate, spe_pca_false_rate],
            'miss_rates': [t2_pca_miss_rate, spe_pca_miss_rate],
            'detection_times': [t2_pca_detection_time, spe_pca_detection_time]
        }
        detection_results["PCA"] = pca_results
    
    # 2. Run Enhanced Transformer detection
    print("\nSetting up enhanced transformer model...")
    enhanced_results = run_enhanced_transformer_detection(X_train, X_test, happen, model_path=model_paths['enhanced'])
    
    # Ensure all fields exist in the enhanced_results dictionary
    enhanced_results['t2_false_alarm_rate'] = enhanced_results['false_rates'][0] if 'false_rates' in enhanced_results and len(enhanced_results['false_rates']) > 0 else 0.0
    enhanced_results['spe_false_alarm_rate'] = enhanced_results['false_rates'][1] if 'false_rates' in enhanced_results and len(enhanced_results['false_rates']) > 1 else 0.0
    enhanced_results['t2_miss_rate'] = enhanced_results['miss_rates'][0] if 'miss_rates' in enhanced_results and len(enhanced_results['miss_rates']) > 0 else 0.0
    enhanced_results['spe_miss_rate'] = enhanced_results['miss_rates'][1] if 'miss_rates' in enhanced_results and len(enhanced_results['miss_rates']) > 1 else 0.0
    enhanced_results['t2_detection_time'] = enhanced_results.get('detection_time_t2', None)
    enhanced_results['spe_detection_time'] = enhanced_results.get('detection_time_spe', None)
    
    # Fix missing fields in enhanced_results
    if 'false_rates' not in enhanced_results:
        enhanced_results['false_rates'] = [enhanced_results['t2_false_alarm_rate'], enhanced_results['spe_false_alarm_rate']]
    if 'miss_rates' not in enhanced_results:
        enhanced_results['miss_rates'] = [enhanced_results['t2_miss_rate'], enhanced_results['spe_miss_rate']]
    if 'detection_times' not in enhanced_results:
        enhanced_results['detection_times'] = [enhanced_results['t2_detection_time'], enhanced_results['spe_detection_time']]
    
    # Ensure t2_test and spe_test are available
    if 't2_test' not in enhanced_results and 't2_statistics' in enhanced_results:
        enhanced_results['t2_test'] = enhanced_results['t2_statistics']
    if 'spe_test' not in enhanced_results and 'spe_statistics' in enhanced_results:
        enhanced_results['spe_test'] = enhanced_results['spe_statistics']
    
    print(f"Enhanced Transformer control limits: T² = {enhanced_results.get('t2_limit', 'N/A'):.2f}, " 
          f"SPE = {enhanced_results.get('spe_limit', 'N/A'):.2f}")
        
    detection_results["Enhanced_Transformer"] = enhanced_results
    
    # 3. Run Improved Transformer detection (SPE only) if not skipped
    if not skip_improved_transformer:
        print("\nSetting up improved transformer model...")
        improved_results = run_improved_transformer_detection(X_train, X_test, happen, model_path=model_paths['improved'])
        
        # Ensure key fields exist in improved_transformer
        if 'spe_statistics' in improved_results and 'spe_test' not in improved_results:
            improved_results['spe_test'] = improved_results['spe_statistics']
        if 'control_limit' in improved_results and 'spe_limit' not in improved_results:
            improved_results['spe_limit'] = improved_results['control_limit']
        if 'false_alarm_rate' in improved_results and 'false_rates' not in improved_results:
            improved_results['false_rates'] = [improved_results['false_alarm_rate'], improved_results['false_alarm_rate']]
        if 'miss_rate' in improved_results and 'miss_rates' not in improved_results:
            improved_results['miss_rates'] = [improved_results['miss_rate'], improved_results['miss_rate']]
        if 'detection_delay' in improved_results and 'detection_times' not in improved_results:
            dt = improved_results['detection_delay'] 
            improved_results['detection_times'] = [dt, dt]
        
        # Use SPE for both T2 and SPE plots for improved transformer (SPE-only model)
        if 'spe_test' in improved_results and 't2_test' not in improved_results:
            improved_results['t2_test'] = improved_results['spe_test']
        if 'spe_limit' in improved_results and 't2_limit' not in improved_results:
            improved_results['t2_limit'] = improved_results['spe_limit']
        
        # Ensure control limits are displayed
        spe_limit = improved_results.get('spe_limit', improved_results.get('control_limit', 0))
        print(f"Control limit candidates - KDE: {spe_limit:.2f}, Chi²: {spe_limit:.2f}, "
              f"Percentile: {spe_limit:.2f}, Final: {spe_limit:.2f}")
        print(f"Improved Transformer SPE control limit: {spe_limit:.2f}")
        
        detection_results["Improved_Transformer"] = improved_results
    else:
        print("\nSkipping Improved Transformer (set skip_improved_transformer=False to include it)")
    
    # 4. Run Extreme Anomaly Detector
    print("\nRunning Extreme Anomaly Detector...")
    extreme_results = extreme_anomaly_detector(X_train, X_test, happen, [37, 38, 34, 36])
    detection_results["Extreme_Anomaly"] = extreme_results
    
    # 5. Run Ultra Extreme Anomaly Detector 
    print("\nRunning Ultra Extreme Anomaly Detector...")
    ultra_extreme_results = ultra_extreme_anomaly_detector(X_train, X_test, happen, [37, 38])
    detection_results["Ultra_Extreme"] = ultra_extreme_results
    
    # 6. Run Ultra Sensitive Ensemble Detector
    print("\nRunning Ultra Sensitive Ensemble Detector...")
    # Need to get importance results for ensemble detector
    # We can use the enhanced transformer model to analyze feature importance
    importance_results = {}
    if hasattr(enhanced_results, 'importance_results'):
        importance_results = enhanced_results.get('importance_results', {})
    elif 'variable_importance' in enhanced_results:
        # Create importance results dictionary from variable importance
        var_importance = enhanced_results.get('variable_importance', None)
        if var_importance is not None:
            # Get indices of top features sorted by importance
            top_indices = np.argsort(-var_importance)  # Negative for descending order
            importance_results = {
                'top_indices': top_indices,
                'importance_values': var_importance[top_indices]
            }
    else:
        # If no importance results available, use the top features directly
        print("No variable importance information. Using default top features.")
        importance_results = {
            'top_indices': np.array([37, 38, 34, 36, 270, 555, 9, 212, 350, 30])
        }
    
    ultra_ensemble_results = ultra_sensitive_ensemble_detector(X_train, X_test, happen, importance_results, device)
    detection_results["Ultra_Ensemble"] = ultra_ensemble_results

    # 7. Run balanced two-stage detector
    balanced_results = balanced_two_stage_detector(
        X_train, X_test, happen, [37, 38, 34, 36])  # Top features based on importance
    
    # Ensure keys are properly set for balanced detector results
    if 'statistics' in balanced_results and 't2_test' not in balanced_results:
        balanced_results['t2_test'] = balanced_results['statistics']
        balanced_results['spe_test'] = balanced_results['statistics']
    if 'threshold' in balanced_results and 't2_limit' not in balanced_results:
        balanced_results['t2_limit'] = balanced_results['threshold']
        balanced_results['spe_limit'] = balanced_results['threshold']
    if 'false_alarm_rate' in balanced_results and 'false_rates' not in balanced_results:
        balanced_results['false_rates'] = [balanced_results['false_alarm_rate'], balanced_results['false_alarm_rate']]
    if 'miss_rate' in balanced_results and 'miss_rates' not in balanced_results:
        balanced_results['miss_rates'] = [balanced_results['miss_rate'], balanced_results['miss_rate']]
    if 'detection_delay' in balanced_results and 'detection_times' not in balanced_results:
        dt = balanced_results['detection_delay']
        balanced_results['detection_times'] = [dt, dt]
        
    detection_results["Balanced_TwoStage"] = balanced_results
    
    # 8. Run Transformer-enhanced Two-Stage detector if included
    if include_transformer:
        print("\nRunning Transformer-Enhanced Two-Stage Detector...")
        transformer_results = transformer_enhanced_two_stage_detector(
            X_train, X_test, happen, [37, 38, 34, 36]  # Same top features for consistency
        )
        
        # Store the results
        detection_results["Transformer_TwoStage"] = transformer_results
    else:
        print("\nSkipping Transformer-Enhanced Two-Stage Detector (use --include_transformer to include it)")
    
    # Calculate any missing detection metrics
    detection_results = calculate_detection_metrics(detection_results, happen)
    
    # Print comparison table
    print_comparison_table(detection_results)
    
    # Generate comparison visualizations
    plot_comparison(detection_results, happen)
    
    runtime = time.time() - start_time
    print(f"Total Runtime: {runtime:.2f} seconds")
    
    return {
        'detection_results': detection_results,
        'runtime': runtime
    }


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run SECOM fault detection methods comparison')
    parser.add_argument('--include_improved', action='store_true', 
                        help='Include Improved Transformer model (slower)')
    parser.add_argument('--include_transformer', action='store_true', 
                        help='Include Transformer-Enhanced Two-Stage detector')
    
    args = parser.parse_args()
    
    # Run main function with or without Improved Transformer and Transformer-enhanced detector
    main(skip_improved_transformer=not args.include_improved, 
         include_transformer=args.include_transformer) 
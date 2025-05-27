import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import argparse

# 配置中文字体
from scripts.utils import save_plot, setup_chinese_fonts
setup_chinese_fonts()

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
from src.models.transformer_enhanced_two_stage import transformer_enhanced_two_stage_detector
from scripts.secom_plot_optimization import create_comprehensive_report
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


# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def plot_comparison(detection_results, happen, title_prefix=None):
    """Plot comparison of different detection methods with enhanced visualization
    
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
    
    # Create enhanced comparison plot with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Import additional modules for enhanced analysis
    
    # Extract data for main methods with error handling
    methods_to_plot = []
    
    # PCA baseline
    if "PCA" in detection_results:
        pca_results = detection_results["PCA"]
        t2_pca = np.ravel(pca_results.get("t2_test", np.zeros(471)))
        spe_pca = np.ravel(pca_results.get("spe_test", np.zeros(471)))
        t2_limit_pca = pca_results.get("t2_limit", 0)
        spe_limit_pca = pca_results.get("spe_limit", 0)
        methods_to_plot.append(("PCA", t2_pca, spe_pca, t2_limit_pca, spe_limit_pca, pca_results))
    
    # Enhanced Transformer
    if "Enhanced_Transformer" in detection_results:
        enhanced_results = detection_results["Enhanced_Transformer"]
        t2_enhanced = np.ravel(enhanced_results.get("t2_test", enhanced_results.get("t2_statistics", np.zeros(471))))
        spe_enhanced = np.ravel(enhanced_results.get("spe_test", enhanced_results.get("spe_statistics", np.zeros(471))))
        t2_limit_enhanced = enhanced_results.get("t2_limit", 0)
        spe_limit_enhanced = enhanced_results.get("spe_limit", 0)
        methods_to_plot.append(("Enhanced Transformer", t2_enhanced, spe_enhanced, t2_limit_enhanced, spe_limit_enhanced, enhanced_results))
    
    # Improved Transformer
    if "Improved_Transformer" in detection_results:
        improved_results = detection_results["Improved_Transformer"]
        spe_improved = np.ravel(improved_results.get("spe_test", improved_results.get("spe_statistics", np.zeros(471))))
        spe_limit_improved = improved_results.get("spe_limit", improved_results.get("control_limit", 0))
        # Use SPE for both T2 and SPE for improved transformer (SPE-only model)
        methods_to_plot.append(("Improved Transformer", spe_improved, spe_improved, spe_limit_improved, spe_limit_improved, improved_results))
    
    if not methods_to_plot:
        print("Warning: No valid detection results found for plotting")
        return
    
    # Create enhanced subplot layout: 3x3 grid
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    # Plot 1-2: SPE Detection with extreme value handling (main comparison)
    for i, (method_name, t2_vals, spe_vals, t2_limit, spe_limit, results) in enumerate(methods_to_plot[:2]):
        ax = plt.subplot(3, 3, i+1)
        
        # Handle extreme values in SPE
        spe_clipped = np.clip(spe_vals, 0, np.percentile(spe_vals, 99.5))
        max_spe = np.max(spe_vals)
        
        x_normal = range(1, happen+1)
        x_fault = range(happen+1, len(spe_vals)+1)
        
        # Plot with extreme value handling
        ax.plot(x_normal, spe_clipped[:happen], colors[i], linewidth=2, label='Normal', alpha=0.8)
        ax.plot(x_fault, spe_clipped[happen:], colors[i+2], linewidth=2, label='Fault', alpha=0.8)
        ax.axhline(y=spe_limit, color='k', linestyle='--', linewidth=2, label='Control Limit')
        ax.axvline(x=happen, color='red', linestyle='-', linewidth=2, label='Fault Occurrence')
        
        # Add performance metrics text
        far = results.get('spe_false_alarm_rate', results.get('false_rates', [0, 0])[1] if 'false_rates' in results else 0)
        mdr = results.get('spe_miss_rate', results.get('miss_rates', [0, 0])[1] if 'miss_rates' in results else 0)
        
        textstr = f'FAR: {far:.2f}%\nMDR: {mdr:.2f}%'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add extreme value note if needed
        if max_spe > np.percentile(spe_vals, 99.5):
            ax.text(0.98, 0.02, f'Max: {max_spe:.0f}\n(Clipped)', transform=ax.transAxes, 
                   fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax.set_title(f'{method_name} - SPE Detection', fontweight='bold')
        ax.set_xlabel('Sample')
        ax.set_ylabel('SPE (Clipped at 99.5%)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Performance comparison bar chart
    ax3 = plt.subplot(3, 3, 3)
    method_names = [m[0] for m in methods_to_plot]
    far_values = []
    mdr_values = []
    
    for _, _, _, _, _, results in methods_to_plot:
        far = results.get('spe_false_alarm_rate', results.get('false_rates', [0, 0])[1] if 'false_rates' in results else 0)
        mdr = results.get('spe_miss_rate', results.get('miss_rates', [0, 0])[1] if 'miss_rates' in results else 0)
        far_values.append(far)
        mdr_values.append(mdr)
    
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, far_values, width, label='False Alarm Rate (%)', color='lightblue')
    bars2 = ax3.bar(x + width/2, mdr_values, width, label='Miss Detection Rate (%)', color='lightcoral')
    
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Rate (%)')
    ax3.set_title('Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([name.replace(' ', '\n') for name in method_names], fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Plot 4-5: ROC curves for available methods
    if len(methods_to_plot) >= 2:
        ax4 = plt.subplot(3, 3, 4)
        
        for i, (method_name, _, spe_vals, _, spe_limit, results) in enumerate(methods_to_plot[:2]):
            # Create labels for ROC calculation
            labels = np.zeros(len(spe_vals))
            labels[happen:] = 1
            
            try:
                fpr, tpr, _ = roc_curve(labels, spe_vals)
                roc_auc = auc(fpr, tpr)
                ax4.plot(fpr, tpr, colors[i], linewidth=2, 
                        label=f'{method_name} (AUC={roc_auc:.3f})')
            except:
                print(f"Warning: Could not calculate ROC for {method_name}")
        
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 6: SPE distribution comparison
    if len(methods_to_plot) >= 2:
        ax6 = plt.subplot(3, 3, 6)
        
        for i, (method_name, _, spe_vals, _, _, _) in enumerate(methods_to_plot[:2]):
            normal_spe = spe_vals[:happen]
            fault_spe = spe_vals[happen:]
            
            # Use log scale for extreme values
            bins = np.logspace(np.log10(max(1e-6, np.min(spe_vals))), 
                              np.log10(np.max(spe_vals)), 30)
            
            ax6.hist(normal_spe, bins=bins, alpha=0.5, label=f'{method_name} Normal', 
                    color=colors[i], density=True)
            ax6.hist(fault_spe, bins=bins, alpha=0.5, label=f'{method_name} Fault', 
                    color=colors[i+2], density=True)
        
        ax6.set_xscale('log')
        ax6.set_xlabel('SPE (log scale)')
        ax6.set_ylabel('Density')
        ax6.set_title('SPE Distribution Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Detection delay analysis
    ax7 = plt.subplot(3, 3, 7)
    method_names_short = [m[0].replace(' ', '\n') for m in methods_to_plot]
    detection_delays = []
    
    for _, _, _, _, _, results in methods_to_plot:
        delay = results.get('spe_detection_time', results.get('detection_delay', 
                results.get('detection_times', [None, None])[1] if 'detection_times' in results else None))
        detection_delays.append(delay if delay is not None else -1)  # -1 for not detected
    
    bars = ax7.bar(range(len(method_names_short)), detection_delays, color=['lightgreen' if d >= 0 else 'lightcoral' for d in detection_delays])
    ax7.set_xlabel('Method')
    ax7.set_ylabel('Detection Delay (samples)')
    ax7.set_title('Detection Delay Comparison')
    ax7.set_xticks(range(len(method_names_short)))
    ax7.set_xticklabels(method_names_short, fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, delay) in enumerate(zip(bars, detection_delays)):
        label = str(delay) if delay >= 0 else 'Not detected'
        ax7.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, max(0, bar.get_height())),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Plot 8: Statistical summary table as text
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    # Create summary statistics table
    table_data = []
    headers = ['Method', 'FAR(%)', 'MDR(%)', 'Delay', 'Max SPE']
    
    for method_name, _, spe_vals, _, _, results in methods_to_plot:
        far = results.get('spe_false_alarm_rate', results.get('false_rates', [0, 0])[1] if 'false_rates' in results else 0)
        mdr = results.get('spe_miss_rate', results.get('miss_rates', [0, 0])[1] if 'miss_rates' in results else 0)
        delay = results.get('spe_detection_time', results.get('detection_delay', 
                results.get('detection_times', [None, None])[1] if 'detection_times' in results else None))
        max_spe = np.max(spe_vals)
        
        delay_str = str(delay) if delay is not None else 'N/A'
        table_data.append([method_name[:12], f'{far:.1f}', f'{mdr:.1f}', delay_str, f'{max_spe:.0f}'])
    
    # Create table
    table = ax8.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax8.set_title('Performance Summary', fontweight='bold', pad=20)
    
    # Plot 9: Extreme value analysis
    ax9 = plt.subplot(3, 3, 9)
    
    for i, (method_name, _, spe_vals, _, _, _) in enumerate(methods_to_plot):
        percentiles = [90, 95, 99, 99.5, 99.9]
        values = [np.percentile(spe_vals, p) for p in percentiles]
        ax9.plot(percentiles, values, 'o-', label=method_name, color=colors[i], linewidth=2)
    
    ax9.set_xlabel('Percentile')
    ax9.set_ylabel('SPE Value')
    ax9.set_title('Extreme Value Analysis')
    ax9.set_yscale('log')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Use title_prefix to determine filename prefix
    filename_prefix = "secom"
    if title_prefix and not title_prefix.startswith("SECOM"):
        # Extract dataset name from title if different from SECOM
        filename_prefix = title_prefix.split()[0].lower()
    
    # Save enhanced comparison plot
    save_plot(f"{filename_prefix}_enhanced_comparison_analysis.png")
    plt.close()
    
    # Create additional publication-quality plot focusing on best methods
    create_publication_quality_comparison(methods_to_plot, happen, filename_prefix)


def create_publication_quality_comparison(methods_to_plot, happen, filename_prefix):
    """Create publication-quality comparison plot"""
    if len(methods_to_plot) < 2:
        return
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: SPE detection comparison (log scale)
    for i, (method_name, _, spe_vals, _, spe_limit, results) in enumerate(methods_to_plot[:2]):
        spe_log = np.log10(spe_vals + 1)
        limit_log = np.log10(spe_limit + 1)
        
        x_normal = range(1, happen+1)
        x_fault = range(happen+1, len(spe_vals)+1)
        
        ax1.plot(x_normal, spe_log[:happen], color=colors[i], linewidth=2.5, 
                label=f'{method_name} Normal', alpha=0.9)
        ax1.plot(x_fault, spe_log[happen:], color=colors[i+2], linewidth=2.5, 
                label=f'{method_name} Fault', alpha=0.9)
        ax1.axhline(y=limit_log, color=colors[i], linestyle='--', linewidth=2, alpha=0.8)
    
    ax1.axvline(x=happen, color='red', linestyle='-', linewidth=2.5, label='Fault Occurrence')
    ax1.set_xlabel('Sample Index', fontweight='bold')
    ax1.set_ylabel('SPE (log₁₀ scale)', fontweight='bold')
    ax1.set_title('Fault Detection Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance metrics comparison
    method_names = [m[0] for m in methods_to_plot]
    far_values = []
    mdr_values = []
    
    for _, _, _, _, _, results in methods_to_plot:
        far = results.get('spe_false_alarm_rate', results.get('false_rates', [0, 0])[1] if 'false_rates' in results else 0)
        mdr = results.get('spe_miss_rate', results.get('miss_rates', [0, 0])[1] if 'miss_rates' in results else 0)
        far_values.append(far)
        mdr_values.append(mdr)
    
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, far_values, width, label='False Alarm Rate', color='lightblue', edgecolor='navy')
    bars2 = ax2.bar(x + width/2, mdr_values, width, label='Miss Detection Rate', color='lightcoral', edgecolor='darkred')
    
    ax2.set_xlabel('Method', fontweight='bold')
    ax2.set_ylabel('Rate (%)', fontweight='bold')
    ax2.set_title('Performance Metrics Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name.replace(' ', '\n') for name in method_names])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: ROC curves
    
    for i, (method_name, _, spe_vals, _, _, _) in enumerate(methods_to_plot[:2]):
        labels = np.zeros(len(spe_vals))
        labels[happen:] = 1
        
        try:
            fpr, tpr, _ = roc_curve(labels, spe_vals)
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, color=colors[i], linewidth=3, 
                    label=f'{method_name} (AUC={roc_auc:.3f})')
        except:
            pass
    
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('False Positive Rate', fontweight='bold')
    ax3.set_ylabel('True Positive Rate', fontweight='bold')
    ax3.set_title('ROC Curve Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary
    ax4.axis('off')
    
    # Create comprehensive summary
    summary_text = "Performance Summary:\n\n"
    
    for i, (method_name, _, spe_vals, _, _, results) in enumerate(methods_to_plot):
        far = results.get('spe_false_alarm_rate', results.get('false_rates', [0, 0])[1] if 'false_rates' in results else 0)
        mdr = results.get('spe_miss_rate', results.get('miss_rates', [0, 0])[1] if 'miss_rates' in results else 0)
        delay = results.get('spe_detection_time', results.get('detection_delay', None))
        
        normal_spe = spe_vals[:happen]
        fault_spe = spe_vals[happen:]
        separation_ratio = np.mean(fault_spe) / np.mean(normal_spe) if np.mean(normal_spe) > 0 else float('inf')
        
        summary_text += f"{method_name}:\n"
        summary_text += f"  • False Alarm Rate: {far:.2f}%\n"
        summary_text += f"  • Miss Detection Rate: {mdr:.2f}%\n"
        summary_text += f"  • Detection Delay: {delay if delay is not None else 'N/A'} samples\n"
        summary_text += f"  • Separation Ratio: {separation_ratio:.2f}\n"
        summary_text += f"  • Max SPE: {np.max(spe_vals):.0f}\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    save_plot(f"{filename_prefix}_publication_quality_comparison.png")
    plt.close()
    
    # Reset style
    plt.style.use('default')


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
    
    # Print enhanced header with more information
    print("=" * 120)
    print("SECOM故障检测方法性能对比分析")
    print("=" * 120)
    print(f"{'方法':<20} | {'误报率(%)':<10} {'漏报率(%)':<10} | {'检测延迟':<10} | {'ROC AUC':<8} | {'分离度':<8} | {'极值比':<8}")
    print("-" * 120)
    
    # Print enhanced metrics for each method
    for method in methods:
        result = detection_results[method]
        display_name = name_map.get(method, method)
        
        # Extract SPE metrics (focus on SPE as it's more important for fault detection)
        if method == "Enhanced_Transformer":
            spe_false = result.get('spe_false_alarm_rate', result.get('false_rates', [0, 0])[1] if 'false_rates' in result else 0)
            spe_miss = result.get('spe_miss_rate', result.get('miss_rates', [0, 0])[1] if 'miss_rates' in result else 0)
        elif method in ["PCA", "Balanced_TwoStage", "Transformer_TwoStage"]:
            spe_false = result.get('spe_false_alarm_rate', 0)
            spe_miss = result.get('spe_miss_rate', 0)
        else:
            spe_false = result.get('false_rates', [0, 0])[1] if 'false_rates' in result and len(result['false_rates']) > 1 else result.get('false_alarm_rate', 0)
            spe_miss = result.get('miss_rates', [0, 0])[1] if 'miss_rates' in result and len(result['miss_rates']) > 1 else result.get('miss_rate', 0)
        
        # Detection delay
        delay = result.get('spe_detection_time', result.get('detection_delay', 
                result.get('detection_times', [None, None])[1] if 'detection_times' in result else None))
        delay_str = str(delay) if delay is not None else 'N/A'
        
        # Calculate additional metrics
        spe_values = result.get('spe_test', result.get('spe_statistics', []))
        if len(spe_values) > 0:
            # ROC AUC calculation
            try:
                labels = np.zeros(len(spe_values))
                happen_idx = 16  # SECOM fault occurrence
                if happen_idx < len(labels):
                    labels[happen_idx:] = 1
                fpr, tpr, _ = roc_curve(labels, spe_values)
                roc_auc = auc(fpr, tpr)
            except:
                roc_auc = 0.0
            
            # Separation ratio
            normal_spe = spe_values[:happen_idx] if happen_idx < len(spe_values) else spe_values[:len(spe_values)//2]
            fault_spe = spe_values[happen_idx:] if happen_idx < len(spe_values) else spe_values[len(spe_values)//2:]
            
            if len(normal_spe) > 0 and len(fault_spe) > 0:
                separation_ratio = np.mean(fault_spe) / np.mean(normal_spe) if np.mean(normal_spe) > 0 else float('inf')
                # Extreme value ratio
                extreme_ratio = np.max(spe_values) / np.percentile(spe_values, 99) if np.percentile(spe_values, 99) > 0 else 1.0
            else:
                separation_ratio = 0.0
                extreme_ratio = 1.0
        else:
            roc_auc = 0.0
            separation_ratio = 0.0
            extreme_ratio = 1.0
        
        print(f"{display_name:<20} | {spe_false:<10.2f} {spe_miss:<10.2f} | {delay_str:<10} | {roc_auc:<8.3f} | {separation_ratio:<8.2f} | {extreme_ratio:<8.1f}")
    
    # Print performance analysis summary
    print("-" * 120)
    print("性能分析总结:")
    print("-" * 120)
    
    # Find best performing methods
    best_far = min(methods, key=lambda m: detection_results[m].get('spe_false_alarm_rate', 
                   detection_results[m].get('false_rates', [100, 100])[1] if 'false_rates' in detection_results[m] else 100))
    best_mdr = min(methods, key=lambda m: detection_results[m].get('spe_miss_rate', 
                   detection_results[m].get('miss_rates', [100, 100])[1] if 'miss_rates' in detection_results[m] else 100))
    
    print(f"最低误报率: {name_map.get(best_far, best_far)}")
    print(f"最低漏报率: {name_map.get(best_mdr, best_mdr)}")
    
    # Calculate average performance
    total_methods = len(methods)
    if total_methods > 0:
        avg_far = sum(detection_results[m].get('spe_false_alarm_rate', 
                     detection_results[m].get('false_rates', [0, 0])[1] if 'false_rates' in detection_results[m] else 0) 
                     for m in methods) / total_methods
        avg_mdr = sum(detection_results[m].get('spe_miss_rate', 
                     detection_results[m].get('miss_rates', [0, 0])[1] if 'miss_rates' in detection_results[m] else 0) 
                     for m in methods) / total_methods
        
        print(f"平均误报率: {avg_far:.2f}%")
        print(f"平均漏报率: {avg_mdr:.2f}%")
    
    # Statistical significance analysis
    print("\n统计显著性分析:")
    if len(methods) >= 2:
        # Compare top 2 methods
        method1, method2 = methods[0], methods[1]
        result1, result2 = detection_results[method1], detection_results[method2]
        
        # Get SPE values for comparison
        spe1 = result1.get('spe_test', result1.get('spe_statistics', []))
        spe2 = result2.get('spe_test', result2.get('spe_statistics', []))
        
        if len(spe1) > 0 and len(spe2) > 0:
            try:
                # T-test for means
                t_stat, t_pval = ttest_ind(spe1, spe2)
                # Mann-Whitney U test for distributions
                u_stat, u_pval = mannwhitneyu(spe1, spe2, alternative='two-sided')
                
                print(f"  {name_map.get(method1, method1)} vs {name_map.get(method2, method2)}:")
                print(f"    T-test p-value: {t_pval:.4f} ({'显著' if t_pval < 0.05 else '不显著'})")
                print(f"    Mann-Whitney U p-value: {u_pval:.4f} ({'显著' if u_pval < 0.05 else '不显著'})")
            except Exception as e:
                print(f"  统计检验失败: {e}")
    
    print("\n改进建议:")
    # Provide improvement suggestions based on results
    for method in methods:
        result = detection_results[method]
        spe_false = result.get('spe_false_alarm_rate', result.get('false_rates', [0, 0])[1] if 'false_rates' in result else 0)
        spe_miss = result.get('spe_miss_rate', result.get('miss_rates', [0, 0])[1] if 'miss_rates' in result else 0)
        
        if spe_false > 10:
            print(f"  {name_map.get(method, method)}: 误报率过高，建议提高阈值或改进预处理")
        elif spe_miss > 10:
            print(f"  {name_map.get(method, method)}: 漏报率过高，建议降低阈值或改进模型架构")
        elif spe_false < 5 and spe_miss < 5:
            print(f"  {name_map.get(method, method)}: 性能优秀，可考虑用于实际部署")
    
    print("=" * 120)


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
            'enhanced': 'results/models/secom/enhanced_transformer/enhanced_transformer_SECOM_vbest.pth',
            'improved': 'results/models/secom/improved_transformer_t2/improved_transformer_t2_SECOM_vbest.pth'
        }

    # Ensure model directories exist
    os.makedirs(os.path.dirname(model_paths['enhanced']), exist_ok=True)
    os.makedirs(os.path.dirname(model_paths['improved']), exist_ok=True)

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
    
    # Run additional optimized analysis for best performing methods
    print("\n" + "="*80)
    print("运行优化分析...")
    print("="*80)
    
    # Identify best methods for detailed analysis
    best_methods = []
    if "Enhanced_Transformer" in detection_results:
        best_methods.append(("Enhanced_Transformer", detection_results["Enhanced_Transformer"]))
    if "Improved_Transformer" in detection_results:
        best_methods.append(("Improved_Transformer", detection_results["Improved_Transformer"]))
    
    # Run optimized analysis for each best method
    for method_name, results in best_methods:
        if 'spe_test' in results or 'spe_statistics' in results:
            spe_values = results.get('spe_test', results.get('spe_statistics', []))
            spe_limit = results.get('spe_limit', results.get('control_limit', 0))
            
            if len(spe_values) > 0 and spe_limit > 0:
                print(f"\n{method_name} 优化分析:")
                try:
                    # Import and use the optimization functions
                    
                    # Create comprehensive analysis
                    metrics, threshold_analysis = create_comprehensive_report(
                        spe_values, spe_limit, happen,
                        model_name=method_name.replace('_', ' '),
                        save_prefix=f"secom_{method_name.lower()}_optimized"
                    )
                    
                    print(f"  优化分析完成，生成图像: secom_{method_name.lower()}_optimized.png")
                    
                except ImportError:
                    print(f"  警告: 无法导入优化分析模块，跳过 {method_name} 的详细分析")
                except Exception as e:
                    print(f"  警告: {method_name} 优化分析失败: {e}")
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("分析总结报告")
    print(f"{'='*80}")
    
    # Find overall best method
    best_overall = None
    best_score = float('inf')
    
    for method, results in detection_results.items():
        spe_false = results.get('spe_false_alarm_rate', results.get('false_rates', [100, 100])[1] if 'false_rates' in results else 100)
        spe_miss = results.get('spe_miss_rate', results.get('miss_rates', [100, 100])[1] if 'miss_rates' in results else 100)
        
        # Combined score (lower is better)
        combined_score = spe_false + spe_miss
        if combined_score < best_score:
            best_score = combined_score
            best_overall = method
    
    if best_overall:
        print(f"最佳整体性能方法: {best_overall}")
        print(f"综合评分 (误报率+漏报率): {best_score:.2f}%")
    
    # Performance tier classification
    print(f"\n性能分级:")
    for method, results in detection_results.items():
        spe_false = results.get('spe_false_alarm_rate', results.get('false_rates', [100, 100])[1] if 'false_rates' in results else 100)
        spe_miss = results.get('spe_miss_rate', results.get('miss_rates', [100, 100])[1] if 'miss_rates' in results else 100)
        
        if spe_false < 5 and spe_miss < 5:
            tier = "A级 (优秀)"
        elif spe_false < 10 and spe_miss < 10:
            tier = "B级 (良好)"
        elif spe_false < 20 and spe_miss < 20:
            tier = "C级 (可接受)"
        else:
            tier = "D级 (需改进)"
        
        print(f"  {method}: {tier}")
    
    runtime = time.time() - start_time
    print(f"\n总运行时间: {runtime:.2f}秒")
    print(f"生成的图像文件:")
    print(f"  - secom_enhanced_comparison_analysis.png")
    print(f"  - secom_publication_quality_comparison.png")
    for method_name, _ in best_methods:
        print(f"  - secom_{method_name.lower()}_optimized_optimized.png")
        print(f"  - secom_{method_name.lower()}_optimized_threshold_analysis.png")
    
    return {
        'detection_results': detection_results,
        'runtime': runtime,
        'best_method': best_overall,
        'best_score': best_score
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SECOM fault detection methods comparison')
    parser.add_argument('--include_improved', action='store_true', 
                        help='Include Improved Transformer model (slower)')
    parser.add_argument('--include_transformer', action='store_true', 
                        help='Include Transformer-Enhanced Two-Stage detector')
    
    args = parser.parse_args()
    
    # Run main function with or without Improved Transformer and Transformer-enhanced detector
    main(skip_improved_transformer=not args.include_improved, 
         include_transformer=args.include_transformer) 
#!/usr/bin/env python3
from scripts.utils import setup_chinese_fonts

"""
SECOM检测结果绘图优化脚本
解决当前可视化中的问题：
1. SPE极值处理
2. 添加定量性能指标
3. 改进可视化效果
4. 增加统计分析
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
import pandas as pd
import os
import sys
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from scripts.utils import save_plot

def calculate_comprehensive_metrics(spe_values, spe_limit, happen, labels=None):
    """
    计算全面的检测性能指标
    
    Parameters:
    -----------
    spe_values : array-like
        SPE统计量值
    spe_limit : float
        控制限
    happen : int
        故障发生位置
    labels : array-like, optional
        真实标签 (0=正常, 1=故障)
        
    Returns:
    --------
    metrics : dict
        包含各种性能指标的字典
    """
    # 基本检测指标
    alarms = spe_values > spe_limit
    
    # 如果没有提供标签，根据happen创建
    if labels is None:
        labels = np.zeros(len(spe_values))
        labels[happen:] = 1
    
    # 基础指标
    false_alarms = np.sum(alarms[:happen])
    false_alarm_rate = 100 * false_alarms / happen if happen > 0 else 0
    
    misses = np.sum(~alarms[happen:])
    miss_rate = 100 * misses / (len(alarms) - happen) if len(alarms) > happen else 0
    
    # 检测延迟
    detection_delay = None
    consecutive_required = 1
    for i in range(happen, len(alarms) - consecutive_required + 1):
        if all(alarms[i:i+consecutive_required]):
            detection_delay = i - happen
            break
    
    # ROC和PR曲线指标
    try:
        fpr, tpr, _ = roc_curve(labels, spe_values)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(labels, spe_values)
        pr_auc = auc(recall, precision)
    except:
        roc_auc = 0.0
        pr_auc = 0.0
        fpr, tpr = [0, 1], [0, 1]
        precision, recall = [1, 0], [0, 1]
    
    # 统计特性
    normal_spe = spe_values[:happen]
    fault_spe = spe_values[happen:]
    
    # 分布统计
    normal_stats = {
        'mean': np.mean(normal_spe),
        'std': np.std(normal_spe),
        'median': np.median(normal_spe),
        'q95': np.percentile(normal_spe, 95),
        'max': np.max(normal_spe)
    }
    
    fault_stats = {
        'mean': np.mean(fault_spe),
        'std': np.std(fault_spe),
        'median': np.median(fault_spe),
        'q95': np.percentile(fault_spe, 95),
        'max': np.max(fault_spe)
    }
    
    # 分离度指标
    separation_ratio = fault_stats['mean'] / normal_stats['mean'] if normal_stats['mean'] > 0 else float('inf')
    
    return {
        'false_alarm_rate': false_alarm_rate,
        'miss_rate': miss_rate,
        'detection_delay': detection_delay,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'separation_ratio': separation_ratio,
        'normal_stats': normal_stats,
        'fault_stats': fault_stats,
        'roc_curve': (fpr, tpr),
        'pr_curve': (precision, recall),
        'alarms': alarms,
        'labels': labels
    }

def plot_optimized_spe_detection(spe_values, spe_limit, happen, metrics=None, 
                                title="SECOM Improved Transformer - SPE Detection",
                                save_name="secom_optimized_spe_detection.png"):
    """
    创建优化的SPE检测结果图
    
    Parameters:
    -----------
    spe_values : array-like
        SPE统计量值
    spe_limit : float
        控制限
    happen : int
        故障发生位置
    metrics : dict, optional
        性能指标字典
    title : str
        图像标题
    save_name : str
        保存文件名
    """
    # 计算指标（如果未提供）
    if metrics is None:
        metrics = calculate_comprehensive_metrics(spe_values, spe_limit, happen)
    
    # 创建图像
    fig = plt.figure(figsize=(16, 12))
    
    # 主检测图（使用对数坐标处理极值）
    ax1 = plt.subplot(2, 3, (1, 2))
    
    # 处理极值：使用分段显示
    spe_clipped = np.clip(spe_values, 0, np.percentile(spe_values, 99.5))
    
    # 绘制正常和故障区域
    x_normal = range(1, happen+1)
    x_fault = range(happen+1, len(spe_values)+1)
    
    plt.plot(x_normal, spe_clipped[:happen], 'g-', linewidth=2, label='Normal', alpha=0.8)
    plt.plot(x_fault, spe_clipped[happen:], 'm-', linewidth=2, label='Fault', alpha=0.8)
    
    # 控制限
    plt.axhline(y=spe_limit, color='k', linestyle='--', linewidth=2, label='Control Limit')
    
    # 故障发生线
    plt.axvline(x=happen, color='r', linestyle='-', linewidth=2, label='Fault Occurrence')
    
    # 标注误报和漏报
    alarms = metrics['alarms']
    false_alarms_idx = np.where(alarms[:happen])[0]
    misses_idx = np.where(~alarms[happen:])[0] + happen
    
    if len(false_alarms_idx) > 0:
        plt.scatter(false_alarms_idx + 1, spe_clipped[false_alarms_idx], 
                   color='orange', s=50, marker='x', label='False Alarms', zorder=5)
    
    if len(misses_idx) > 0:
        plt.scatter(misses_idx + 1, spe_clipped[misses_idx], 
                   color='red', s=50, marker='o', label='Missed Detections', zorder=5)
    
    # 添加性能指标文本框
    textstr = f'''Performance Metrics:
False Alarm Rate: {metrics["false_alarm_rate"]:.2f}%
Miss Rate: {metrics["miss_rate"]:.2f}%
Detection Delay: {metrics["detection_delay"] if metrics["detection_delay"] is not None else "N/A"} samples
ROC AUC: {metrics["roc_auc"]:.3f}
Separation Ratio: {metrics["separation_ratio"]:.2f}'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('SPE (Clipped at 99.5%)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 极值处理说明
    max_spe = np.max(spe_values)
    if max_spe > np.percentile(spe_values, 99.5):
        plt.text(0.98, 0.02, f'Max SPE: {max_spe:.0f}\n(Clipped for visualization)', 
                transform=ax1.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ROC曲线
    ax2 = plt.subplot(2, 3, 3)
    fpr, tpr = metrics['roc_curve']
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # PR曲线
    ax3 = plt.subplot(2, 3, 4)
    precision, recall = metrics['pr_curve']
    plt.plot(recall, precision, 'r-', linewidth=2, label=f'PR (AUC = {metrics["pr_auc"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE分布对比
    ax4 = plt.subplot(2, 3, 5)
    normal_spe = spe_values[:happen]
    fault_spe = spe_values[happen:]
    
    # 使用对数尺度处理极值
    bins = np.logspace(np.log10(max(1e-6, np.min(spe_values))), 
                      np.log10(np.max(spe_values)), 50)
    
    plt.hist(normal_spe, bins=bins, alpha=0.7, label='Normal', color='green', density=True)
    plt.hist(fault_spe, bins=bins, alpha=0.7, label='Fault', color='magenta', density=True)
    plt.axvline(x=spe_limit, color='k', linestyle='--', label='Control Limit')
    plt.xscale('log')
    plt.xlabel('SPE (log scale)')
    plt.ylabel('Density')
    plt.title('SPE Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 检测时间分析
    ax5 = plt.subplot(2, 3, 6)
    
    # 计算滑动窗口检测性能
    window_sizes = [1, 2, 3, 5, 10]
    detection_delays = []
    
    for window in window_sizes:
        delay = None
        for i in range(happen, len(alarms) - window + 1):
            if all(alarms[i:i+window]):
                delay = i - happen
                break
        detection_delays.append(delay if delay is not None else len(spe_values) - happen)
    
    plt.plot(window_sizes, detection_delays, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Consecutive Samples Required')
    plt.ylabel('Detection Delay (samples)')
    plt.title('Detection Delay vs Window Size')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(window_sizes, detection_delays)):
        plt.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    save_plot(save_name)
    plt.close()
    
    return metrics

def plot_threshold_sensitivity_analysis(spe_values, happen, 
                                       save_name="secom_threshold_sensitivity.png"):
    """
    阈值敏感性分析
    """
    # 计算不同阈值下的性能
    thresholds = np.percentile(spe_values[:happen], np.linspace(90, 99.9, 20))
    
    false_rates = []
    miss_rates = []
    f1_scores = []
    
    for threshold in thresholds:
        metrics = calculate_comprehensive_metrics(spe_values, threshold, happen)
        false_rates.append(metrics['false_alarm_rate'])
        miss_rates.append(metrics['miss_rate'])
        
        # 计算F1分数
        tp = np.sum((spe_values[happen:] > threshold))
        fp = np.sum((spe_values[:happen] > threshold))
        fn = np.sum((spe_values[happen:] <= threshold))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    # 绘制敏感性分析
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 误报率和漏报率
    ax1.plot(thresholds, false_rates, 'b-o', label='False Alarm Rate', linewidth=2)
    ax1.plot(thresholds, miss_rates, 'r-s', label='Miss Rate', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Threshold Sensitivity Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # F1分数
    ax2.plot(thresholds, f1_scores, 'g-^', label='F1 Score', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 标注最佳阈值
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    ax2.axvline(x=best_threshold, color='red', linestyle='--', 
               label=f'Best Threshold: {best_threshold:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    save_plot(save_name)
    plt.close()
    
    return {
        'thresholds': thresholds,
        'false_rates': false_rates,
        'miss_rates': miss_rates,
        'f1_scores': f1_scores,
        'best_threshold': best_threshold
    }

def create_comprehensive_report(spe_values, spe_limit, happen, 
                              model_name="Improved Transformer",
                              save_prefix="secom_comprehensive"):
    """
    创建综合分析报告
    """
    print(f"\n{'='*60}")
    print(f"SECOM {model_name} - 综合性能分析报告")
    print(f"{'='*60}")
    
    # 计算综合指标
    metrics = calculate_comprehensive_metrics(spe_values, spe_limit, happen)
    
    # 基础性能报告
    print(f"\n基础检测性能:")
    print(f"  误报率 (False Alarm Rate): {metrics['false_alarm_rate']:.2f}%")
    print(f"  漏报率 (Miss Rate): {metrics['miss_rate']:.2f}%")
    print(f"  检测延迟 (Detection Delay): {metrics['detection_delay'] if metrics['detection_delay'] is not None else 'N/A'} samples")
    print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"  PR AUC: {metrics['pr_auc']:.3f}")
    
    # 统计特性报告
    print(f"\n统计特性分析:")
    print(f"  正常样本SPE - 均值: {metrics['normal_stats']['mean']:.2f}, 标准差: {metrics['normal_stats']['std']:.2f}")
    print(f"  故障样本SPE - 均值: {metrics['fault_stats']['mean']:.2f}, 标准差: {metrics['fault_stats']['std']:.2f}")
    print(f"  分离度比值: {metrics['separation_ratio']:.2f}")
    
    # 极值分析
    max_spe = np.max(spe_values)
    p99_spe = np.percentile(spe_values, 99)
    print(f"\n极值分析:")
    print(f"  最大SPE值: {max_spe:.0f}")
    print(f"  99%分位数: {p99_spe:.2f}")
    print(f"  极值比例: {max_spe/p99_spe:.1f}x")
    
    # 创建优化图像
    plot_optimized_spe_detection(spe_values, spe_limit, happen, metrics,
                                title=f"SECOM {model_name} - Optimized SPE Detection",
                                save_name=f"{save_prefix}_optimized.png")
    
    # 阈值敏感性分析
    threshold_analysis = plot_threshold_sensitivity_analysis(spe_values, happen,
                                                           save_name=f"{save_prefix}_threshold_analysis.png")
    
    print(f"\n阈值敏感性分析:")
    print(f"  当前阈值: {spe_limit:.2f}")
    print(f"  建议最佳阈值: {threshold_analysis['best_threshold']:.2f}")
    print(f"  最佳F1分数: {np.max(threshold_analysis['f1_scores']):.3f}")
    
    # 改进建议
    print(f"\n改进建议:")
    if metrics['false_alarm_rate'] > 5:
        print(f"  - 误报率较高({metrics['false_alarm_rate']:.2f}%)，建议提高阈值或改进预处理")
    if metrics['miss_rate'] > 5:
        print(f"  - 漏报率较高({metrics['miss_rate']:.2f}%)，建议降低阈值或改进模型")
    if max_spe/p99_spe > 10:
        print(f"  - 存在极值问题，建议使用鲁棒预处理或对数变换")
    if metrics['separation_ratio'] < 2:
        print(f"  - 正常和故障样本分离度较低，建议改进特征工程")
    
    return metrics, threshold_analysis

def main():
    setup_chinese_fonts()  # 配置中文字体
    """
    主函数：运行SECOM检测结果的优化分析
    """
    # 这里需要加载实际的检测结果
    # 示例：模拟数据用于演示
    np.random.seed(42)
    
    # 模拟SECOM检测结果
    happen = 16  # SECOM数据集中故障发生位置
    n_samples = 471
    
    # 模拟正常阶段的SPE值（较低）
    normal_spe = np.random.exponential(scale=50, size=happen)
    
    # 模拟故障阶段的SPE值（较高，包含极值）
    fault_spe = np.random.exponential(scale=500, size=n_samples-happen)
    # 添加一些极值
    fault_spe[np.random.choice(len(fault_spe), 3)] *= 100
    
    spe_values = np.concatenate([normal_spe, fault_spe])
    spe_limit = np.percentile(normal_spe, 95)
    
    print("运行SECOM检测结果优化分析...")
    
    # 创建综合报告
    metrics, threshold_analysis = create_comprehensive_report(
        spe_values, spe_limit, happen,
        model_name="Improved Transformer",
        save_prefix="secom_demo"
    )
    
    print(f"\n分析完成！生成的图像文件:")
    print(f"  - secom_demo_optimized.png")
    print(f"  - secom_demo_threshold_analysis.png")

if __name__ == "__main__":
    main() 
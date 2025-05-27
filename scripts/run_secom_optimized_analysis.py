#!/usr/bin/env python3
"""
SECOM检测结果优化分析集成脚本
使用实际的检测结果进行优化分析和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from scripts.run_secom_fault_detection import (
    load_secom_data, 
    run_improved_transformer_detection,
    run_enhanced_transformer_detection
)
from scripts.secom_plot_optimization import (
    create_comprehensive_report,
    plot_optimized_spe_detection,
    plot_threshold_sensitivity_analysis,
    calculate_comprehensive_metrics
)
from scripts.utils import save_plot

def run_optimized_secom_analysis():
    """
    运行SECOM数据集的优化分析
    """
    print("="*80)
    print("SECOM检测结果优化分析")
    print("="*80)
    
    # 1. 加载SECOM数据
    print("\n1. 加载SECOM数据...")
    X_train, X_test, happen, y_test, normal_indices, fault_indices = load_secom_data()
    print(f"   训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")
    print(f"   特征维度: {X_train.shape[1]}, 故障发生位置: {happen}")
    
    # 2. 运行改进的Transformer检测
    print("\n2. 运行改进的Transformer检测...")
    try:
        improved_results = run_improved_transformer_detection(X_train, X_test, happen)
        
        # 提取结果
        spe_values = improved_results['spe_test']
        spe_limit = improved_results['spe_limit']
        
        print(f"   SPE控制限: {spe_limit:.2f}")
        print(f"   SPE值范围: {np.min(spe_values):.2f} - {np.max(spe_values):.0f}")
        
        # 3. 创建优化分析报告
        print("\n3. 创建优化分析报告...")
        metrics, threshold_analysis = create_comprehensive_report(
            spe_values, spe_limit, happen,
            model_name="Improved Transformer",
            save_prefix="secom_improved_transformer"
        )
        
        # 4. 额外的对比分析
        print("\n4. 进行对比分析...")
        
        # 运行增强Transformer进行对比
        try:
            enhanced_results = run_enhanced_transformer_detection(X_train, X_test, happen)
            enhanced_spe = enhanced_results['spe_test']
            enhanced_limit = enhanced_results['spe_limit']
            
            # 对比分析
            create_comparison_analysis(
                spe_values, spe_limit, enhanced_spe, enhanced_limit, happen
            )
            
        except Exception as e:
            print(f"   增强Transformer分析失败: {e}")
            print("   继续使用改进Transformer的结果...")
        
        # 5. 生成论文质量的图像
        print("\n5. 生成论文质量的图像...")
        create_publication_quality_plots(spe_values, spe_limit, happen, metrics)
        
        # 6. 输出改进建议
        print("\n6. 输出改进建议...")
        provide_improvement_suggestions(metrics, threshold_analysis, spe_values, spe_limit)
        
        return {
            'spe_values': spe_values,
            'spe_limit': spe_limit,
            'metrics': metrics,
            'threshold_analysis': threshold_analysis
        }
        
    except Exception as e:
        print(f"检测分析失败: {e}")
        print("使用模拟数据进行演示...")
        return run_demo_analysis(happen)

def create_comparison_analysis(spe1, limit1, spe2, limit2, happen):
    """
    创建两种方法的对比分析
    """
    print("   创建方法对比分析...")
    
    # 计算两种方法的指标
    metrics1 = calculate_comprehensive_metrics(spe1, limit1, happen)
    metrics2 = calculate_comprehensive_metrics(spe2, limit2, happen)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 方法1的SPE图
    ax1 = axes[0, 0]
    x_normal = range(1, happen+1)
    x_fault = range(happen+1, len(spe1)+1)
    
    # 处理极值
    spe1_clipped = np.clip(spe1, 0, np.percentile(spe1, 99.5))
    
    ax1.plot(x_normal, spe1_clipped[:happen], 'g-', linewidth=2, label='Normal', alpha=0.8)
    ax1.plot(x_fault, spe1_clipped[happen:], 'm-', linewidth=2, label='Fault', alpha=0.8)
    ax1.axhline(y=limit1, color='k', linestyle='--', linewidth=2, label='Control Limit')
    ax1.axvline(x=happen, color='r', linestyle='-', linewidth=2, label='Fault Occurrence')
    
    ax1.set_title('Improved Transformer SPE Detection', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('SPE (Clipped)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加性能指标
    textstr1 = f'''FAR: {metrics1["false_alarm_rate"]:.2f}%
MDR: {metrics1["miss_rate"]:.2f}%
Delay: {metrics1["detection_delay"] if metrics1["detection_delay"] is not None else "N/A"}
AUC: {metrics1["roc_auc"]:.3f}'''
    ax1.text(0.02, 0.98, textstr1, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 方法2的SPE图
    ax2 = axes[0, 1]
    spe2_clipped = np.clip(spe2, 0, np.percentile(spe2, 99.5))
    
    ax2.plot(x_normal, spe2_clipped[:happen], 'g-', linewidth=2, label='Normal', alpha=0.8)
    ax2.plot(x_fault, spe2_clipped[happen:], 'm-', linewidth=2, label='Fault', alpha=0.8)
    ax2.axhline(y=limit2, color='k', linestyle='--', linewidth=2, label='Control Limit')
    ax2.axvline(x=happen, color='r', linestyle='-', linewidth=2, label='Fault Occurrence')
    
    ax2.set_title('Enhanced Transformer SPE Detection', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('SPE (Clipped)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加性能指标
    textstr2 = f'''FAR: {metrics2["false_alarm_rate"]:.2f}%
MDR: {metrics2["miss_rate"]:.2f}%
Delay: {metrics2["detection_delay"] if metrics2["detection_delay"] is not None else "N/A"}
AUC: {metrics2["roc_auc"]:.3f}'''
    ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ROC曲线对比
    ax3 = axes[1, 0]
    fpr1, tpr1 = metrics1['roc_curve']
    fpr2, tpr2 = metrics2['roc_curve']
    
    ax3.plot(fpr1, tpr1, 'b-', linewidth=2, label=f'Improved (AUC={metrics1["roc_auc"]:.3f})')
    ax3.plot(fpr2, tpr2, 'r-', linewidth=2, label=f'Enhanced (AUC={metrics2["roc_auc"]:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 性能指标对比
    ax4 = axes[1, 1]
    methods = ['Improved\nTransformer', 'Enhanced\nTransformer']
    far_values = [metrics1['false_alarm_rate'], metrics2['false_alarm_rate']]
    mdr_values = [metrics1['miss_rate'], metrics2['miss_rate']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, far_values, width, label='False Alarm Rate (%)', color='lightblue')
    bars2 = ax4.bar(x + width/2, mdr_values, width, label='Miss Detection Rate (%)', color='lightcoral')
    
    ax4.set_xlabel('Method')
    ax4.set_ylabel('Rate (%)')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot("secom_methods_comparison.png")
    plt.close()
    
    # 输出对比结果
    print(f"   改进Transformer: FAR={metrics1['false_alarm_rate']:.2f}%, MDR={metrics1['miss_rate']:.2f}%")
    print(f"   增强Transformer: FAR={metrics2['false_alarm_rate']:.2f}%, MDR={metrics2['miss_rate']:.2f}%")

def create_publication_quality_plots(spe_values, spe_limit, happen, metrics):
    """
    创建论文发表质量的图像
    """
    print("   创建论文质量图像...")
    
    # 设置论文风格
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
    
    # 主检测图（论文风格）
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 处理极值并使用更好的可视化
    spe_log = np.log10(spe_values + 1)  # 对数变换
    limit_log = np.log10(spe_limit + 1)
    
    x_normal = range(1, happen+1)
    x_fault = range(happen+1, len(spe_values)+1)
    
    # 绘制数据
    line1 = ax.plot(x_normal, spe_log[:happen], 'g-', linewidth=2.5, 
                   label='Normal Operation', alpha=0.9)
    line2 = ax.plot(x_fault, spe_log[happen:], color='#FF6B6B', linewidth=2.5, 
                   label='Fault Condition', alpha=0.9)
    
    # 控制限
    ax.axhline(y=limit_log, color='black', linestyle='--', linewidth=2, 
              label='Control Limit', alpha=0.8)
    
    # 故障发生线
    ax.axvline(x=happen, color='red', linestyle='-', linewidth=2.5, 
              label='Fault Occurrence', alpha=0.8)
    
    # 标注检测点
    alarms = metrics['alarms']
    first_detection = None
    for i in range(happen, len(alarms)):
        if alarms[i]:
            first_detection = i
            break
    
    if first_detection is not None:
        ax.annotate('First Detection', 
                   xy=(first_detection+1, spe_log[first_detection]), 
                   xytext=(first_detection+50, spe_log[first_detection]+0.5),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, color='red', fontweight='bold')
    
    # 设置标签和标题
    ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('SPE (log₁₀ scale)', fontsize=14, fontweight='bold')
    ax.set_title('SECOM Fault Detection using Improved Transformer Autoencoder', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 图例
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 性能指标框
    textstr = f'''Performance Metrics:
• False Alarm Rate: {metrics["false_alarm_rate"]:.2f}%
• Miss Detection Rate: {metrics["miss_rate"]:.2f}%
• Detection Delay: {metrics["detection_delay"] if metrics["detection_delay"] is not None else "N/A"} samples
• ROC AUC: {metrics["roc_auc"]:.3f}
• Separation Ratio: {metrics["separation_ratio"]:.2f}'''
    
    props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # 调整布局
    plt.tight_layout()
    save_plot("secom_publication_quality.png", dpi=300)
    plt.close()
    
    # 恢复默认样式
    plt.style.use('default')

def provide_improvement_suggestions(metrics, threshold_analysis, spe_values, spe_limit):
    """
    提供具体的改进建议
    """
    print("\n" + "="*60)
    print("改进建议和论文发表指导")
    print("="*60)
    
    # 性能评估
    far = metrics['false_alarm_rate']
    mdr = metrics['miss_rate']
    auc = metrics['roc_auc']
    
    print(f"\n当前性能评估:")
    print(f"  误报率: {far:.2f}% ({'优秀' if far < 2 else '良好' if far < 5 else '需改进'})")
    print(f"  漏报率: {mdr:.2f}% ({'优秀' if mdr < 2 else '良好' if mdr < 5 else '需改进'})")
    print(f"  ROC AUC: {auc:.3f} ({'优秀' if auc > 0.95 else '良好' if auc > 0.9 else '需改进'})")
    
    # 具体改进建议
    print(f"\n具体改进建议:")
    
    if far > 5:
        print(f"  1. 误报率过高 ({far:.2f}%):")
        print(f"     - 建议提高控制限阈值")
        print(f"     - 考虑使用更鲁棒的数据预处理")
        print(f"     - 增加训练数据量")
    
    if mdr > 5:
        print(f"  2. 漏报率过高 ({mdr:.2f}%):")
        print(f"     - 建议降低控制限阈值")
        print(f"     - 改进模型架构或训练策略")
        print(f"     - 考虑集成学习方法")
    
    # 极值处理建议
    max_spe = np.max(spe_values)
    p99_spe = np.percentile(spe_values, 99)
    if max_spe / p99_spe > 10:
        print(f"  3. 极值问题严重 (最大值是99%分位数的{max_spe/p99_spe:.1f}倍):")
        print(f"     - 使用对数变换: log(SPE + 1)")
        print(f"     - 采用鲁棒标准化: RobustScaler")
        print(f"     - 考虑异常值检测和处理")
    
    # 论文发表建议
    print(f"\n论文发表建议:")
    print(f"  1. 强调的优势:")
    if far < 5 and mdr < 5:
        print(f"     - 同时实现低误报率和低漏报率")
    if metrics['detection_delay'] is not None and metrics['detection_delay'] < 3:
        print(f"     - 快速检测能力 ({metrics['detection_delay']}样本延迟)")
    if auc > 0.9:
        print(f"     - 优秀的分类性能 (AUC={auc:.3f})")
    
    print(f"  2. 需要补充的实验:")
    print(f"     - 与更多基线方法对比 (PCA, ICA, OCSVM等)")
    print(f"     - 消融实验 (移除注意力机制、不同层数等)")
    print(f"     - 鲁棒性测试 (噪声、缺失数据)")
    print(f"     - 计算复杂度分析")
    
    print(f"  3. 可视化改进:")
    print(f"     - 添加置信区间")
    print(f"     - 特征重要性分析")
    print(f"     - 注意力权重可视化")
    print(f"     - 错误案例分析")

def run_demo_analysis(happen):
    """
    运行演示分析（当实际数据不可用时）
    """
    print("使用模拟数据进行演示分析...")
    
    # 模拟SECOM类似的数据
    np.random.seed(42)
    n_samples = 471
    
    # 模拟正常阶段的SPE值
    normal_spe = np.random.exponential(scale=50, size=happen)
    
    # 模拟故障阶段的SPE值（包含极值）
    fault_spe = np.random.exponential(scale=500, size=n_samples-happen)
    fault_spe[np.random.choice(len(fault_spe), 3)] *= 100  # 添加极值
    
    spe_values = np.concatenate([normal_spe, fault_spe])
    spe_limit = np.percentile(normal_spe, 95)
    
    # 创建分析报告
    metrics, threshold_analysis = create_comprehensive_report(
        spe_values, spe_limit, happen,
        model_name="Improved Transformer (Demo)",
        save_prefix="secom_demo"
    )
    
    return {
        'spe_values': spe_values,
        'spe_limit': spe_limit,
        'metrics': metrics,
        'threshold_analysis': threshold_analysis
    }

def main():
    """
    主函数
    """
    start_time = time.time()
    
    try:
        results = run_optimized_secom_analysis()
        
        print(f"\n{'='*80}")
        print("分析完成！")
        print(f"总耗时: {time.time() - start_time:.2f}秒")
        print(f"{'='*80}")
        
        print(f"\n生成的文件:")
        print(f"  - secom_improved_transformer_optimized.png")
        print(f"  - secom_improved_transformer_threshold_analysis.png")
        print(f"  - secom_methods_comparison.png (如果有对比数据)")
        print(f"  - secom_publication_quality.png")
        
        return results
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 
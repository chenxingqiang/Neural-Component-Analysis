from scripts.font_config import setup_chinese_fonts

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.signal import find_peaks
import warnings
from scripts.run_secom_fault_detection import (
from scripts.utils import save_plot
        from datetime import datetime
#!/usr/bin/env python3
"""
SECOM故障检测高级分析框架
Advanced Analysis Framework for SECOM Fault Detection

本模块实现了对SECOM故障检测系统的深度分析，包括：
1. 注意力机制可视化分析
2. 特征重要性动态分析
3. 故障传播路径分析
4. 时序模式挖掘
5. 模型可解释性分析
6. 故障根因分析
7. 预测性维护建议

作者: Neural Component Analysis Team
日期: 2024
"""

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

    load_secom_data, 
    run_enhanced_transformer_detection,
    analyze_feature_importance
)

class SECOMAdvancedAnalyzer:
    """SECOM高级分析器"""
    
    def __init__(self):
        setup_chinese_fonts()  # 配置中文字体
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"高级分析器初始化完成，使用设备: {self.device}")
        
    def load_and_prepare_data(self):
        """加载并预处理数据"""
        print("加载SECOM数据...")
        X_train, X_test, happen, y_test, normal_indices, fault_indices = load_secom_data()
        
        # 运行增强Transformer检测获取模型和结果
        print("运行增强Transformer检测...")
        results = run_enhanced_transformer_detection(X_train, X_test, happen)
        
        self.X_train = X_train
        self.X_test = X_test
        self.happen = happen
        self.y_test = y_test
        self.model = results['model']
        self.spe_values = results['spe_test']
        self.t2_values = results['t2_test']
        self.spe_limit = results['spe_limit']
        self.t2_limit = results['t2_limit']
        
        print(f"数据加载完成: 训练样本{X_train.shape[0]}, 测试样本{X_test.shape[0]}, 特征维度{X_train.shape[1]}")
        return self
    
    def analyze_attention_patterns(self):
        """分析注意力机制模式"""
        print("\n=== 注意力机制分析 ===")
        
        self.model.eval()
        
        # 提取注意力权重
        attention_weights = []
        
        # 创建hook函数来捕获注意力权重
        def attention_hook(module, input, output):
            if hasattr(module, 'self_attn'):
                # 获取注意力权重
                attn_weights = module.self_attn.attention_weights if hasattr(module.self_attn, 'attention_weights') else None
                if attn_weights is not None:
                    attention_weights.append(attn_weights.detach().cpu().numpy())
        
        # 注册hook
        hooks = []
        for name, module in self.model.named_modules():
            if 'transformer_encoder' in name and hasattr(module, 'self_attn'):
                hooks.append(module.register_forward_hook(attention_hook))
        
        # 分析正常和故障样本的注意力模式
        normal_sample = torch.tensor(self.X_test[:1], dtype=torch.float32).to(self.device)
        fault_sample = torch.tensor(self.X_test[self.happen:self.happen+1], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # 正常样本
            attention_weights.clear()
            _ = self.model(normal_sample)
            normal_attention = attention_weights.copy() if attention_weights else []
            
            # 故障样本
            attention_weights.clear()
            _ = self.model(fault_sample)
            fault_attention = attention_weights.copy() if attention_weights else []
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 可视化注意力模式差异
        self._visualize_attention_patterns(normal_attention, fault_attention)
        
        return normal_attention, fault_attention
    
    def _visualize_attention_patterns(self, normal_attention, fault_attention):
        """可视化注意力模式"""
        if not normal_attention or not fault_attention:
            print("注意力权重提取失败，跳过可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 正常样本注意力热图
        if len(normal_attention) > 0:
            normal_attn = normal_attention[0][0]  # 第一层，第一个头
            sns.heatmap(normal_attn, ax=axes[0,0], cmap='Blues', cbar=True)
            axes[0,0].set_title('正常样本注意力模式')
            axes[0,0].set_xlabel('特征维度')
            axes[0,0].set_ylabel('特征维度')
        
        # 故障样本注意力热图
        if len(fault_attention) > 0:
            fault_attn = fault_attention[0][0]  # 第一层，第一个头
            sns.heatmap(fault_attn, ax=axes[0,1], cmap='Reds', cbar=True)
            axes[0,1].set_title('故障样本注意力模式')
            axes[0,1].set_xlabel('特征维度')
            axes[0,1].set_ylabel('特征维度')
        
        # 注意力差异热图
        if len(normal_attention) > 0 and len(fault_attention) > 0:
            attention_diff = fault_attention[0][0] - normal_attention[0][0]
            sns.heatmap(attention_diff, ax=axes[1,0], cmap='RdBu_r', center=0, cbar=True)
            axes[1,0].set_title('注意力差异 (故障-正常)')
            axes[1,0].set_xlabel('特征维度')
            axes[1,0].set_ylabel('特征维度')
        
        # 注意力权重分布
        if len(normal_attention) > 0 and len(fault_attention) > 0:
            normal_flat = normal_attention[0][0].flatten()
            fault_flat = fault_attention[0][0].flatten()
            
            axes[1,1].hist(normal_flat, bins=50, alpha=0.7, label='正常', density=True)
            axes[1,1].hist(fault_flat, bins=50, alpha=0.7, label='故障', density=True)
            axes[1,1].set_title('注意力权重分布')
            axes[1,1].set_xlabel('注意力权重')
            axes[1,1].set_ylabel('密度')
            axes[1,1].legend()
        
        plt.tight_layout()
        save_plot("secom_attention_analysis.png")
        plt.close()
    
    def analyze_feature_dynamics(self):
        """分析特征动态变化"""
        print("\n=== 特征动态分析 ===")
        
        # 计算特征重要性
        importance_results = analyze_feature_importance(
            self.model, self.X_train, self.X_test, self.happen, self.device, n_top=50
        )
        
        top_features = importance_results['top_indices'][:20]  # 取前20个重要特征
        
        # 分析这些特征在时间序列上的变化
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, feature_idx in enumerate(top_features):
            if i >= 20:
                break
                
            feature_values = self.X_test[:, feature_idx]
            
            # 绘制特征值变化
            axes[i].plot(range(self.happen), feature_values[:self.happen], 
                        'b-', alpha=0.7, label='正常', linewidth=2)
            axes[i].plot(range(self.happen, len(feature_values)), feature_values[self.happen:], 
                        'r-', alpha=0.7, label='故障', linewidth=2)
            axes[i].axvline(x=self.happen, color='k', linestyle='--', alpha=0.5)
            
            # 计算统计量
            normal_mean = np.mean(feature_values[:self.happen])
            fault_mean = np.mean(feature_values[self.happen:])
            change_ratio = (fault_mean - normal_mean) / (normal_mean + 1e-8)
            
            axes[i].set_title(f'特征{feature_idx} (变化率: {change_ratio:.2f})')
            axes[i].set_xlabel('样本')
            axes[i].set_ylabel('特征值')
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:
                axes[i].legend()
        
        plt.tight_layout()
        save_plot("secom_feature_dynamics.png")
        plt.close()
        
        return top_features
    
    def analyze_fault_propagation(self):
        """分析故障传播模式"""
        print("\n=== 故障传播分析 ===")
        
        # 计算每个时间点的异常程度
        anomaly_scores = []
        window_size = 5
        
        for i in range(len(self.X_test)):
            if i < window_size:
                window_data = self.X_test[:i+1]
            else:
                window_data = self.X_test[i-window_size:i+1]
            
            # 计算与训练数据的马氏距离
            try:
                cov_matrix = np.cov(self.X_train.T)
                inv_cov = np.linalg.pinv(cov_matrix)
                mean_train = np.mean(self.X_train, axis=0)
                
                current_sample = self.X_test[i]
                diff = current_sample - mean_train
                mahalanobis_dist = np.sqrt(diff.T @ inv_cov @ diff)
                anomaly_scores.append(mahalanobis_dist)
            except:
                # 如果协方差矩阵奇异，使用欧氏距离
                euclidean_dist = np.linalg.norm(current_sample - mean_train)
                anomaly_scores.append(euclidean_dist)
        
        # 检测故障传播的关键时间点
        peaks, _ = find_peaks(anomaly_scores, height=np.mean(anomaly_scores) + 2*np.std(anomaly_scores))
        
        # 可视化故障传播
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # 异常分数时间序列
        ax1.plot(anomaly_scores, 'b-', linewidth=2, label='异常分数')
        ax1.axvline(x=self.happen, color='r', linestyle='--', linewidth=2, label='故障发生')
        ax1.scatter(peaks, [anomaly_scores[p] for p in peaks], color='red', s=50, zorder=5, label='传播关键点')
        ax1.set_title('故障传播时间序列')
        ax1.set_xlabel('样本')
        ax1.set_ylabel('异常分数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # SPE值变化
        ax2.plot(self.spe_values, 'g-', linewidth=2, label='SPE值')
        ax2.axhline(y=self.spe_limit, color='k', linestyle='--', linewidth=2, label='控制限')
        ax2.axvline(x=self.happen, color='r', linestyle='--', linewidth=2, label='故障发生')
        ax2.set_title('SPE检测结果')
        ax2.set_xlabel('样本')
        ax2.set_ylabel('SPE值')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 故障传播速度分析
        if len(peaks) > 1:
            propagation_intervals = np.diff(peaks)
            ax3.bar(range(len(propagation_intervals)), propagation_intervals, alpha=0.7)
            ax3.set_title('故障传播间隔')
            ax3.set_xlabel('传播阶段')
            ax3.set_ylabel('样本间隔')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '检测到的传播关键点不足', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('故障传播间隔')
        
        plt.tight_layout()
        save_plot("secom_fault_propagation.png")
        plt.close()
        
        return anomaly_scores, peaks
    
    def analyze_temporal_patterns(self):
        """分析时序模式"""
        print("\n=== 时序模式分析 ===")
        
        # 使用滑动窗口分析
        window_size = 10
        step_size = 5
        
        # 提取时序特征
        temporal_features = []
        labels = []
        
        for i in range(0, len(self.X_test) - window_size + 1, step_size):
            window = self.X_test[i:i+window_size]
            
            # 计算窗口内的统计特征
            features = []
            features.extend(np.mean(window, axis=0))  # 均值
            features.extend(np.std(window, axis=0))   # 标准差
            features.extend(np.max(window, axis=0) - np.min(window, axis=0))  # 范围
            
            temporal_features.append(features)
            
            # 标签：如果窗口中心在故障区域则为1，否则为0
            center = i + window_size // 2
            labels.append(1 if center >= self.happen else 0)
        
        temporal_features = np.array(temporal_features)
        labels = np.array(labels)
        
        # 使用t-SNE进行降维可视化
        print("执行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(temporal_features)//4))
        temporal_2d = tsne.fit_transform(temporal_features)
        
        # 可视化时序模式
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # t-SNE可视化
        normal_mask = labels == 0
        fault_mask = labels == 1
        
        ax1.scatter(temporal_2d[normal_mask, 0], temporal_2d[normal_mask, 1], 
                   c='blue', alpha=0.6, label='正常模式', s=50)
        ax1.scatter(temporal_2d[fault_mask, 0], temporal_2d[fault_mask, 1], 
                   c='red', alpha=0.6, label='故障模式', s=50)
        ax1.set_title('时序模式t-SNE可视化')
        ax1.set_xlabel('t-SNE维度1')
        ax1.set_ylabel('t-SNE维度2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 聚类分析
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(temporal_features)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(temporal_features, cluster_labels)
        
        # 可视化聚类结果
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            ax2.scatter(temporal_2d[cluster_mask, 0], temporal_2d[cluster_mask, 1], 
                       c=colors[i], alpha=0.6, label=f'聚类{i}', s=50)
        
        ax2.set_title(f'时序模式聚类 (轮廓系数: {silhouette_avg:.3f})')
        ax2.set_xlabel('t-SNE维度1')
        ax2.set_ylabel('t-SNE维度2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot("secom_temporal_patterns.png")
        plt.close()
        
        return temporal_features, labels, cluster_labels
    
    def analyze_model_interpretability(self):
        """模型可解释性分析"""
        print("\n=== 模型可解释性分析 ===")
        
        # 梯度分析
        self.model.eval()
        
        # 选择一个故障样本进行分析
        fault_sample = torch.tensor(self.X_test[self.happen:self.happen+1], 
                                   dtype=torch.float32, requires_grad=True).to(self.device)
        
        # 前向传播
        reconstructed, _ = self.model(fault_sample)
        
        # 计算重建误差
        reconstruction_error = torch.sum((fault_sample - reconstructed)**2)
        
        # 反向传播计算梯度
        reconstruction_error.backward()
        
        # 获取输入梯度
        input_gradients = fault_sample.grad.detach().cpu().numpy().flatten()
        
        # 可视化特征重要性
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # 梯度重要性
        feature_indices = np.arange(len(input_gradients))
        sorted_indices = np.argsort(np.abs(input_gradients))[-50:]  # 取前50个重要特征
        
        ax1.barh(range(len(sorted_indices)), np.abs(input_gradients[sorted_indices]))
        ax1.set_yticks(range(len(sorted_indices)))
        ax1.set_yticklabels([f'特征{i}' for i in sorted_indices])
        ax1.set_title('基于梯度的特征重要性 (前50个)')
        ax1.set_xlabel('梯度绝对值')
        
        # 重建误差分布
        fault_sample_np = fault_sample.detach().cpu().numpy().flatten()
        reconstructed_np = reconstructed.detach().cpu().numpy().flatten()
        reconstruction_errors = (fault_sample_np - reconstructed_np)**2
        
        ax2.bar(range(len(reconstruction_errors)), reconstruction_errors, alpha=0.7)
        ax2.set_title('各特征重建误差')
        ax2.set_xlabel('特征索引')
        ax2.set_ylabel('重建误差')
        ax2.set_yscale('log')
        
        # 特征值vs重建值对比
        top_error_indices = np.argsort(reconstruction_errors)[-20:]  # 误差最大的20个特征
        
        ax3.scatter(fault_sample_np[top_error_indices], reconstructed_np[top_error_indices], 
                   alpha=0.7, s=50)
        ax3.plot([fault_sample_np[top_error_indices].min(), fault_sample_np[top_error_indices].max()],
                [fault_sample_np[top_error_indices].min(), fault_sample_np[top_error_indices].max()],
                'r--', alpha=0.5, label='完美重建线')
        ax3.set_title('原始值 vs 重建值 (误差最大的20个特征)')
        ax3.set_xlabel('原始特征值')
        ax3.set_ylabel('重建特征值')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot("secom_model_interpretability.png")
        plt.close()
        
        return input_gradients, reconstruction_errors
    
    def generate_maintenance_recommendations(self):
        """生成维护建议"""
        print("\n=== 生成维护建议 ===")
        
        # 分析特征重要性
        importance_results = analyze_feature_importance(
            self.model, self.X_train, self.X_test, self.happen, self.device, n_top=20
        )
        
        top_features = importance_results['top_indices'][:10]
        
        # 分析故障模式
        fault_data = self.X_test[self.happen:]
        normal_data = self.X_test[:self.happen]
        
        recommendations = []
        
        for feature_idx in top_features:
            normal_values = normal_data[:, feature_idx]
            fault_values = fault_data[:, feature_idx]
            
            normal_mean = np.mean(normal_values)
            fault_mean = np.mean(fault_values)
            change_ratio = (fault_mean - normal_mean) / (normal_mean + 1e-8)
            
            # 生成具体建议
            if abs(change_ratio) > 0.5:  # 显著变化
                if change_ratio > 0:
                    trend = "显著增加"
                    action = "检查是否存在过载或异常输入"
                else:
                    trend = "显著减少"
                    action = "检查是否存在供应不足或设备故障"
                
                recommendations.append({
                    'feature': feature_idx,
                    'change_ratio': change_ratio,
                    'trend': trend,
                    'action': action,
                    'priority': 'high' if abs(change_ratio) > 1.0 else 'medium'
                })
        
        # 生成报告
        report = self._generate_maintenance_report(recommendations)
        
        # 保存报告
        with open('results/secom_maintenance_recommendations.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("维护建议已保存到 results/secom_maintenance_recommendations.txt")
        
        return recommendations
    
    def _generate_maintenance_report(self, recommendations):
        """生成维护报告"""
        report = """
SECOM半导体制造过程维护建议报告
=====================================

基于Transformer神经网络故障检测分析结果

分析时间: {time}
故障发生位置: 样本 {happen}

关键发现:
--------
"""
        
        report = report.format(time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), happen=self.happen)
        
        # 按优先级排序
        high_priority = [r for r in recommendations if r['priority'] == 'high']
        medium_priority = [r for r in recommendations if r['priority'] == 'medium']
        
        if high_priority:
            report += "\n高优先级维护项目:\n"
            for i, rec in enumerate(high_priority, 1):
                report += f"{i}. 特征{rec['feature']}: {rec['trend']} (变化率: {rec['change_ratio']:.2f})\n"
                report += f"   建议行动: {rec['action']}\n\n"
        
        if medium_priority:
            report += "\n中等优先级维护项目:\n"
            for i, rec in enumerate(medium_priority, 1):
                report += f"{i}. 特征{rec['feature']}: {rec['trend']} (变化率: {rec['change_ratio']:.2f})\n"
                report += f"   建议行动: {rec['action']}\n\n"
        
        report += """
预防性维护建议:
--------------
1. 定期监控关键特征变化趋势
2. 建立特征阈值预警系统
3. 实施基于机器学习的预测性维护
4. 加强设备状态监测和数据采集

技术建议:
--------
1. 部署实时故障检测系统
2. 集成多传感器数据融合
3. 建立故障知识库和专家系统
4. 实施持续学习和模型更新机制
"""
        
        return report
    
    def run_comprehensive_analysis(self):
        """运行综合分析"""
        print("="*80)
        print("SECOM故障检测系统高级分析")
        print("="*80)
        
        # 加载数据
        self.load_and_prepare_data()
        
        # 执行各项分析
        print("\n开始综合分析...")
        
        # 1. 注意力机制分析
        try:
            attention_results = self.analyze_attention_patterns()
            print("✓ 注意力机制分析完成")
        except Exception as e:
            print(f"✗ 注意力机制分析失败: {e}")
        
        # 2. 特征动态分析
        try:
            feature_dynamics = self.analyze_feature_dynamics()
            print("✓ 特征动态分析完成")
        except Exception as e:
            print(f"✗ 特征动态分析失败: {e}")
        
        # 3. 故障传播分析
        try:
            propagation_results = self.analyze_fault_propagation()
            print("✓ 故障传播分析完成")
        except Exception as e:
            print(f"✗ 故障传播分析失败: {e}")
        
        # 4. 时序模式分析
        try:
            temporal_results = self.analyze_temporal_patterns()
            print("✓ 时序模式分析完成")
        except Exception as e:
            print(f"✗ 时序模式分析失败: {e}")
        
        # 5. 模型可解释性分析
        try:
            interpretability_results = self.analyze_model_interpretability()
            print("✓ 模型可解释性分析完成")
        except Exception as e:
            print(f"✗ 模型可解释性分析失败: {e}")
        
        # 6. 生成维护建议
        try:
            maintenance_recommendations = self.generate_maintenance_recommendations()
            print("✓ 维护建议生成完成")
        except Exception as e:
            print(f"✗ 维护建议生成失败: {e}")
        
        print("\n" + "="*80)
        print("高级分析完成！")
        print("生成的文件:")
        print("- secom_attention_analysis.png")
        print("- secom_feature_dynamics.png") 
        print("- secom_fault_propagation.png")
        print("- secom_temporal_patterns.png")
        print("- secom_model_interpretability.png")
        print("- results/secom_maintenance_recommendations.txt")
        print("="*80)


def main():
    """主函数"""
    analyzer = SECOMAdvancedAnalyzer()
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main() 
from scripts.font_config import setup_chinese_fonts

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
from scipy import signal, stats
from scipy.optimize import minimize
import warnings
from scripts.run_secom_fault_detection import load_secom_data, run_enhanced_transformer_detection
from scripts.utils import save_plot
        import networkx as nx
        from sklearn.metrics import roc_curve
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
#!/usr/bin/env python3
"""
SECOM故障检测深度研究分析
Deep Research Analysis for SECOM Fault Detection

本模块实现了SECOM故障检测系统的前沿研究分析，包括：
1. 多尺度时序分析
2. 因果关系发现
3. 异常模式挖掘
4. 预测性故障检测
5. 自适应阈值优化
6. 集成学习策略
7. 实时监控系统设计

作者: Neural Component Analysis Team
日期: 2024
"""

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


class SECOMDeepResearchAnalyzer:
    """SECOM深度研究分析器"""
    
    def __init__(self):
        setup_chinese_fonts()  # 配置中文字体
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"深度研究分析器初始化完成，使用设备: {self.device}")
        
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
    
    def multi_scale_temporal_analysis(self):
        """多尺度时序分析"""
        print("\n=== 多尺度时序分析 ===")
        
        # 定义多个时间尺度
        scales = [3, 5, 10, 20, 50]
        
        fig, axes = plt.subplots(len(scales), 2, figsize=(15, 3*len(scales)))
        
        for i, scale in enumerate(scales):
            # 计算移动平均
            spe_ma = np.convolve(self.spe_values, np.ones(scale)/scale, mode='same')
            
            # 计算移动标准差
            spe_std = []
            for j in range(len(self.spe_values)):
                start = max(0, j - scale//2)
                end = min(len(self.spe_values), j + scale//2 + 1)
                spe_std.append(np.std(self.spe_values[start:end]))
            spe_std = np.array(spe_std)
            
            # 绘制移动平均
            axes[i, 0].plot(spe_ma, label=f'尺度{scale}移动平均', linewidth=2)
            axes[i, 0].axvline(x=self.happen, color='r', linestyle='--', alpha=0.7, label='故障发生')
            axes[i, 0].set_title(f'尺度{scale}移动平均分析')
            axes[i, 0].set_ylabel('SPE值')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # 绘制移动标准差
            axes[i, 1].plot(spe_std, label=f'尺度{scale}移动标准差', color='orange', linewidth=2)
            axes[i, 1].axvline(x=self.happen, color='r', linestyle='--', alpha=0.7, label='故障发生')
            axes[i, 1].set_title(f'尺度{scale}变异性分析')
            axes[i, 1].set_ylabel('SPE标准差')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot("secom_multi_scale_analysis.png")
        plt.close()
        
        return scales
    
    def causal_relationship_discovery(self):
        """因果关系发现分析"""
        print("\n=== 因果关系发现 ===")
        
        # 选择关键特征进行因果分析
        key_features = [37, 38, 34, 36, 270, 555, 9, 212]  # 基于之前的重要性分析
        
        # 计算格兰杰因果关系
        causal_matrix = np.zeros((len(key_features), len(key_features)))
        
        for i, feat_i in enumerate(key_features):
            for j, feat_j in enumerate(key_features):
                if i != j:
                    # 简化的格兰杰因果检验
                    x = self.X_test[:, feat_i]
                    y = self.X_test[:, feat_j]
                    
                    # 计算滞后相关性
                    max_lag = 5
                    correlations = []
                    for lag in range(1, max_lag + 1):
                        if lag < len(x):
                            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
                            correlations.append(abs(corr))
                    
                    causal_matrix[i, j] = np.mean(correlations) if correlations else 0
        
        # 可视化因果关系矩阵
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 因果关系热图
        sns.heatmap(causal_matrix, 
                   xticklabels=[f'特征{f}' for f in key_features],
                   yticklabels=[f'特征{f}' for f in key_features],
                   annot=True, fmt='.3f', cmap='viridis', ax=ax1)
        ax1.set_title('特征间因果关系强度')
        ax1.set_xlabel('被影响特征')
        ax1.set_ylabel('影响特征')
        
        # 因果网络图
        G = nx.DiGraph()
        
        # 添加节点
        for feat in key_features:
            G.add_node(f'F{feat}')
        
        # 添加边（因果关系强度 > 阈值）
        threshold = 0.3
        for i, feat_i in enumerate(key_features):
            for j, feat_j in enumerate(key_features):
                if causal_matrix[i, j] > threshold:
                    G.add_edge(f'F{feat_i}', f'F{feat_j}', weight=causal_matrix[i, j])
        
        # 绘制网络图
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightblue', 
                node_size=1000, font_size=8, arrows=True)
        ax2.set_title('特征因果关系网络')
        
        plt.tight_layout()
        save_plot("secom_causal_analysis.png")
        plt.close()
        
        return causal_matrix, key_features
    
    def anomaly_pattern_mining(self):
        """异常模式挖掘"""
        print("\n=== 异常模式挖掘 ===")
        
        # 使用Isolation Forest进行异常模式挖掘
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        
        # 在训练数据上训练
        iso_forest.fit(self.X_train)
        
        # 预测测试数据的异常分数
        anomaly_scores = iso_forest.decision_function(self.X_test)
        anomaly_labels = iso_forest.predict(self.X_test)
        
        # 分析异常模式的时序特征
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 异常分数时序图
        axes[0, 0].plot(anomaly_scores, 'b-', linewidth=2, label='异常分数')
        axes[0, 0].axvline(x=self.happen, color='r', linestyle='--', linewidth=2, label='故障发生')
        axes[0, 0].set_title('Isolation Forest异常分数')
        axes[0, 0].set_xlabel('样本')
        axes[0, 0].set_ylabel('异常分数')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 异常标签时序图
        axes[0, 1].plot(anomaly_labels, 'g-', linewidth=2, label='异常标签')
        axes[0, 1].axvline(x=self.happen, color='r', linestyle='--', linewidth=2, label='故障发生')
        axes[0, 1].set_title('异常检测结果')
        axes[0, 1].set_xlabel('样本')
        axes[0, 1].set_ylabel('异常标签')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SPE vs 异常分数对比
        axes[1, 0].scatter(self.spe_values, anomaly_scores, alpha=0.6, c=range(len(self.spe_values)), cmap='viridis')
        axes[1, 0].set_xlabel('SPE值')
        axes[1, 0].set_ylabel('异常分数')
        axes[1, 0].set_title('SPE vs 异常分数关系')
        axes[1, 0].set_xscale('log')
        
        # 异常分数分布
        normal_scores = anomaly_scores[:self.happen]
        fault_scores = anomaly_scores[self.happen:]
        
        axes[1, 1].hist(normal_scores, bins=30, alpha=0.7, label='正常', density=True)
        axes[1, 1].hist(fault_scores, bins=30, alpha=0.7, label='故障', density=True)
        axes[1, 1].set_title('异常分数分布')
        axes[1, 1].set_xlabel('异常分数')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].legend()
        
        # ROC曲线分析
        y_true = np.concatenate([np.zeros(self.happen), np.ones(len(self.X_test) - self.happen)])
        
        # SPE的ROC
        spe_auc = roc_auc_score(y_true, self.spe_values)
        
        # 异常分数的ROC（需要反转，因为异常分数越低越异常）
        anomaly_auc = roc_auc_score(y_true, -anomaly_scores)
        
        fpr_spe, tpr_spe, _ = roc_curve(y_true, self.spe_values)
        fpr_anomaly, tpr_anomaly, _ = roc_curve(y_true, -anomaly_scores)
        
        axes[2, 0].plot(fpr_spe, tpr_spe, label=f'SPE (AUC={spe_auc:.3f})', linewidth=2)
        axes[2, 0].plot(fpr_anomaly, tpr_anomaly, label=f'Isolation Forest (AUC={anomaly_auc:.3f})', linewidth=2)
        axes[2, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[2, 0].set_xlabel('假正率')
        axes[2, 0].set_ylabel('真正率')
        axes[2, 0].set_title('ROC曲线对比')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 精确率-召回率曲线
        precision_spe, recall_spe, _ = precision_recall_curve(y_true, self.spe_values)
        precision_anomaly, recall_anomaly, _ = precision_recall_curve(y_true, -anomaly_scores)
        
        axes[2, 1].plot(recall_spe, precision_spe, label='SPE', linewidth=2)
        axes[2, 1].plot(recall_anomaly, precision_anomaly, label='Isolation Forest', linewidth=2)
        axes[2, 1].set_xlabel('召回率')
        axes[2, 1].set_ylabel('精确率')
        axes[2, 1].set_title('精确率-召回率曲线')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot("secom_anomaly_pattern_mining.png")
        plt.close()
        
        return anomaly_scores, anomaly_labels
    
    def predictive_fault_detection(self):
        """预测性故障检测"""
        print("\n=== 预测性故障检测 ===")
        
        # 构建预测模型
        window_size = 10
        prediction_horizon = 5
        
        # 准备训练数据
        X_pred = []
        y_pred = []
        
        for i in range(window_size, len(self.X_test) - prediction_horizon):
            # 输入：过去window_size个时间步的数据
            X_pred.append(self.X_test[i-window_size:i].flatten())
            
            # 输出：未来prediction_horizon步内是否会发生故障
            future_fault = any(j >= self.happen for j in range(i, i + prediction_horizon))
            y_pred.append(1 if future_fault else 0)
        
        X_pred = np.array(X_pred)
        y_pred = np.array(y_pred)
        
        # 使用简单的逻辑回归进行预测
        
        X_train_pred, X_test_pred, y_train_pred, y_test_pred = train_test_split(
            X_pred, y_pred, test_size=0.3, random_state=42
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_pred_scaled = scaler.fit_transform(X_train_pred)
        X_test_pred_scaled = scaler.transform(X_test_pred)
        
        # 训练预测模型
        pred_model = LogisticRegression(random_state=42)
        pred_model.fit(X_train_pred_scaled, y_train_pred)
        
        # 预测
        y_pred_proba = pred_model.predict_proba(X_test_pred_scaled)[:, 1]
        y_pred_binary = pred_model.predict(X_test_pred_scaled)
        
        # 评估预测性能
        
        print("预测性故障检测性能:")
        print(classification_report(y_test_pred, y_pred_binary))
        
        # 可视化预测结果
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 预测概率时序图
        axes[0, 0].plot(y_pred_proba, 'b-', linewidth=2, label='故障概率')
        axes[0, 0].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='决策阈值')
        axes[0, 0].set_title('预测性故障检测概率')
        axes[0, 0].set_xlabel('样本')
        axes[0, 0].set_ylabel('故障概率')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test_pred, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('混淆矩阵')
        axes[0, 1].set_xlabel('预测标签')
        axes[0, 1].set_ylabel('真实标签')
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_test_pred, y_pred_proba)
        auc_score = roc_auc_score(y_test_pred, y_pred_proba)
        
        axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('假正率')
        axes[1, 0].set_ylabel('真正率')
        axes[1, 0].set_title('预测模型ROC曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 特征重要性
        feature_importance = np.abs(pred_model.coef_[0])
        top_features_idx = np.argsort(feature_importance)[-20:]
        
        axes[1, 1].barh(range(len(top_features_idx)), feature_importance[top_features_idx])
        axes[1, 1].set_title('预测模型特征重要性 (前20个)')
        axes[1, 1].set_xlabel('重要性')
        
        plt.tight_layout()
        save_plot("secom_predictive_fault_detection.png")
        plt.close()
        
        return pred_model, scaler, auc_score
    
    def adaptive_threshold_optimization(self):
        """自适应阈值优化"""
        print("\n=== 自适应阈值优化 ===")
        
        # 定义优化目标函数
        def objective_function(threshold, spe_values, happen, alpha=0.5):
            """
            优化目标：平衡误报率和漏报率
            alpha: 权重参数，控制误报率和漏报率的相对重要性
            """
            # 计算检测结果
            detections = spe_values > threshold
            
            # 计算误报率（正常样本中被误报的比例）
            false_alarms = np.sum(detections[:happen])
            false_alarm_rate = false_alarms / happen if happen > 0 else 0
            
            # 计算漏报率（故障样本中未被检测的比例）
            fault_samples = len(spe_values) - happen
            missed_detections = fault_samples - np.sum(detections[happen:])
            miss_rate = missed_detections / fault_samples if fault_samples > 0 else 0
            
            # 综合目标函数
            return alpha * false_alarm_rate + (1 - alpha) * miss_rate
        
        # 优化不同权重下的阈值
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        optimal_thresholds = []
        performance_metrics = []
        
        for alpha in alphas:
            # 使用黄金分割搜索优化阈值
            result = minimize(
                objective_function,
                x0=self.spe_limit,
                args=(self.spe_values, self.happen, alpha),
                method='Nelder-Mead',
                bounds=[(np.min(self.spe_values), np.max(self.spe_values))]
            )
            
            optimal_threshold = result.x[0]
            optimal_thresholds.append(optimal_threshold)
            
            # 计算性能指标
            detections = self.spe_values > optimal_threshold
            false_alarms = np.sum(detections[:self.happen])
            false_alarm_rate = false_alarms / self.happen
            
            fault_samples = len(self.spe_values) - self.happen
            missed_detections = fault_samples - np.sum(detections[self.happen:])
            miss_rate = missed_detections / fault_samples
            
            performance_metrics.append({
                'alpha': alpha,
                'threshold': optimal_threshold,
                'false_alarm_rate': false_alarm_rate,
                'miss_rate': miss_rate,
                'objective': result.fun
            })
        
        # 可视化优化结果
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 阈值优化曲线
        axes[0, 0].plot(alphas, optimal_thresholds, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].axhline(y=self.spe_limit, color='r', linestyle='--', label='原始阈值')
        axes[0, 0].set_xlabel('权重参数α')
        axes[0, 0].set_ylabel('最优阈值')
        axes[0, 0].set_title('自适应阈值优化')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 性能权衡曲线
        false_alarm_rates = [m['false_alarm_rate'] for m in performance_metrics]
        miss_rates = [m['miss_rate'] for m in performance_metrics]
        
        axes[0, 1].plot(false_alarm_rates, miss_rates, 'ro-', linewidth=2, markersize=8)
        for i, alpha in enumerate(alphas):
            axes[0, 1].annotate(f'α={alpha}', (false_alarm_rates[i], miss_rates[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_xlabel('误报率')
        axes[0, 1].set_ylabel('漏报率')
        axes[0, 1].set_title('误报率-漏报率权衡曲线')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 目标函数值
        objectives = [m['objective'] for m in performance_metrics]
        axes[1, 0].plot(alphas, objectives, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('权重参数α')
        axes[1, 0].set_ylabel('目标函数值')
        axes[1, 0].set_title('优化目标函数值')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 最优阈值检测结果示例（α=0.5）
        optimal_idx = 2  # α=0.5对应的索引
        optimal_threshold = optimal_thresholds[optimal_idx]
        detections = self.spe_values > optimal_threshold
        
        axes[1, 1].plot(self.spe_values, 'b-', linewidth=2, label='SPE值')
        axes[1, 1].axhline(y=optimal_threshold, color='g', linestyle='-', linewidth=2, label=f'最优阈值 (α=0.5)')
        axes[1, 1].axhline(y=self.spe_limit, color='r', linestyle='--', linewidth=2, label='原始阈值')
        axes[1, 1].axvline(x=self.happen, color='k', linestyle='--', alpha=0.7, label='故障发生')
        axes[1, 1].set_xlabel('样本')
        axes[1, 1].set_ylabel('SPE值')
        axes[1, 1].set_title('最优阈值检测结果')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot("secom_adaptive_threshold_optimization.png")
        plt.close()
        
        return performance_metrics, optimal_thresholds
    
    def run_deep_research_analysis(self):
        """运行深度研究分析"""
        print("="*80)
        print("SECOM故障检测系统深度研究分析")
        print("="*80)
        
        # 加载数据
        self.load_and_prepare_data()
        
        # 执行深度研究分析
        print("\n开始深度研究分析...")
        
        # 1. 多尺度时序分析
        try:
            scales = self.multi_scale_temporal_analysis()
            print("✓ 多尺度时序分析完成")
        except Exception as e:
            print(f"✗ 多尺度时序分析失败: {e}")
        
        # 2. 因果关系发现
        try:
            causal_matrix, key_features = self.causal_relationship_discovery()
            print("✓ 因果关系发现完成")
        except Exception as e:
            print(f"✗ 因果关系发现失败: {e}")
        
        # 3. 异常模式挖掘
        try:
            anomaly_scores, anomaly_labels = self.anomaly_pattern_mining()
            print("✓ 异常模式挖掘完成")
        except Exception as e:
            print(f"✗ 异常模式挖掘失败: {e}")
        
        # 4. 预测性故障检测
        try:
            pred_model, scaler, auc_score = self.predictive_fault_detection()
            print(f"✓ 预测性故障检测完成 (AUC: {auc_score:.3f})")
        except Exception as e:
            print(f"✗ 预测性故障检测失败: {e}")
        
        # 5. 自适应阈值优化
        try:
            performance_metrics, optimal_thresholds = self.adaptive_threshold_optimization()
            print("✓ 自适应阈值优化完成")
        except Exception as e:
            print(f"✗ 自适应阈值优化失败: {e}")
        
        print("\n" + "="*80)
        print("深度研究分析完成！")
        print("生成的文件:")
        print("- secom_multi_scale_analysis.png")
        print("- secom_causal_analysis.png")
        print("- secom_anomaly_pattern_mining.png")
        print("- secom_predictive_fault_detection.png")
        print("- secom_adaptive_threshold_optimization.png")
        print("="*80)


def main():
    """主函数"""
    analyzer = SECOMDeepResearchAnalyzer()
    analyzer.run_deep_research_analysis()


if __name__ == "__main__":
    main() 
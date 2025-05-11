import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# 加载数据
features = pd.read_csv('F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM.TXT',
                       delim_whitespace=True, header=None)

labels = pd.read_csv('F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM_labels.TXT',
                     delim_whitespace=True, header=None, names=["Label", "Timestamp"])
labels["Label"] = labels["Label"].astype(int)

# 数据清洗和预处理
valid_indices = labels["Label"] != 0  # 过滤掉标签为 0 的无效数据
features = features[valid_indices]
labels = labels[valid_indices]

# 填充缺失值
features.fillna(features.mean(), inplace=True)

# 分离正常值和异常值
features_normal = features[labels["Label"] == -1]
features_anomaly = features[labels["Label"] == 1]

# 数据标准化（基于正常值数据计算均值和标准差）
scaler = StandardScaler()
features_normal_scaled = scaler.fit_transform(features_normal)
features_anomaly_scaled = scaler.transform(features_anomaly)

# KPCA降维（基于正常值数据进行训练）
kpca = KernelPCA(n_components=10, kernel='rbf', gamma=0.01, fit_inverse_transform=True)
features_normal_kpca = kpca.fit_transform(features_normal_scaled)

# 重构正常值数据
features_normal_reconstructed = kpca.inverse_transform(features_normal_kpca)

# 计算正常值的T²和SPE
T2_normal = np.sum(features_normal_kpca ** 2, axis=1)
SPE_normal = np.sum((features_normal_scaled - features_normal_reconstructed) ** 2, axis=1)

# 利用核密度估计确定控制限
def kde_control_limit(data, confidence=0.95):
    """计算核密度估计下的控制限."""
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)
    cdf = np.cumsum(kde(x)) * (x[1] - x[0])  # 计算核密度的累计分布
    limit_index = np.argmax(cdf >= confidence)
    return x[limit_index]

T2_limit = kde_control_limit(T2_normal, confidence=0.95)
SPE_limit = kde_control_limit(SPE_normal, confidence=0.95)

# 对异常值数据进行降维和重构
features_anomaly_kpca = kpca.transform(features_anomaly_scaled)
features_anomaly_reconstructed = kpca.inverse_transform(features_anomaly_kpca)

# 计算异常值的T²和SPE
T2_anomaly = np.sum(features_anomaly_kpca ** 2, axis=1)
SPE_anomaly = np.sum((features_anomaly_scaled - features_anomaly_reconstructed) ** 2, axis=1)

# 异常检测
anomalies_T2 = T2_anomaly > T2_limit
anomalies_SPE = SPE_anomaly > SPE_limit
anomalies_combined = anomalies_T2 | anomalies_SPE

# 绘图
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(T2_anomaly, label="T² (Anomaly Data)")
plt.axhline(T2_limit, color='r', linestyle='--', label="T² Limit (KDE)")
plt.legend()
plt.title("T² Statistic with KDE Limit (Anomaly Data)")

plt.subplot(2, 1, 2)
plt.plot(SPE_anomaly, label="SPE (Anomaly Data)")
plt.axhline(SPE_limit, color='r', linestyle='--', label="SPE Limit (KDE)")
plt.legend()
plt.title("SPE Statistic with KDE Limit (Anomaly Data)")

plt.tight_layout()
plt.show()

# 性能评估
y_true = np.ones(len(features_anomaly))  # 真实标签为异常（1）
y_pred = anomalies_combined.astype(int)  # 检测结果

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


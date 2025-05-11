import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib
matplotlib.use('TkAgg')

# 加载特征数据
features = pd.read_csv(
    'F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM.TXT',
    delim_whitespace=True, header=None
)

# 加载标签数据
labels = pd.read_csv(
    'F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM_labels.TXT',
    delim_whitespace=True, header=None, names=["Label", "Timestamp"]
)
labels["Label"] = labels["Label"].astype(int)  # 将标签转换为整数

# 填补缺失值
features.fillna(features.mean(), inplace=True)

# 分离正常值和异常值
features_normal = features[labels["Label"] == -1]
features_anomaly = features[labels["Label"] == 1]

# 数据标准化（基于正常值计算均值和标准差）
scaler = StandardScaler()
features_normal_scaled = scaler.fit_transform(features_normal)
features_anomaly_scaled = scaler.transform(features_anomaly)

# PCA降维，保留95%的累计方差
pca = PCA(n_components=0.95)
features_normal_pca = pca.fit_transform(features_normal_scaled)

# 查看保留的主成分数量和解释方差
print(f"Number of components retained: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# 计算正常值的T²和SPE
T2_normal = np.sum((features_normal_pca / pca.singular_values_) ** 2, axis=1)
features_normal_reconstructed = pca.inverse_transform(features_normal_pca)
SPE_normal = np.sum((features_normal_scaled - features_normal_reconstructed) ** 2, axis=1)

# 使用核密度估计（KDE）确定控制限
def compute_kde_limit(data, quantile=0.95):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data.reshape(-1, 1))
    scores = kde.score_samples(data.reshape(-1, 1))
    threshold_index = int((1 - quantile) * len(scores))
    sorted_indices = np.argsort(scores)
    limit = data[sorted_indices[threshold_index]]
    return limit

T2_limit = compute_kde_limit(T2_normal, quantile=0.95)
SPE_limit = compute_kde_limit(SPE_normal, quantile=0.95)

print(f"T² Limit (KDE): {T2_limit}")
print(f"SPE Limit (KDE): {SPE_limit}")

# 对异常值数据降维和重构
features_anomaly_pca = pca.transform(features_anomaly_scaled)
features_anomaly_reconstructed = pca.inverse_transform(features_anomaly_pca)

# 计算异常值的T²和SPE
T2_anomaly = np.sum((features_anomaly_pca / pca.singular_values_) ** 2, axis=1)
SPE_anomaly = np.sum((features_anomaly_scaled - features_anomaly_reconstructed) ** 2, axis=1)

# 判断异常
anomalies = (T2_anomaly > T2_limit) | (SPE_anomaly > SPE_limit)

# 绘制 T² 和 SPE 图
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(T2_anomaly, label="T² (Anomaly Data)")
plt.axhline(T2_limit, color='r', linestyle='--', label="T² Limit (KDE)")
plt.legend()
plt.title("T² Statistic (Anomaly Data)")

plt.subplot(2, 1, 2)
plt.plot(SPE_anomaly, label="SPE (Anomaly Data)")
plt.axhline(SPE_limit, color='r', linestyle='--', label="SPE Limit (KDE)")
plt.legend()
plt.title("SPE Statistic (Anomaly Data)")

plt.tight_layout()
plt.show()

# 性能评估
y_true = np.ones(len(features_anomaly))  # 异常值的真实标签为 1
y_pred = anomalies.astype(int)  # 检测结果

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_true, anomalies)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


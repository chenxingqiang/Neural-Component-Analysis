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

# 检查数据形状
print(features.shape)
print(labels.shape)

# 填补缺失值
features.fillna(features.mean(), inplace=True)

# 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA降维，保留95%的累计方差
pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_scaled)

# 查看保留的主成分数量和解释方差
print(f"Number of components retained: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# 计算 T² 和 SPE
T2 = np.sum((features_pca / pca.singular_values_) ** 2, axis=1)
features_reconstructed = pca.inverse_transform(features_pca)
SPE = np.sum((features_scaled - features_reconstructed) ** 2, axis=1)

# 使用核密度估计（KDE）确定控制限
def compute_kde_limit(data, quantile=0.95):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data.reshape(-1, 1))
    scores = kde.score_samples(data.reshape(-1, 1))
    threshold_index = int((1 - quantile) * len(scores))
    sorted_indices = np.argsort(scores)
    limit = data[sorted_indices[threshold_index]]
    return limit

T2_limit = compute_kde_limit(T2, quantile=0.95)
SPE_limit = compute_kde_limit(SPE, quantile=0.95)

print(f"T² Limit (KDE): {T2_limit}")
print(f"SPE Limit (KDE): {SPE_limit}")

# 绘制 T² 和 SPE 图
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(T2, label="T²")
plt.axhline(T2_limit, color='r', linestyle='--', label="T² Limit (KDE)")
plt.legend()
plt.title("T² Statistic")

plt.subplot(2, 1, 2)
plt.plot(SPE, label="SPE")
plt.axhline(SPE_limit, color='r', linestyle='--', label="SPE Limit (KDE)")
plt.legend()
plt.title("SPE Statistic")

plt.tight_layout()
plt.show()

# 判断异常
anomalies = (T2 > T2_limit) | (SPE > SPE_limit)

# 计算混淆矩阵和分类报告
y_true = labels["Label"].values
y_pred = anomalies.astype(int)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_true, anomalies)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

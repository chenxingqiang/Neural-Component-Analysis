import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 读取 SECOM 数据集
# 加载特征数据 (SECOM.TXT)
X = pd.read_csv('/NCA/secom_data/SECOM.TXT', sep=' ', header=None)

# 加载标签数据 (SECOM_labels.TXT)
y = pd.read_csv('/NCA/secom_data/SECOM_labels.TXT', sep=' ', header=None).squeeze()  # 转换为 Series

# 查看数据的前几行
print("特征数据 (X) 的前几行：")
print(X.head())
print("\n标签数据 (y) 的前几行：")
print(y.head())

# 2. 划分数据集
# 80% 用于训练，20% 用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印训练集和测试集的大小
print(f"\n训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# 3. 数据标准化
scaler = StandardScaler()

# 训练集标准化
X_train_scaled = scaler.fit_transform(X_train)

# 测试集标准化（使用训练数据的均值和标准差进行转换）
X_test_scaled = scaler.transform(X_test)

# 打印标准化后的数据的一些信息
print("\n标准化后的训练数据 (X_train_scaled) 的前几行：")
print(X_train_scaled[:5])

# 4. 将数据保存为 .dat 格式
# 保存训练集特征数据
pd.DataFrame(X_train_scaled).to_csv('X_train.dat', sep=' ', header=False, index=False)

# 保存测试集特征数据
pd.DataFrame(X_test_scaled).to_csv('X_test.dat', sep=' ', header=False, index=False)

# 保存训练集标签数据
y_train.to_csv('y_train.dat', sep=' ', header=False, index=False)

# 保存测试集标签数据
y_test.to_csv('y_test.dat', sep=' ', header=False, index=False)

print("\n数据已成功保存为 .dat 格式！")

import numpy as np

# 假设 train_data 和 test_data 是您的原始数据
train_data = np.array([[1, 2, 3], [4, 5, 6]])  # 用您的实际数据替换
test_data = np.array([[7, 8, 9], [10, 11, 12]])

# 保存为 .npy 文件
np.save("train_data.npy", train_data)
np.save("test_data.npy", test_data)

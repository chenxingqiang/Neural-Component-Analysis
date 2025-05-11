import os
import numpy as np
import sklearn.preprocessing as prep
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 按空格分割特征，过滤空值
            row = [float(x) if x not in ['NaN', 'nan', ''] else np.nan
                   for x in line.strip().split()]
            if len(row) > 0:
                data.append(row)
    return np.array(data)

def replace_NaN_with_row_mean(arr):
    # 检查输入是否为二维数组
    if arr.ndim != 2:
        raise ValueError("输入必须是二维数组")

    # 计算行均值（忽略NaN）
    row_means = np.nanmean(arr, axis=1)

    # 处理全NaN行：将均值设为0或全局均值
    global_mean = np.nanmean(arr) if not np.isnan(np.nanmean(arr)) else 0.0
    row_means = np.nan_to_num(row_means, nan=global_mean)

    # 生成NaN位置的掩码并填充
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.take(row_means, np.where(nan_mask)[0])
    return arr



def extract_first_column_values():
    # 指定文件路径（注意大小写和路径正确性）
    file_path = r"F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM_labels.TXT"

    try:
        # 读取文件（假设无标题行）
        df = pd.read_csv(file_path, sep='\s+', header=None)

        # 提取所有行的第一列（第0列）
        first_column_values = df.iloc[:, 0].values  # 转换为NumPy数组

        print("提取的第一列数据数组：")
        print(first_column_values)

        return first_column_values

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在！")
        return np.array([])  # 返回空数组
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return np.array([])



def combine_secom_data_labels_from_arrays(data_array, label_array):
    """
    直接通过数组合并 SECOM 数据与标签
    参数：
        data_array (np.ndarray或pd.DataFrame): 形状为 (n_samples, n_features) 的数据集
        label_array (np.ndarray或pd.Series): 形状为 (n_samples,) 的标签数组（-1 或 1）
    返回：
        normal_samples (pd.DataFrame): 正常样本（标签为 -1）
        anomaly_samples (pd.DataFrame): 异常样本（标签为 1）
    """
    try:
        # 1. 将输入转换为 DataFrame（确保数据类型兼容）
        if isinstance(data_array, np.ndarray):
            df_data = pd.DataFrame(data_array)
        elif isinstance(data_array, pd.DataFrame):
            df_data = data_array.copy()
        else:
            raise TypeError("data_array 必须是 NumPy 数组或 Pandas DataFrame")

        # 2. 验证标签数组
        labels = label_array.squeeze()  # 确保标签为一维
        if labels.ndim != 1:
            raise ValueError("标签数组必须是一维的")
        if len(labels) != df_data.shape[0]:
            raise ValueError(f"数据样本数 ({df_data.shape[0]}) 与标签数 ({len(labels)}) 不一致")

        # 3. 将标签添加到 DataFrame
        df_data['label'] = labels

        # 4. 分离正常/异常样本
        normal_samples = df_data[df_data['label'] == -1]
        anomaly_samples = df_data[df_data['label'] == 1]

        return normal_samples, anomaly_samples

    except Exception as e:
        print(f"合并数据时发生错误: {e}")
        return None, None



from sklearn.model_selection import train_test_split


def split_data_1(normal_samples, anomaly_samples, test_normal_ratio=0.2, train_normal_ratio=0.3, random_seed=42):
    """
    数据集划分函数
    :param normal_samples: 正常样本 DataFrame
    :param anomaly_samples: 异常样本 DataFrame
    :param test_normal_ratio: 测试集使用的正常样本比例 (默认20%)
    :param train_normal_ratio: 训练集使用的正常样本比例 (默认30%)
    :param random_seed: 随机种子 (默认42)
    :return: (训练集, 测试集)
    """
    # 计算划分比例
    total_normal = len(normal_samples)

    # 第一阶段：划分训练集
    train_normal = normal_samples.sample(frac=train_normal_ratio,
                                         random_state=random_seed)

    # 第二阶段：划分测试集的正常样本
    remaining_normal = normal_samples.drop(train_normal.index)
    test_normal = remaining_normal.sample(frac=test_normal_ratio / (1 - train_normal_ratio),
                                          random_state=random_seed)

    # 构造最终数据集
    train_set = train_normal
    test_set = pd.concat([test_normal, anomaly_samples], axis=0).sample(frac=1, random_state=random_seed)  # 打乱顺序

    return train_set, test_set

def split_data(normal_samples, anomaly_samples, test_normal_ratio=0.2, train_normal_ratio=0.3, random_seed=42):
    """
    数据集划分函数
    :param normal_samples: 正常样本 DataFrame
    :param anomaly_samples: 异常样本 DataFrame
    :param test_normal_ratio: 测试集使用的正常样本比例 (默认20%)
    :param train_normal_ratio: 训练集使用的正常样本比例 (默认30%)
    :param random_seed: 随机种子 (默认42)
    :return: (训练集, 测试集)
    """
    # 计算划分比例
    total_normal = len(normal_samples)

    # 第一阶段：划分训练集
    train_normal = normal_samples.sample(frac=train_normal_ratio,
                                         random_state=random_seed)

    # 第二阶段：划分测试集的正常样本
    remaining_normal = normal_samples.drop(train_normal.index)
    test_normal = remaining_normal.sample(frac=test_normal_ratio / (1 - train_normal_ratio),
                                      random_state=random_seed)

    # 构造最终数据集
    train_set = train_normal
    test_set = pd.concat([test_normal, anomaly_samples], axis=0).sample(frac=1, random_state=random_seed)  # 打乱顺序

    # 确保返回的数据集包含标签列
    return train_set, test_set


def read_secom_data():
    # 加载数据集数据
    data_file = load_data(
        "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM.TXT")
    print("数据形状:", data_file.shape)  # 预期输出 (1567, n_features)

    # data_file,labels_file=load_data()
    data_file = replace_NaN_with_row_mean(data_file)
    # 处理NaN
    processed_data = replace_NaN_with_row_mean(data_file)
    print("处理后数据形状:", processed_data.shape)


    #处理标签数据，获取第一列
    data_array = extract_first_column_values()

    print("处理后标签数据形状:", data_array.shape)


    #组合数据集和标签
    # 调用函数并输出统计信息
    normal, anomaly=combine_secom_data_labels_from_arrays(processed_data, data_array)

    if normal is not None:
        print(f"正常样本数: {len(normal)}")
        print(f"异常样本数: {len(anomaly)}")

    # 划分数据集为训练集和测试集
    train_data, test_data = split_data(normal_samples=normal,
                                       anomaly_samples=anomaly)
    # 去掉最后一列
    train_data = train_data.iloc[:, :-1]
    test_data = test_data.iloc[:, :-1]

    print(f"训练集维度: {train_data.shape}")
    print(f"测试集维度: {test_data.shape}")
    #print(f"测试集组成: {test_data['label'].value_counts()}")

    return train_data, test_data

def write_data(file_name, data):
    """
    """
    fi = os.path.join('data/{}.dat'.format(file_name))
    np.savetxt(fi, data, fmt='%f', delimiter='\t')



if __name__ == "__main__":
    read_secom_data()
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import matplotlib

# 设置交互模式
matplotlib.use('TkAgg')  # 或其他支持的交互框架
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

def load_secom_data():
    data = pd.read_csv(
        '/NCA/secom_data/SECOM.TXT',
        sep=' ', header=None)
    labels = pd.read_csv(
        '/NCA/secom_data/SECOM_labels.TXT',
        sep=' ', header=None, usecols=[0])
    labels = labels.iloc[:, 0].astype(int)
    data = data.dropna(axis=1, how='any')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, labels

def pca_dimensionality_reduction(data, n_components=6):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    print(f"Explained variance ratio by PCA components: {pca.explained_variance_ratio_}")
    return reduced_data

def visualize_pca_result(pca_result, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[labels == -1, 0], pca_result[labels == -1, 1], color='blue', label='Class -1 (Pass)', alpha=0.5)
    plt.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], color='red', label='Class 1 (Fail)', alpha=0.5)
    plt.title('PCA of SECOM Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
from mpl_toolkits.mplot3d import Axes3D

def visualize_pca_result_3d(pca_result, labels):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_result[labels == -1, 0], pca_result[labels == -1, 1], pca_result[labels == -1, 2],
               color='blue', label='Class -1 (Pass)', alpha=0.5)
    ax.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], pca_result[labels == 1, 2],
               color='red', label='Class 1 (Fail)', alpha=0.5)
    ax.set_title('PCA of SECOM Dataset (3D)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    scaled_data, labels = load_secom_data()
    print(f"Labels unique values after processing: {np.unique(labels)}")
    pca_result = pca_dimensionality_reduction(scaled_data, n_components=6)
    visualize_pca_result_3d(pca_result, labels)

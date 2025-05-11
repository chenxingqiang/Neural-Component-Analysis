import read_TEdat
import diagnosis_pca as pca
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

# Check if data directories exist
if not os.path.exists('./data/test') or not os.path.exists('./data/train'):
    os.makedirs('./data/test', exist_ok=True)
    os.makedirs('./data/train', exist_ok=True)
    print("Warning: Data directories created but may be empty. Using mock data.")
    # Create mock data for testing
    def create_mock_data(num_samples=500, num_features=52):
        return np.random.randn(num_samples, num_features).astype(np.float32)
    
    # Create 3 datasets for both train and test (normal and 2 fault conditions)
    data_train = [create_mock_data() for _ in range(3)]
    data_test = [create_mock_data() for _ in range(3)]
else:
    path_test = r'./data/test'
    path_train = r'./data/train'
    try:
        data_test, data_train = read_TEdat.read_all_data(path_test, path_train)
    except Exception as e:
        print(f"Error reading data: {e}. Using mock data.")
        # Create mock data for testing
        def create_mock_data(num_samples=500, num_features=52):
            return np.random.randn(num_samples, num_features).astype(np.float32)
        
        # Create 3 datasets for both train and test (normal and 2 fault conditions)
        data_train = [create_mock_data() for _ in range(3)]
        data_test = [create_mock_data() for _ in range(3)]

fault02_train, nor_train = data_train[2], data_train[0]
fault02_test, nor_test = data_test[2], data_test[0]

# 数据标准化
scaler = StandardScaler().fit(nor_train)
Xtrain_nor = scaler.transform(nor_train)
Xtest_nor = scaler.transform(nor_test)
Xtrain_fault = scaler.transform(fault02_train)
Xtest_fault = scaler.transform(fault02_test)
# PCA
t_limit, spe_limit, p, v, v_all, k, p_all = pca.pca_control_limit(Xtrain_nor)

# 在线监测
t2, spe = pca.pca_model_online(Xtest_fault, p, v)

# Set index for contribution graph (defaulting to 600 if too large)
index = min(600, Xtest_fault.shape[0]-1)  # Ensure index doesn't exceed data dimensions

# Draw control limit figure
pca.figure_control_limit(Xtest_fault, t_limit, spe_limit, t2, spe)

# Draw contribution graph with standardized data
# Pass Xtrain_nor as the training data parameter, not nor_train
pca.Contribution_graph(Xtest_fault, Xtrain_nor, index, p, p_all, v_all, k, t_limit)

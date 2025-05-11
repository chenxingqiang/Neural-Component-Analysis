import os
import numpy as np
import sklearn.preprocessing as prep
from scipy.io import loadmat
import matplotlib.pyplot as plt


def read_data(error=0, is_train=True):
    """
    Args:
        error (int): The index of error, 0 means normal data
        is_train (bool): Read train or test data
    Returns:
        data and labels
    """
    fi = os.path.join('data/',
        ('d0' if error < 10 else 'd') + str(error) + ('_te.dat' if is_train else '_te.dat'))
    data = np.fromfile(fi, dtype=np.float32, sep='   ')
    if fi == 'data/d00.dat':
        data = data.reshape(-1, 500).T
        # data = data.reshape(-1, 500)
    else:
        data = data.reshape(-1, 52)
    # if not is_train:
    #     data = data[160: ]
    return data, np.ones(data.shape[0], np.float32) * error


def write_data(file_name, data):
    """
    """
    fi = os.path.join('data/{}.dat'.format(file_name))
    np.savetxt(fi, data, fmt='%f', delimiter='\t')

def TE_prepare_data():
    fault_number = [0, 3]
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    j = 0
    for key in fault_number:
        if key < 10:
            file_name = './data/d0{}_te.dat'.format(key)
        else:
            file_name = './data/d{}_te.dat'.format(key)
        tmp_x = []
        tmp_y = []
        with open(file_name, "r") as f:  # 打开文件
            file = f.readlines()
            for line in file:
                x = []
                tmp_y.append(j)
                tmp = line.split()
                for k in tmp:
                    x.append(np.float32(k))
                tmp_x.append(x)
        tmp_x = tmp_x[159:-1]
        if len(train_data) == 0:
            train_data = tmp_x[0:500]
            test_data = tmp_x[500:800]
            train_label = tmp_y[0:500]
            test_label = tmp_y[500:800]
        else:
            train_data = np.vstack((train_data, tmp_x[0:500]))
            test_data = np.vstack((test_data, tmp_x[500:800]))
            train_label = np.hstack((train_label, tmp_y[0:500]))
            test_label = np.hstack((test_label, tmp_y[500:800]))
        j = j+1
    # 对训练、测试数据进行预处理
    scaler = prep.StandardScaler()  # 标准化
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, train_label, test_data, test_label

def one_hot(x, num_classes):
    x = x.flatten().astype('uint8')
    m = x.shape[0]
    x_onehot = np.zeros((m, num_classes))
    for i in range(m):
        x_onehot[i, x[i]] = 1
    return x_onehot

bearing_data = {                     #西储大学轴承数据
    "type_1": {
        "file": '097.mat',
        "key_name": 'X097_DE_time',
        "type": 1
    },
    "type_2": {
        "file": '118.mat',
        "key_name": 'X118_DE_time',
        "type": 2
    },
    "type_3": {
        "file": '185.mat',
        "key_name": 'X185_DE_time',
        "type": 3
    },
    "type_4": {
        "file": '222.mat',
        "key_name": 'X222_DE_time',
        "type": 4
    },
    "type_5": {
        "file": '105.mat',
        "key_name": 'X105_DE_time',
        "type": 5
    },
    "type_6": {
        "file": '169.mat',
        "key_name": 'X169_DE_time',
        "type": 6
    },
    "type_7": {
        "file": '3001.mat',
        "key_name": 'X056_DE_time',
        "type": 7
    },
    "type_8": {
        "file": '144.mat',
        "key_name": 'X144_DE_time',
        "type": 8
    },
    "type_9": {
        "file": '130.mat',
        "key_name": 'X130_DE_time',
        "type": 9
    },
    "type_10": {
        "file": '197.mat',
        "key_name": 'X197_DE_time',
        "type": 10
    },
    "type_11": {
        "file": '246.mat',
        "key_name": 'X246_DE_time',
        "type": 11
    },
    "type_12": {
        "file": '234.mat',
        "key_name": 'X234_DE_time',
        "type": 12
    }
}

def bearing_prepare_data():
    batch_size = 800
    n = 120000   #12000/800 = 150, 其中100个正常样本， 500个测试样本
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for k, v in bearing_data.items():
        file = loadmat("../bearing_data/12k_Drive_End_Bearing_Fault_Data/{}".format(v['file']))
        tmp_x = []
        tmp_y = []
        for k in range(0, n, batch_size):
            mini_batches = file[v['key_name']][k:k + batch_size, :].flatten()
            tmp_x.append(np.float32(mini_batches))
            tmp_y.append(v['type']-1)
            # np.random.shuffle(tmp_x)
        if len(train_data) == 0:
            # train_data = tmp_x[0:400]
            # test_data = tmp_x[400:600]
            # train_label = tmp_y[0:400]
            # test_label = tmp_y[400:600]
            train_data = tmp_x[0:100]
            test_data = tmp_x[100:150]
            train_label = tmp_y[0:100]
            test_label = tmp_y[100:150]
        else:
            # train_data = np.vstack((train_data, tmp_x[0:400]))
            # test_data = np.vstack((test_data, tmp_x[400:600]))
            # train_label = np.hstack((train_label, tmp_y[0:400]))
            # test_label = np.hstack((test_label, tmp_y[400:600]))
            train_data = np.vstack((train_data, tmp_x[0:100]))
            test_data = np.vstack((test_data, tmp_x[100:150]))
            train_label = np.hstack((train_label, tmp_y[0:100]))
            test_label = np.hstack((test_label, tmp_y[100:150]))
    # 对训练、测试数据进行预处理
    # scaler = prep.StandardScaler()  # 标准化
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.transform(test_data)
    print (np.shape(train_data))
    return train_data, train_label, test_data, test_label

def save_bearing_data():
    path = '/Users/liulang/Desktop/WSAE_Resnext/codes/Pytorch-Deep-Neural-Networks-master/fuzz/data/Bearing'
    path='F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/data'
    batch_size = 839
    # n = 120000   #12000/800 = 150, 其中100个正常样本， 500个测试样本
    final_size = 800
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for k, v in bearing_data.items():
        if k != 'type_3':
            continue
        file = loadmat("../bearing_data/12k_Drive_End_Bearing_Fault_Data/{}".format(v['file']))
        print(file.keys())
        tmp_x = []
        tmp_y = []
        # print ('file', len(file[v['key_name']]))
        plt.plot(file[v['key_name']][0:400], color='#1f77b4', linestyle=':')
        plt.show()
        n = len(file[v['key_name']])
        for k in range(0, n, batch_size):
            mini_batches = file[v['key_name']][k:k + batch_size, :].flatten()
            tmp_x.append(np.float32(mini_batches))
            tmp_y.append(v['type']-1)
            # np.random.shuffle(tmp_x)
            # train_data = tmp_x[0:400]
            # test_data = tmp_x[400:600]
            # train_label = tmp_y[0:400]
            # test_label = tmp_y[400:600]
            # np.random.shuffle(tmp_x)
        train_data = tmp_x[0:100]
        test_data = tmp_x[100:150]
        train_label = tmp_y[0:100]
        test_label = tmp_y[100:150]
        train_data = np.array(train_data)[:, 0:800]
        test_data = np.array(test_data)[:, 0:800]
        if v['type']-1 < 10:
            train_path = path + '/train/' + 'd0{}.dat'.format(v['type'] - 1)
            test_path = path + '/test/' + 'd0{}.dat'.format(v['type'] - 1)
        else:
            train_path = path + '/train/' + 'd{}.dat'.format(v['type'] - 1)
            test_path = path + '/test/' + 'd{}.dat'.format(v['type'] - 1)
        train_file = open(train_path, 'w')
        test_file = open(test_path, 'w')
        np.savetxt(train_file, train_data, fmt='%f', delimiter='\t')
        np.savetxt(test_file, test_data, fmt='%f', delimiter='\t')

def save_drive_data():
    path = '/Users/liulang/Desktop/WSAE_Resnext/codes/Pytorch-Deep-Neural-Networks-master/fuzz/data/Drive_3'
    file = '/Users/liulang/Desktop/WSAE_Resnext/codes/Neural-Component-Analysis-master/data/drive.dat'
    data = np.loadtxt(file, dtype=np.float32)
    m, n = np.shape(data[:,:-1])
    # for i in range(12):
    #     plt.plot(np.arange(len(data[:,i])), data[:,i])
    # plt.legend(range(12))
    # plt.show()
    labels = set(data[:,-1])
    print (labels)
    for k in labels:
        data_k = []
        train_data = []
        test_data = []
        for i in range(len(data)):
            if data[i,-1] == k:
                data_k.append(data[i,:-1])
            # np.random.shuffle(data_k)
            train_data = data_k
            # train_data = data_k[0:3546]
            # test_data = data_k[3546:5319]
        if k-1 < 10:
            train_path = path + '/train/' + 'd0{}.dat'.format(int(k) - 1)
            # test_path = path + '/test/' + 'd0{}.dat'.format(int(k) - 1)
        else:
            train_path = path + '/train/' + 'd{}.dat'.format(int(k) - 1)
            # test_path = path + '/test/' + 'd{}.dat'.format(int(k) - 1)

        train_file = open(train_path, 'w')
        # test_file = open(test_path, 'w')
        np.savetxt(train_file, train_data, fmt='%f', delimiter='\t')
        train_file.close()
        # np.savetxt(test_file, test_data, fmt='%f', delimiter='\t')
        # test_file.close()

def save_UAV_data():
    path = '/Users/liulang/Desktop/WSAE_Resnext/codes/Pytorch-Deep-Neural-Networks-master/fuzz/data/UAV_15'
    file = '/Users/liulang/Desktop/WSAE_Resnext/codes/Neural-Component-Analysis-master/data/UAV_15.mat'
    raw_data = loadmat(file)
    data = raw_data['x']
    m, n = np.shape(data[:,:-1])
    # for i in range(12):
    #     plt.plot(np.arange(len(data[:,i])), data[:,i])
    # plt.legend(range(12))
    # plt.show()
    labels = set(data[:,-1])
    print (labels)
    for k in labels:
        data_k = []
        train_data = []
        test_data = []
        for i in range(len(data)):
            if data[i,-1] == k:
                data_k.append(data[i,:-1])
            # np.random.shuffle(data_k)
            train_data = data_k
            train_data = data_k[0:3000]
            test_data = data_k[3000:5000]
        if k-1 < 10:
            train_path = path + '/train/' + 'd0{}.dat'.format(int(k) - 1)
            test_path = path + '/test/' + 'd0{}.dat'.format(int(k) - 1)
        else:
            train_path = path + '/train/' + 'd{}.dat'.format(int(k) - 1)
            test_path = path + '/test/' + 'd{}.dat'.format(int(k) - 1)

        train_file = open(train_path, 'w')
        test_file = open(test_path, 'w')
        np.savetxt(train_file, train_data, fmt='%f', delimiter='\t')
        train_file.close()
        np.savetxt(test_file, test_data, fmt='%f', delimiter='\t')
        test_file.close()

def MFF():
    file = loadmat('/Users/liulang/Desktop/WSAE_Resnext/codes/Pytorch-Deep-Neural-Networks-master/fuzz/data/Multiphase_Flow_Facility/test/Training.mat')
    # print (file)
    for key in file:
        print (key)
if __name__ == "__main__":
    # MFF()
     save_UAV_data()
    # save_bearing_data()
    # a = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]
    # a = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    # b = np.array(a)
    # b = b.reshape(-1,4)
    # np.random.shuffle(b)
    # c = b.reshape(-1,2)
    # print (c)
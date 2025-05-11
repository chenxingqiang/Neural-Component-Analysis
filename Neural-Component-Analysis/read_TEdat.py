import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def read_all_data(path_test,path_train):
    '''
    读取TE过程的所有.dat数据并存人DataFrame中，输入参数为测试数据和训练数据的绝对路径
    '''
    var_name = []
    for i in range(1,42):
        var_name.append('XMEAS(' + str(i) + ')')
    for i in range(1,12):
        var_name.append('XMV(' + str(i) + ')')
    data_test, data_train = [], []

    test_join = glob.glob(os.path.join(path_test,'*.dat'))
    train_join = glob.glob(os.path.join(path_train,'*.dat'))
    for filename in test_join:
        data_test.append(pd.read_table(filename, sep = '\s+', header=None, engine='python', names = var_name))
    for filename2 in train_join:
        data_train.append(pd.read_table(filename2, sep = '\s+', header=None, engine='python', names = var_name))
    return data_test, data_train


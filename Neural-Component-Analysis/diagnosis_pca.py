import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import norm, chi2
from sklearn.preprocessing import StandardScaler

# Use standard fonts to avoid "font not found" errors
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# Use more common fonts that should be available on most systems
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = True


def t2_online(x, p, v):
    '''
    p：特征向量组成的降维矩阵，负载矩阵
    x：在线样本，shape为m*1
    v：特征值由大至小构成的对角矩阵
    '''
    T_2 = np.dot(np.dot((np.dot((np.dot(x.T, p)), np.linalg.inv(v))), p.T), x)
    return T_2

def spe_online(x, p):
    '''
    p：特征向量组成的降维矩阵，负载矩阵
    x：在线样本，shape为m*1
    '''
    I = np.eye(len(x))
    spe = np.dot(np.dot(x.T, I - np.dot(p, p.T)), x)
    # Q_count = np.linalg.norm(np.dot((I - np.dot(p_k, p_k.T)), test_data_nor), ord=None, axis=None, keepdims=False)  #二范数计算方法
    return spe

def pca_control_limit(Xtrain, ratio = 0.95, confidence = 0.99):
    '''
    计算出T2和SPE统计量
    '''
    pca = PCA(n_components = ratio)
    pca.fit(Xtrain)
    evr = pca.explained_variance_ratio_
    ev = pca.explained_variance_ # 方差，相当于X的协方差的最大的前几个特征值
    n_com = pca.n_components
    p = (pca.components_).T # 负载矩阵
    v = np.diag(ev) # 特征值组成的对角矩阵
    v_all = PCA(n_components = Xtrain.shape[1]).fit(Xtrain).explained_variance_
    p_all = (PCA(n_components = Xtrain.shape[1]).fit(Xtrain).components_).T
    k = len(evr)
    n_sample = pca.n_samples_
    ##T统计量阈值计算
    coe = k* (n_sample - 1) * (n_sample + 1) / ((n_sample - k) * n_sample)
    t_limit = coe * stats.f.ppf(confidence, k, (n_sample - k))

    ##SPE统计量阈值计算
    theta1 = np.sum((v_all[k:]) ** 1)
    theta2 = np.sum((v_all[k:]) ** 2)
    theta3 = np.sum((v_all[k:]) ** 3)
    h0 = 1 - (2 * theta1 * theta3) / (3 * (theta2 ** 2))
    c_alpha = norm.ppf(confidence)
    spe_limit = theta1 * ((h0 * c_alpha * ((2 * theta2) ** 0.5)
                           / theta1 + 1 + theta2 * h0 * (h0 - 1) / (theta1 ** 2)) ** (1 / h0))
    return t_limit, spe_limit, p, v, v_all, k, p_all

def pca_model_online(X, p, v):
    t_total = []
    q_total = []
    for x in range(np.shape(X)[0]):
        data_in = X[x]
        t = t2_online(data_in, p, v)
        q = spe_online(data_in, p)
        t_total.append(t)
        q_total.append(q)
    return t_total, q_total

def figure_control_limit(X, t_limit, spe_limit, t_total, q_total):
    ## 画控制限的图
    plt.figure(2, figsize=(12, 7))
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(t_total)
    plt.plot(np.ones((len(X))) * t_limit, 'r', label='95% $T^2$ control limit')
    # ax1.set_ylim(0,100)
    # plt.xlim(0,100)
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Hotelling $T^2$ statistic')
    plt.legend()

    ax2 = plt.subplot(2, 1, 2)
    plt.plot(q_total)
    plt.plot(np.ones((len(X))) * spe_limit, 'r', label='95% spe control limit')
    # ax1.set_ylim(0,30)
    # plt.xlim(0,100)
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('SPE statistic')
    plt.legend()
    plt.show()

def Contribution_graph(test_data, train_data, index, p, p_all, v_all, k, t_limit):
    # 贡献图
    # Using index parameter directly instead of hardcoded value
    # 1. 确定造成失控状态的得分
    # Use the passed test_data parameter instead of global variable
    data_mean = np.mean(train_data, 0)  # Using train_data parameter instead of Xtrain_nor
    data_std = np.std(train_data, 0)    # Using train_data parameter
    test_data_submean = np.array(test_data - data_mean)
    test_data_norm = np.array((test_data - data_mean) / data_std)
    
    t = test_data_norm[index,:].reshape(1, test_data.shape[1])
    S = np.dot(t, p[:,:])
    r = []
    for i in range(k):
        if S[0,i]**2/v_all[i] > t_limit/k:
            r.append(i)
    print(r)
    # 2. 计算每个变量相对于上述失控得分的贡献
    cont = np.zeros([len(r), test_data.shape[1]])
    for i in range(len(r)):
        for j in range(test_data.shape[1]):
            cont[i,j] = S[0,i]/v_all[r[i]]*p_all[r[i],j]*test_data_submean[index,j]
            if cont[i,j] < 0:
                cont[i,j] = 0
    # 3. 计算每个变量对T的总贡献
    a = cont.sum(axis = 0)
    # 4. 计算每个变量对Q的贡献
    I = np.eye(test_data.shape[1])
    e = (np.dot(test_data_norm[index,:],(I - np.dot(p, p.T))))**2

    ## 画图
    # Use standard fonts instead of SimHei
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # font1 = {'family' : 'SimHei','weight' : 'normal','size'   : 23,}
    
    # Use standard font
    font1 = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 20}
    
    plt.figure(2,figsize=(16,9))
    ax1=plt.subplot(2,1,1)
    plt.bar(range(test_data.shape[1]),a)
    plt.xlabel('Variable Index',font1)
    plt.ylabel('$T^2$ Contribution %',font1)
    plt.legend()
    plt.show
    ax1=plt.subplot(2,1,2)
    plt.bar(range(test_data.shape[1]),e)
    plt.xlabel('Variable Index',font1)
    plt.ylabel('Q Contribution %',font1)
    plt.legend()
    plt.show()


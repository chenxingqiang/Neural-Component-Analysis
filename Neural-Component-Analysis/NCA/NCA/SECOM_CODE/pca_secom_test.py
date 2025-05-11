#利用 PCA 对半导体制造数据降维
#数据集secom.data是半导体数据集,该数据集总共590个特征，要求利用pca算法对数据
#(1)对数据清洗,清洗的方法就是将每个特征里取值为null值的用该特征的平均值代替。
#(2)对清洗后的数据利用pca降维,只保留20个主成分。得到降维后的数据，以及前20个主成分对应的方差占总方差的
#百分比并画出示意图。注：请使用sklearn中的pca进行降维


import numpy as np
np.set_printoptions(suppress=True)
data=np.genfromtxt('F:\\0002025_bishe\\code\\Neural-Component-Analysis-master\\Neural-Component-Analysis-master\\NCA\\NCA\\secom_data\\secom.data')
n,m=data.shape
print(m)
print(n)
print('处理前',data)
for i in range(m):
    inx=np.nonzero(~np.isnan(data[:,i]))
    meanVal=np.mean(data[inx,i])
    data[np.nonzero(np.isnan(data[:,i])),i]=meanVal
print('处理后',data)



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca=PCA(n_components=20)
pca.fit(data)
ratio=pca.explained_variance_ratio_
print(ratio)

x=np.linspace(1,20,num=20)
plt.plot(x,ratio)
print(x)
plt.scatter(x,ratio,marker='^',c='r')
plt.xticks(x)
plt.show()

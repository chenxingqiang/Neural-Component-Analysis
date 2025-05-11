from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import numpy as np

def loadDataSet(filename, delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)

dataMat = loadDataSet('F:\0002025_bishe\code\Neural-Component-Analysis-master\Neural-Component-Analysis-master\NCA\NCA\secom_data\SECOM.TXT')
print(shape(dataMat))

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax1 = Axes3D(fig)
ax1.scatter3D(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0],dataMat[:,2].flatten().A[0], cmap='Blues')  #绘制散点图
plt.show()

pca = PCA(n_components=2)
pca.fit(dataMat)
new_data=pca.transform(dataMat)


print(shape(new_data))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(new_data[:,0], new_data[:,1],marker='^',s=90)
plt.show()

#在很多应用中，当我们将数据降维并用于训练后，训练出来的模型之后的输出也是降维后的数据，
#需要还原回原始维度。这时候需要将pca算法进行逆运算：
old_data=np.dot(new_data,pca.components_)+pca.mean_
fig=plt.figure()
ax1 = Axes3D(fig)
ax1.scatter3D(old_data[:,0],old_data[:,1],old_data[:,2], cmap='Blues')  #绘制散点图
plt.show()

pca = PCA(n_components=1)
pca.fit(dataMat)
new_data=pca.transform(dataMat)


print(shape(new_data))


#在很多应用中，当我们将数据降维并用于训练后，训练出来的模型之后的输出也是降维后的数据，
#需要还原回原始维度。这时候需要将pca算法进行逆运算：
old_data=np.dot(new_data,pca.components_)+pca.mean_
fig=plt.figure()
ax1 = Axes3D(fig)
ax1.scatter3D(old_data[:,0],old_data[:,1],old_data[:,2], cmap='Blues')  #绘制散点图
plt.show()
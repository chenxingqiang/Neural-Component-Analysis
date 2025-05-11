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

def replaceNanWithMean():
    datMat = loadDataSet('F:\0002025_bishe\code\Neural-Component-Analysis-master\Neural-Component-Analysis-master\NCA\NCA\secom_data\secom.data',' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

dataMat = replaceNanWithMean()

pca = PCA(n_components=6)
pca.fit(dataMat)
new_data=pca.transform(dataMat)

print(shape(new_data))
print(new_data)

pca = PCA(n_components=20)
pca.fit(dataMat)
new_data=pca.transform(dataMat)

print(shape(new_data))
print(new_data)
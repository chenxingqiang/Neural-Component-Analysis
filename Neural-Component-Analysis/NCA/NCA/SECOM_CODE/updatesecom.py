from pca import *
from numpy import *

dataMat = replaceNanWithMean()
meanVals = mean(dataMat,axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved,rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))

print(shape(eigVals))
print(eigVals)

print(shape(eigVects))
print(eigVects)

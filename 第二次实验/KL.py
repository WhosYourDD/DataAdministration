import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#样本矩阵中心化,得到中心化矩阵，平均值矩阵
def center(matrix):
    rows,cols=matrix.shape
    average=np.mean(matrix,axis=0)
    average=np.tile(average,(rows,1))
    matrix=matrix-average
    return matrix,average

#确定k值
def getK(eigVals,percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num

#选取最大的p个特征值
def maxEig(matrix,p):
    #得到特征值和特征向量
    E,V=np.linalg.eig(matrix)
    #确定k值
    k=getK(E,p)
    print("得到",k,"维矩阵")
    eigenvalue = np.argsort(E)
    K_eigenValue = eigenvalue[-1:-(k+1):-1]
    K_eigenVector = V[:,K_eigenValue]
    return K_eigenValue, K_eigenVector

#得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
    #return np.dot(DataMat,K_eigenVector)
    return DataMat * K_eigenVector

def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    #reconDataMat = np.dot(K_eigenVector.T,lowDataMat) + meanVal
    return reconDataMat

if __name__== "__main__":
    #保留信息占比
    p=0.8
    data=pd.read_csv('/Applications/Projects/ColorHistogram.asc',header=None,index_col=0,sep=" ")
    data_array = np.float32(np.mat(data))
    #样本矩阵的中心化
    matrix,average=center(data_array)
    #计算样本矩阵的协方差矩阵
    cov_array=np.cov(matrix,rowvar=0)
    #协方差矩阵的对角化，选取最大的p个特征值，对应的特征向量组成投影矩阵
    D,V = maxEig(cov_array, p)
    #得到降维后的数据
    lowDataMat = getlowDataMat(matrix, V)
    reconDataMat = Reconstruction(lowDataMat, V, average)
    print(reconDataMat)

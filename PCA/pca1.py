import numpy as np

# ? 寻找特征向量和特征值的模块linalg，eig()求解特征值和特征现象


def createDataset():
    dataset = np.array([[-5, -5], [-5, -4], [-4, -5], [-5, -6],
                       [-6, -5], [5, 5], [5, 4], [4, 5], [5, 6], [6, 5]])  # ? 10*2
    return dataset

# ? (数据集[, N个特征])

def pca(dataMat, topNFeat=999999):
    # ? 1.去平均值 data-preprocessing
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # ? 10*2

    covMat = np.cov(meanRemoved, rowvar=0)  # ? 协方差  2*2
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # ? 特征值、特征向量

    # ? 2.从小到大对N个值排序，根据特征值排序就可以得到topNfeat个最大的特征向量
    eigValInd = np.argsort(eigVals)  # ? 获得特征值从小到大的索引
    eigValInd = eigValInd[: -(topNFeat + 1): -1]
    redEigVects = eigVects[:, eigValInd]  # ? 获得投影的k个向量

    # ? 3.将数据转换到新空间
    # ? 特征向量构成对数据进行转换的矩阵，该矩阵则利用N个特征将原始数据转换成新空间了
    lowDDataMat = meanRemoved * redEigVects  # ? (10*2) * (2*1) => 降到1维
    reconMat = (lowDDataMat * redEigVects.T) + meanVals

    return lowDDataMat, reconMat


def main():
    dataset = createDataset()
    lowDDataMat, reconMat = pca(dataset, 1)


main()

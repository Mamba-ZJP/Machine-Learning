import numpy as np  # 科学计算包numpy
import operator  # 运算符模块

# 1. 设置数据集：
# 读文件，设置dataset为4维的（后四列），label为第1列

# 2.对后75个数据集循环分类
# 1) 计算距离
# 2) 选择距离最近的k个点
# 3) 排序

def createDataset():
    fileObject = open("E:\Code\Machine-Learning\KNN\iris.txt", 'r')
    trainDataset = np.empty(4)
    print(trainDataset)
    for line in fileObject:
        trainDataset.append(line)

    return trainDataset 


def classify(inX, dataset, labels, k):
    # 1.计算距离
    datasetSize = dataset.shape[0]  # 这里应该是4维
    # 这里矩阵都支持乘方和减运算
    diffMat = tile(inX, (datasetSize, 1)) - dataset # 将分类向量纵向捕开成训练数据一样的大小，算出差值矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5

    # 2.选择距离最小的 k 个点
    sortedDistIndices = distances.argsort() # 返回indices 可以升序排序数组的
    classCount = dict()
    for i in range(k):
        cntLabel = labels[sortedDistIndices[i]]
        classCount[cntLabel] = classCount.get(cntLabel, 0) + 1

    # 3.sort 反向排序 多的在前
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)

    return sortedClassCount[0][0]

def main():
    trainDataset = createDataset()

main()
    

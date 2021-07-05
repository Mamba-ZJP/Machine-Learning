from numpy import *  # 科学计算包numpy
import operator  # 运算符模块


def createDataSet():
    group = array( [[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]] )  # array( [ele1, ele2, ele3] )
    labels = ['A', 'A', 'B', 'B']
    # print(group.ndim)
    return group, labels

def classify(inX, dataSet, labels, k):  # (分类的输入向量)
    # 1.计算距离
    dataSetSize = dataSet.shape[0]  # shape获取维数大小，shape[0]表示第一维的大小
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile 这里是纵向铺开，然后矩阵数值直接相减
    sqDiffMat = diffMat ** 2  # 直接差值矩阵数值平方
    sqDistances = sqDiffMat.sum(axis = 1)  # 按axis求和,一维数组 ？
    distances = sqDistances ** 0.5
    
    # 2.选择距离最小的k个点
    sortedDistIndices = distances.argsort()  # return the indices that would sort an array
    classCount = {} # dict() 
    for i in range(k):
        cntLabel = labels[sortedDistIndices[i]]  # 获取对应的label
        classCount[cntLabel] = classCount.get(cntLabel, 0) + 1  # get(key[, default])
    
    # 3.Sort 
    # py内置：(, key, reverse = true => 反向排序) return a new sorted list
    #   itemgetter(k)相当于访问item（list）的第k个元素
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    print(classCount.items())
    return sortedClassCount[0][0]



def main():
    group, labels = createDataSet()
    print(classfiy([2, 2], group, labels, 3))

main()

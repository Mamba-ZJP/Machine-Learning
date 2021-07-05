from numpy import *  # 科学计算包numpy
import operator  # 运算符模块


def createDataSet():
    group = array( [[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]] )  # array( [ele1, ele2, ele3] )
    labels = ['A', 'A', 'B', 'B']
    # print(group.ndim)
    return group, labels

def classfiy(inX, dataSet, labels, k):  # (分类的输入向量)
    # 计算距离
    dataSetSize = dataSet.shape[0]  # shape获取维数大小，shape[0]表示第一维的大小
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile 这里是纵向铺开，然后矩阵数值直接相减
    sqDiffMat = diffMat ** 2  # 直接差值矩阵数值平方
    sqDistances = sqDiffMat.sum(axis = 1)  # 按axis求和,一维数组 ？
    distances = sqDistances ** 0.5
    
    #选择距离最小的k个点
    sortedDistIndices = distances.argsort()  # return the indices that would sort an array
    classCount = {} # dict() 
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]  # 获取对应的label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # get(key[, default])
    
    #Sort 
    # py内置：(, key, reverse = true => 反向排序) return a new sorted list
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)

    return sortedClassCount[0][0]



def main():
    group, labels = createDataSet()
    print(classfiy([2, 2], group, labels, 3))

main()

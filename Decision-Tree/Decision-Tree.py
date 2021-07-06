import numpy as np
from math import log
import operator


def createDataset():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no', ], [0, 1, 'no']]
    labels = ['no sufrfacing', 'flippers']
    return dataset, labels

# 计算信源熵：entropy 熵
# 获得信息增益（熵）最高的特征就是最好的选择


def calShannonEnt(dataset):
    numEntries = len(dataset)

    # 为所有可能分类创建字典
    labelCounts = dict()
    for featVec in dataset:
        cntLabel = featVec[-1]  # key(label) 为矩阵最后一个元素
        if cntLabel not in labelCounts.keys():
            labelCounts[cntLabel] = 0
        labelCounts[cntLabel] += 1

    # 计算熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算每个分类（label/key）出现的改率
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


#  (数据集，特征，需要返回的特征的值):根据特征划分数据集
# py 引用传参
# 相当于只分出一条枝来，将dataset中 feature==value 的数据返回
def splitDataset(dataset, axis, value):
    # 创建新的list对象
    retDataset = []

    # 抽取
    for featVec in dataset:
        if featVec[axis] == value:  # 需要判断的特征 == 返回的特征值
            reducedFeatVec = featVec[:axis]  # 判断特征前的特征
            reducedFeatVec.extend(featVec[axis + 1:])   # 判断特征后的特征 全部加进减少的特征向量中
            retDataset.append(reducedFeatVec)

    return retDataset


# 选择最好的特征值
def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1  # 属性特征的数量
    baseEntropy = calShannonEnt(dataset)  # 计算原始香农码
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):  # 遍历每个特征
        # ? 1.创建唯一的分类标签列表
        # list comprehension 列表推导,拿到所有样本的第i个特征值
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)  # 拿到不同值
        newEntropy = 0.0

        # ? 2.计算每种划分方式的信息熵
        for val in uniqueVals:
            subDataset = splitDataset(dataset, i, val)  # 拿到splited的数据集
            prob = len(subDataset) / float(len(dataset))
            newEntropy += prob * calShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy  # 计算熵的减少（无序度的下降）

        # ? 3.计算最好的信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


# 返回次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    # * [[名称， 次数]] 返回最多分类的名称
    return sortedClassCount[0][0]

# 递归构建决策树


def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]  # 将dataset中的标签导入一个list

    # ? 1.类别相同，停止划分
    if classList.count(classList[0]) == len(classList):  # 全是同一个标签
        return classList[0]

    # ? 2.遍历完所有特征时返回出现次数最多的类别
    if len(dataset[0]) == 1:  # 没有特征只剩下类别标签了
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataset)  # 获取最好的特征
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])  # 在labels删掉最好的那个feature，然后递归调用（因为下一次要换feature分类了）

    # ? 3.得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataset]  # 每个样本的feacture value
    uniqueVals = set(featValues)  # 肯定要用同样的feature不同的值就做分类

    for val in uniqueVals:  # 根据不同的feature value去split，分支
        subLabels = labels[:]
        myTree[bestFeatLabel][val] = createTree( splitDataset(dataset, bestFeat, val), subLabels )

    return myTree


def main():
    dataset, labels = createDataset()
    print(createTree(dataset, labels))


main()

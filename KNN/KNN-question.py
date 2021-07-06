import numpy as np  # 科学计算包numpy
import operator  # 运算符模块

# 1. 设置数据集：
# 读文件，设置dataset为4维的（后四列），label为第1列

# 2.对后75个数据集循环分类
# 1) 计算距离
# 2) 选择距离最近的k个点
# 3) 排序

def createDataset():
    with open("E:\Code\Machine-Learning\KNN\iris.txt", 'r') as fileObject:
        read = fileObject.read()
        lines = read.split('\n')
        dataset, labels, trainLables = [], [], []
        trainDataset, classifyDataset = [], []
        # trainDataset = [list(map(float, each[:-1].split(" "))) for each in list(fileObject)]

        for line in lines:
           dataset.append(list(map(float, line.split(" "))))
        
        labels = [each[0] for each in dataset]
        dataset = [each[1:] for each in dataset]

        trainDataset.extend(dataset[0:25])
        trainLables.extend(labels[0:25])
        classifyDataset.extend(dataset[25: 50])

        trainDataset.extend(dataset[50: 75])
        trainLables.extend(labels[50: 75])
        classifyDataset.extend(dataset[75: 100])

        trainDataset.extend(dataset[100:125])
        trainLables.extend(labels[100: 125])
        classifyDataset.extend(dataset[125: 150])
        print("run")

    return trainDataset, classifyDataset, trainLables


def classify(inX, dataset, labels, k):
    # 1.计算距离
    datasetSize = dataset.shape[0]  # 这里应该是4维
    # 这里矩阵都支持乘方和减运算
    diffMat = np.tile(inX, (datasetSize, 1)) - dataset # 将分类向量纵向捕开成训练数据一样的大小，算出差值矩阵
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
    trainDataset, classifyDataset, labels = createDataset()
    npLabels = np.array(labels)
    npTrainDataset = np.array(trainDataset)
    npClassifyDataset = np.array(classifyDataset)

    for data in npClassifyDataset:
        res = classify(data, npTrainDataset, npLabels, 3)
        print(res)

main()
    

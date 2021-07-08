import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt

dataset = []

def main():
    dataset_1 = [[5.1418, 0.595], [5.5519, 3.5091], [5.3836, 2.8033], [3.2419, 3.7278], [4.4427, 3.8981], [4.9111, 2.871], [2.9259, 3.4879], [4.2018, 2.4973], [4.7629, 2.5163], [2.7118, 2.4264], [3.047, 1.5699], [4.7782, 3.3504], [3.9937, 4.8529], [4.5245, 2.1322], [5.3643, 2.2477], [4.482, 4.0843], [3.2129, 3.0592], [4.752, 5.3119], [3.8331, 0.4484], [3.1838, 1.4494], [6.0941, 1.8544], [4.0802, 6.2646], [
        3.0627, 3.6474], [4.6357, 2.3344], [5.682, 3.045], [4.5936, 2.5265], [4.7902, 4.4668], [4.1053, 3.0274], [3.8414, 4.2269], [4.8709, 4.0535], [3.8052, 2.6531], [4.0755, 2.8295], [3.4734, 3.1919], [3.3145, 1.8009], [3.7316, 2.6421], [2.8117, 2.8658], [4.2486, 1.4651], [4.1025, 4.4063], [3.959, 1.3024], [1.7524, 1.9339], [3.4892, 1.2457], [4.2492, 4.5982], [4.3692, 1.9794], [4.1792, 0.4113], [3.9627, 4.2198]]
    dataset_2 = [[9.7302, 5.508], [8.8067, 5.1319], [8.1664, 5.2801], [6.9686, 4.0172], [7.0973, 4.0559], [9.4755, 4.9869], [9.3809, 5.3543], [7.2704, 4.1053], [8.9674, 5.8121], [8.2606, 5.1095], [7.5518, 7.7316], [7.0016, 5.4111], [8.3442, 3.6931], [5.8173, 5.3838], [6.1123, 5.4995], [10.418, 4.4892], [7.9136, 5.2349], [11.154, 4.4022], [7.708, 5.0208], [8.2079, 5.4194], [9.1078, 6.1911], [7.7857, 5.7712], [7.374, 2.3558], [9.7184, 5.2854], [6.9559, 5.8261], [8.9691, 4.9919], [7.3872, 5.8584], [
        8.8922, 5.7748], [9.0175, 6.3059], [7.0041, 6.2315], [8.6396, 5.9586], [9.2394, 3.3455], [6.7376, 4.0096], [8.4345, 5.6852], [7.9559, 4.0251], [6.5268, 4.3933], [7.6699, 5.6868], [7.8075, 5.02], [6.6997, 6.0638], [5.6549, 3.659], [6.9086, 5.4795], [7.9933, 3.366], [5.9318, 3.5573], [9.5157, 5.2938], [7.2795, 4.8596], [5.5233, 3.8697], [8.1331, 4.7075], [9.7851, 4.4175], [8.0636, 4.1037], [8.1944, 5.2486], [7.9677, 3.5103], [8.2083, 5.3135], [9.0586, 2.9749], [8.2188, 5.529], [8.9064, 5.3435]]
    dataset = [each + [1] for each in dataset_1] + [each + [2]
                                                    for each in dataset_2]

    dataset, label = np.split(dataset, (2,), axis=1)
    trainDataset, testDataset, trainLabel, testLabel = train_test_split(
        dataset, label, random_state=1, train_size=0.65)

    clf = svm.SVC(C=0.8, kernel='rbf', gamma=12, decision_function_shape='ovr')
    clf.fit(trainDataset, trainLabel.ravel())

    print('训练集的精度：' + str(clf.score(trainDataset, trainLabel)))  # ? 精度 训练集肯定为1.0
    y_hat = clf.predict(trainDataset)
    # show_accuracy(y_hat, trainLabel, '训练集')

    print('分类集的精度：' + str(clf.score(testDataset, testLabel)))
    y_hat = clf.predict(testDataset)
    # show_accuracy(y_hat, testLabel, '测试集')

    x1_min, x1_max = dataset[:, 0].min(), dataset[:, 0].max()  
    x2_min, x2_max = dataset[:, 1].min(), dataset[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    grid_hat = clf.predict(grid_test)       # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['black', '#FFA0A0', 'white'])
    cm_dark = mpl.colors.ListedColormap(['r', 'r', 'b'])
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=label, edgecolors='k', s=50, cmap=cm_dark)  
    plt.scatter(testDataset[:, 0], testDataset[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
    plt.xlabel(u'Feature One', fontsize=10)
    plt.ylabel(u'Feature Two', fontsize=10)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'二特征分类', fontsize=13)
    # plt.grid()
    plt.show()

main()

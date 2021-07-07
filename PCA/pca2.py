import numpy as np


def createDataset():
    dataset_1 = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0]])
    dataset_2 = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]])
    return dataset_1, dataset_2



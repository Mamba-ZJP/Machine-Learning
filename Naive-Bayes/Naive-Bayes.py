from typing import ClassVar


def loadDataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']]
    ClassVec = [0, 1]  #? 1 is absuive, 0 not
    return postingList, ClassVec


def createVocabList(dataset):
    vocabSet = set([])
    for doc in dataset:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in inputSet:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return returnVec

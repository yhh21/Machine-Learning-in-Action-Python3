# -*- coding: utf-8 -*-

from MyPath import *

from numpy import *
from FPTreeBuild import *
from MineFPTree import *


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


if __name__ == '__main__':
    simpDat = loadSimpDat()
    print(simpDat)

    initSet = createInitSet(simpDat)
    print(initSet)

    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()

    condPats = findPrefixPath('x', myHeaderTab['x'][1])
    print(condPats)

    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)


    parsedDat = [line.split() for line in open(PROJECT_DATA_PATH + 'kosarak.dat').readlines()]
    initSet = createInitSet(parsedDat)
    myFPtree, myHeaderTab = createTree(initSet, 100000)

    myFreqList = []
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
    print(myFreqList)
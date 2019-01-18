# -*- coding: utf-8 -*-

from MyPath import *

from numpy import *
from AprioriAlgorithmHelper import *
from AprioriAlgorithm import *
from AssociationRuleGeneration import *


if __name__ == '__main__':
    dataSet = loadDataSet()

    L, suppData = apriori(dataSet)
    #L, suppData = apriori(dataSet, 0.7)
    print(L)

    rules = generateRules(L, suppData, minConf=0.7)
    #rules = generateRules(L, suppData, minConf=0.5)
    print(rules)


    mushDatSet = [line.split() for line in open(PROJECT_DATA_PATH + 'mushroom.dat').readlines()]
    L, suppData = apriori(mushDatSet, minSupport=0.3)

    for item in L[1]:
        if item.intersection('2'):
            print(item)
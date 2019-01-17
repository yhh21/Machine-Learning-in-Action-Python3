# -*- coding: utf-8 -*-

from MyPath import *

from numpy import *
from CARTTreeBuilding import *
from RegressionTreePruning import *
from LeafGeneration import *
from ForecastTreeBaseRegression import *


if __name__ == '__main__':
    testMat=mat(eye(4))
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)

    myMat = mat(loadDataSet(PROJECT_DATA_PATH + 'ex00.txt'))
    #myMat = mat(loadDataSet(PROJECT_DATA_PATH + 'ex0.txt'))
    #myMat = mat(loadDataSet(PROJECT_DATA_PATH + 'ex2.txt'))
    #myMat = mat(loadDataSet(PROJECT_DATA_PATH + 'exp2.txt'))

    tree = createTree(myMat)
    #tree = createTree(myMat, ops=(10000,4))
    #tree = createTree(myMat, ops=(0,1))
    #tree = createTree(myMat, modelLeaf, modelErr, (1, 10))

    print(tree)

    '''
    myDatTest = loadDataSet(PROJECT_DATA_PATH + 'ex2test.txt')

    myMatTest = mat(myDatTest)
    prune(tree, myMatTest)
    print(tree)
    '''

    trainMat = mat(loadDataSet(PROJECT_DATA_PATH + 'bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet(PROJECT_DATA_PATH + 'bikeSpeedVsIq_test.txt'))

    myTree = createTree(trainMat, ops=(1,20))
    #myTree = createTree(trainMat, modelLeaf, modelErr, (1,20))

    yHat = createForeCast(myTree, testMat[:,0])
    #yHat = createForeCast(myTree, testMat[:,0], modelTreeEval)
    
    '''
    ws, X, Y = linearSolve(trainMat)
    m, n = shape(testMat)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i, 0] = testMat[i,0] * ws[1,0] + ws[0,0]
    '''

    print(corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])
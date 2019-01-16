# -*- coding: utf-8 -*-

from numpy import *
from RegressionTreePruning import *

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
       return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
           return treeForeCast(tree['left'], inData, modelEval)
        else:
           return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
           return treeForeCast(tree['right'], inData, modelEval)
        else:
           return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
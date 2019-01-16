# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def Plot_BestFitLine(xArr, yArr, ws) :
    xMat=mat(xArr)
    yMat=mat(yArr)
    yHat = xMat*ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.title("Data from ex0.txt with a best-fit line fitted to the data")
    plt.show()

def Plot_LWLR(xArr, yArr, yHat):
    xMat=mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0] , s=2, c='red')
    plt.title("Plot showing locally weighted linear regression")
    plt.show()
    
def Plot_RidgeRegression(ridgeWeights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.title("Regression coefficient values while using ridge regression")
    plt.show()
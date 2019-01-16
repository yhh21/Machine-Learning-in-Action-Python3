# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:02:16 2018

@author: Administrator
"""

from MyPath import *
from DrawPlot import *

from Build_model import *
from Forward_stepReg import *
from Local_weightLR import *
from Price_predict import *
from Ridge_regre import *
from Stand_Linear import *

def PredictAgeOfAbaloneByLWLE(abX, abY) : 
    yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    print(rssError(abY[0:99],yHat01.T))
    yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    print(rssError(abY[0:99],yHat1.T))
    yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
    print(rssError(abY[0:99],yHat10.T))

    yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
    print(rssError(abY[100:199],yHat01.T))
    yHat1 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
    print(rssError(abY[100:199],yHat1.T))
    yHat10 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
    print(rssError(abY[100:199],yHat10.T))

    ws =  standRegres(abX[0:99],abY[0:99])
    yHat = mat(abX[100:199])*ws
    print(rssError(abY[100:199],yHat.T.A))

def GetLeastSquaresWeights(xArr, yArr) :
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xMat = regularize(xMat)
    yM = mean(yMat,0)
    yMat = yMat - yM
    weights = standRegres(xMat,yMat.T)
    print(weights.T)


if __name__ == '__main__':
    xArr, yArr = loadDataSet(PROJECT_PATH + 'ex0.txt')

    ws = standRegres(xArr, yArr)
    Plot_BestFitLine(xArr, yArr, ws)

    print(lwlr(xArr[0],xArr,yArr,1.0))
    #print(lwlr(xArr[0],xArr,yArr,0.001))
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    Plot_LWLR(xArr, yArr, yHat)
    
    abX, abY = loadDataSet(PROJECT_PATH + 'abalone.txt')

    PredictAgeOfAbaloneByLWLE(abX, abY)

    ridgeWeights = ridgeTest(abX, abY)
    Plot_RidgeRegression(ridgeWeights)

    stageWiseWeights = stageWise(abX, abY, 0.01, 200)
    #stageWiseWeights = stageWise(abX, abY, 0.001, 5000)

    GetLeastSquaresWeights(abX, abY)
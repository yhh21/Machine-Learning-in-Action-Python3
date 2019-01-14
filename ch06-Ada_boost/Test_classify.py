# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:59:35 2018

@author: Administrator
"""

from MyPath import *

from Stump_classify import *
from Train_adaboost import *
from numpy import *

from ROC_plot import *


#测试adaBoost，adaBoost分类函数
#@datToClass:测试数据点
#@classifierArr：构建好的最终分类器
def adaClassify(datToClass,classifierArr):
    #构建数据向量或矩阵
    dataMatrix=mat(datToClass)
    #获取矩阵行数
    m=shape(dataMatrix)[0]
    #初始化最终分类器
    aggClassEst=mat(zeros((m,1)))
    #遍历分类器列表中的每一个弱分类器
    for i in range(len(classifierArr)):
        #每一个弱分类器对测试数据进行预测分类
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                classifierArr[i]['thresh'],
                                classifierArr[i]['ineq'])
        #对各个分类器的预测结果进行加权累加
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print('aggClassEst',aggClassEst)
    #通过sign函数根据结果大于或小于0预测出+1或-1
    return sign(aggClassEst)

def loadDataSet(filename):
    #创建数据集矩阵，标签向量
    dataMat=[]
    labelMat=[]
    #获取特征数目(包括最后一类标签)
    #readline():读取文件的一行
    #readlines:读取整个文件所有行
    numFeat=len(open(filename).readline().split('\t'))
    #打开文件
    fr=open(filename)
    #遍历文本每一行
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        #数据矩阵
        dataMat.append(lineArr)
        #标签向量
        #labelMat.append(float(curLine[-1]))
        # {0, 1} to {-1, 1}
        labelMat.append(float(curLine[-1]) * 2 - 1)
    return dataMat,labelMat

#训练和测试分类器
def classify():
    #利用训练集训练分类器
    datArr,labelArr = loadDataSet(PROJECT_PATH + 'horseColicTraining.txt')
    #得到训练好的分类器
    classifierArr, aggClassEst = adaBoostTrainDS(datArr,labelArr, 10)
    plotROC(aggClassEst.T, labelArr)
    #利用测试集测试分类器的分类效果
    testArr,testLabelArr = loadDataSet(PROJECT_PATH + 'horseColicTest.txt')
    prediction = adaClassify(testArr,classifierArr)
    #输出错误率
    num=shape(mat(testLabelArr))[1]
    errArr=mat(ones((num,1)))
    error=errArr[prediction!=mat(testLabelArr).T].sum()
    print("the errorRate is: %.2f" % (float(error)/float(num)))
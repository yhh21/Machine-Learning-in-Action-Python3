# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:54:15 2018

@author: Administrator
"""

from MyPath import *

from numpy import *
from load_data import *
from smo_simp import *
from Opt_smo import *
from SMO_platt import *
from Kernel import *
from image_recog import *

if __name__ == '__main__':
    data, label = loadDataSet(PROJECT_PATH + 'testSet.txt')
    print(label)

    #b, alpha = smoSimple(data, label, 0.6, 0.001, 40)
    b, alpha = smoP(data, label, 0.6, 0.001, 40)

    print(b)
    print(alpha[alpha > 0])
    for i in range(len(label)) :
        if alpha[i] > 0 :
            print(data[i], label[i])

    testRbf()

    # 下面那行代码运行时间非常非常久
    #testDigits()
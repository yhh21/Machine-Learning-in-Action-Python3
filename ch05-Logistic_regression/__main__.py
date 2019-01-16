# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:05:29 2018

@author: Administrator
"""

from MyPath import *

from numpy import *
from Grad_descent import *
from Plot_boundary import *
from matplotlib import *
from Random_GDS import *
from Logistic_classify import *

if __name__ == '__main__':
    data, label = loadData(PROJECT_DATA_PATH + 'testSet.txt')

    weights = Grad_descent(data, label)
    print(weights)
    plot_fit(data, label, weights)

    weights = Stoch_gdescent(data, label)
    # 这行代码耗时比较久
    #weights = Stoch_gdescent(data, label, 500)
    print(weights)
    plot_fit(data, label, weights)
    
    # 这行代码耗时比较久，可以单独运行
    #multTest()
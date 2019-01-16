# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:10:01 2018

@author: Administrator
"""

from MyPath import *

from word_vector import *
from Train_bayes import *
from classifier_test import *
from Mail_filter import *
#import feedparser
from RSS_filter import *

if __name__ == "__main__":

    wordlist, classlist = loadDataSet()
    
    myVoc = Create_wordVec(wordlist)
    print(myVoc)
    print(Words2Vec(myVoc, wordlist[0]))
    
    train_matrix = []
    for doc in wordlist:
        train_matrix.append(Words2Vec(myVoc, doc))
    
    print(train_matrix)
    p1,p2,p_pos = train_bayes(train_matrix, classlist)
    print(p1,p2,p_pos)
    
    Test_classify()

    sum_error_rate = 0.0
    loop_num = 10
    for i in range(loop_num) :
        sum_error_rate += Spam_filter(PROJECT_DATA_PATH + 'email')
    print("the average error rate : ", sum_error_rate / float(loop_num))
    
    '''
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    a, b, c = localWords(ny,sf)
    print(a,b,c)
    '''
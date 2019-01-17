# -*- coding: utf-8 -*-

from MyPath import *

from numpy import *
from KMeansSup import *
from KMeansClustering import *
from BisectingKMeansClustering import *
from TestPlaceKMeansClustering import *

if __name__ == '__main__':
    datMat = mat(loadDataSet(PROJECT_DATA_PATH + 'testSet.txt'))

    myCentroids, clustAssing = kMeans(datMat, 4)
    
    datMat = mat(loadDataSet(PROJECT_DATA_PATH + 'testSet2.txt'))
    centList, myNewAssments = biKmeans(datMat, 3)

    clusterClubs(5)
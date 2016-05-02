__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import numpy as np
from libs.mrqap import MRQAP
import time

#######################################################################
# Functions
#######################################################################
NCOUNTRIES = 249
def getMatrix(path, log=False):
    matrix = np.zeros(shape=(NCOUNTRIES,NCOUNTRIES))
    with open(path, 'rb') as f:
        for line in f:
            data = line.split(' ')
            c1 = int(data[0])-1
            c2 = int(data[1])-1
            v = np.log1p(float(data[2])) if log else float(data[2])
            matrix[c1][c2] = v # real data from file
            matrix[c2][c1] = v # symmetry
    print '{} loaded as a matrix!'.format(path)
    return matrix

start_time = time.time()
#######################################################################
# Data Matrices
#######################################################################
X1 = getMatrix('data/country_trade_index.txt',True)
X2 = getMatrix('data/country_distance_index.txt',True)
X3 = getMatrix('data/country_colonial_index.txt')
Y  = getMatrix('data/country_lang_index.txt')
X = {'TRADE':X1, 'DISTANCE':X2, 'COLONIAL':X3}

#######################################################################
# QAP
#######################################################################
mrqap = MRQAP(Y, X)
mrqap.mrqap(npermutations=2000, diagonal=False)
mrqap.summary()
print("--- %s seconds ---" % (time.time() - start_time))
#mrqap.plot()
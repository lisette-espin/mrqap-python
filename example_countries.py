__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import numpy as np
from libs.mrqap import MRQAP
import time

#######################################################################
# Constants
#######################################################################
NCOUNTRIES = 249
DIRECTED = True
NPERMUTATIONS = 2000

#######################################################################
# Functions
#######################################################################
def getMatrix(path, directed=False, log1p=False):
    matrix = np.zeros(shape=(NCOUNTRIES,NCOUNTRIES))
    with open(path, 'rb') as f:
        for line in f:
            data = line.split(' ')
            c1 = int(data[0])-1
            c2 = int(data[1])-1
            v = np.log1p(float(data[2])) if log1p else float(data[2])
            matrix[c1][c2] = v # real data from file
            if not DIRECTED:
                matrix[c2][c1] = v # symmetry
    print '{} loaded as a matrix!'.format(path)
    return matrix

#######################################################################
# Data Matrices
#######################################################################
X1 = getMatrix('data/country_trade_index.txt',DIRECTED,True)
X2 = getMatrix('data/country_distance_index.txt',DIRECTED,True)
X3 = getMatrix('data/country_colonial_index.txt',DIRECTED)
Y  = getMatrix('data/country_lang_index.txt',DIRECTED)
X = {'TRADE':X1, 'DISTANCE':X2, 'COLONIAL':X3}
Y = {'LANG':Y}
np.random.seed(1)

#######################################################################
# QAP
#######################################################################
start_time = time.time()
mrqap = MRQAP(Y=Y, X=X, npermutations=NPERMUTATIONS, diagonal=False, directed=True)
mrqap.mrqap()
mrqap.summary()
print("--- {}, {}: {} seconds ---".format('directed' if DIRECTED else 'undirected', NPERMUTATIONS, time.time() - start_time))
mrqap.plot('betas')
mrqap.plot('tvalues')

__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import numpy as np
from libs.mrqap import MRQAP
import time
from pandas.core.series import Series

#######################################################################
# Constants
#######################################################################
N = 50
DIRECTED = True
NPERMUTATIONS = 2000

#######################################################################
# Data Matrices
#######################################################################
Y  = np.loadtxt('data-rg/data.matrix',delimiter=',')
X1 = np.loadtxt('data-rg/noise-1.0.matrix',delimiter=',')
X2 = np.loadtxt('data-rg/noise-5.0.matrix',delimiter=',')
X3  = np.loadtxt('data-rg/noise-10.0.matrix',delimiter=',')
X4  = np.loadtxt('data-rg/noise-100.0.matrix',delimiter=',')
X5  = np.loadtxt('data-rg/noise-1000.0.matrix',delimiter=',')
X6  = np.loadtxt('data-rg/Erdos-Renyi-p0.5.matrix',delimiter=',')
X7  = np.loadtxt('data-rg/Erdos-Renyi-p1.0.matrix',delimiter=',')
X8  = np.loadtxt('data-rg/Geometric-r1.0.matrix',delimiter=',')
X9  = np.loadtxt('data-rg/Barabassi-m49.matrix',delimiter=',')
X10  = np.loadtxt('data-rg/uniform-0.02.matrix',delimiter=',')

X = {'NOISE1':X1, 'NOISE5':X2, 'NOISE10':X3, 'NOISE100':X4, 'NOISE1000':X5,'ERDOS05':X6, 'ERDOS1':X7, 'GEOMETRIC1':X8, 'BARABASI49':X9, 'UNIFORM':X10}
Y = {'DATA':Y}
np.random.seed(1)

#######################################################################
# QAP
#######################################################################
start_time = time.time()
mrqap = MRQAP(Y=Y, X=X, npermutations=NPERMUTATIONS, diagonal=False, directed=DIRECTED)
mrqap.mrqap()
mrqap.summary()
print("--- {}, {}: {} seconds ---".format('directed' if DIRECTED else 'undirected', NPERMUTATIONS, time.time() - start_time))
mrqap.plot('betas','results-rg/betas.pdf')
mrqap.plot('tvalues','results-rg/tvalues.pdf')

print '=== RANKING ==='
print mrqap.model.params.sort_values(ascending=False)

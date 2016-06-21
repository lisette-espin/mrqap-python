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
Y  = np.loadtxt('data-cg/data.matrix',delimiter=',')
X1 = np.loadtxt('data-cg/noise-1.0.matrix',delimiter=',')
X2 = np.loadtxt('data-cg/noise-5.0.matrix',delimiter=',')
X3  = np.loadtxt('data-cg/noise-10.0.matrix',delimiter=',')
X4  = np.loadtxt('data-cg/noise-100.0.matrix',delimiter=',')
X5  = np.loadtxt('data-cg/noise-1000.0.matrix',delimiter=',')
X6  = np.loadtxt('data-cg/20Homophily-80Heterophily.matrix',delimiter=',')
X7  = np.loadtxt('data-cg/80Homophily-20Heterophily.matrix',delimiter=',')
X8  = np.loadtxt('data-cg/80ToBlue-20ToRed.matrix',delimiter=',')
X9  = np.loadtxt('data-cg/80ToRed-20ToBlue.matrix',delimiter=',')
X10  = np.loadtxt('data-cg/90Homophily-10Heterophily.matrix',delimiter=',')
X11  = np.loadtxt('data-cg/100Homophily-0Heterophily.matrix',delimiter=',')
X12  = np.loadtxt('data-cg/ToBlueOnly.matrix',delimiter=',')
X13  = np.loadtxt('data-cg/ToRedOnly.matrix',delimiter=',')
X14  = np.loadtxt('data-cg/uniform-0.02.matrix',delimiter=',')

X = {'NOISE1':X1, 'NOISE5':X2, 'NOISE10':X3, 'NOISE100':X4, 'NOISE1000':X5,
     'Hom20Het80':X6, 'Hom80Het20':X7, 'Blue80Red20':X8, 'Red80Blue20':X9,
     'Hom90Het10':X10,'Hom100Het0':X11, 'ToBlue':X12, 'ToRed':X13, 'UNIFORM':X14}
Y = {'DATA':Y}
np.random.seed(1)

#######################################################################
# QAP
#######################################################################
start_time = time.time()
mrqap = MRQAP(Y=Y, X=X, npermutations=NPERMUTATIONS, diagonal=False, directed=DIRECTED, standarized=False)
mrqap.mrqap()
mrqap.summary()
print("--- {}, {}: {} seconds ---".format('directed' if DIRECTED else 'undirected', NPERMUTATIONS, time.time() - start_time))
mrqap.plot('betas','results-cg/betas.pdf')
mrqap.plot('tvalues','results-cg/tvalues.pdf')

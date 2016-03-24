__author__ = 'lisette.espin'

#######################################################################
# References
# - http://www.albany.edu/faculty/kretheme/PAD637/ClassNotes/Spring%202013/Lab8.pdf
#######################################################################

#######################################################################
# Dependencies
#######################################################################
import numpy as np
import utils
from qap import QAP

#######################################################################
# Data
# Source: http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm
#######################################################################
X = np.loadtxt('data/crudematerials.dat')
Y = np.loadtxt('data/manufacturedgoods.dat')
utils.printf('Crude Materials: \n{}'.format(X))
utils.printf('Manufactured Goods: \n{}'.format(Y))
np.random.seed(15843)

#######################################################################
# QAP
#######################################################################
qap = QAP(X, Y)
qap.init()
qap.qap(npermutations=5000)
qap.summary()
qap.plot()


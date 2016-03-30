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
from mrqap import MRQAP

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
qap = QAP(X, Y, 5000)
qap.init()
qap.qap()
qap.summary()
qap.plot()
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
X =  minfo = np.loadtxt('data/info.dat')
Y = np.loadtxt('data/money.dat')
utils.printf('Information: \n{}'.format(X))
utils.printf('Money Exchange: \n{}'.format(Y))

#######################################################################
# QAP
#######################################################################
qap = QAP(X, Y)
qap.init()
qap.qap(npermutations=2000)
qap.summary()
qap.plot()


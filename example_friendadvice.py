__author__ = 'lisette.espin'

#######################################################################
# References
# - http://www.albany.edu/faculty/kretheme/PAD637/ClassNotes/Spring%202013/Lab8.pdf
#######################################################################

#######################################################################
# Dependencies
#######################################################################
import numpy as np

from libs import utils
from libs.qap import QAP


#######################################################################
# Data
# Source: http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm
#######################################################################
X =  minfo = np.loadtxt('data/friendship.dat')
Y = np.loadtxt('data/advice.dat')
utils.printf('Friendship: \n{}'.format(X))
utils.printf('Advise: \n{}'.format(Y))
np.random.seed(831)

#######################################################################
# QAP
#######################################################################
qap = QAP(Y, X, 2000)
qap.qap()
qap.summary()
qap.plot()

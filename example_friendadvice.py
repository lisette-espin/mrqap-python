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

#######################################################################
# QAP
#######################################################################
qap = QAP(X, Y, 2000)
qap.init()
qap.qap()
qap.summary()
qap.plot()

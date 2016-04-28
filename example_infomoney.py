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
minfo =  minfo = np.loadtxt('data/info.dat')
mmoney = np.loadtxt('data/money.dat')
utils.printf('Information: \n{}'.format(minfo))
utils.printf('Money Exchange: \n{}'.format(mmoney))

#######################################################################
# QAP
#######################################################################
qap = QAP(mmoney, minfo)
qap.init()
qap.qap()

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
mfriendship =  minfo = np.loadtxt('data/friendship.dat')
madvise = np.loadtxt('data/advice.dat')
utils.printf('Friendship: \n{}'.format(mfriendship))
utils.printf('Advise: \n{}'.format(madvise))

#######################################################################
# QAP
#######################################################################
qap = QAP(madvise, mfriendship)
qap.init()
qap.qap()

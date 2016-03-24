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
from mrqap import MRQAP

#######################################################################
# Data
# Source: http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm
#######################################################################
X1 = np.loadtxt('data/crudematerials.dat')
X2 = np.loadtxt('data/foods.dat')
X3 = np.loadtxt('data/manufacturedgoods.dat')
X4 = np.loadtxt('data/minerals.dat')
Y = np.loadtxt('data/diplomatic.dat')
X = {'CRUDEMATERIALS':X1, 'FOODS':X2, 'MANUFACTUREDGOODS':X3, 'MINERALS':X4}
utils.printf('Crude Materials: \n{}'.format(X1))
utils.printf('Foods: \n{}'.format(X2))
utils.printf('Manufactured Goods: \n{}'.format(X3))
utils.printf('Minerals: \n{}'.format(X4))
utils.printf('Diplomatic Exchange: \n{}'.format(Y))
np.random.seed(473)

#######################################################################
# QAP
#######################################################################
mrqap = MRQAP(Y, X)
mrqap.init()
mrqap.fit()
mrqap.mrqap(npermutations=2000)
mrqap.summary()
#mrqap.plot()


__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import numpy as np
import utils
from scipy.stats.stats import pearsonr
from scipy.stats import linregress
import math
import itertools
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest
import random
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
#from decimal import Decimal

#######################################################################
# MRQAP
#######################################################################
class MRQAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, Y=None, *X):
        '''
        Initialization of variables
        :param Y: numpy array depended variable
        :param X: numpy array independed variables
        :return:
        '''
        self.Y = Y
        self.X = X

    def init(self):
        return

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def mrqap(self,npermutations=None):
        '''
        MultipleRegression Quadratic Assignment Procedure
        :param npermutations:
        :return:
        '''
        return

    #####################################################################################
    # Plots & Prints
    #####################################################################################

    def summary(self):
        return

    def plot(self):
        return

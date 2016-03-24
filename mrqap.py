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
import pandas
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

#######################################################################
# MRQAP
#######################################################################
class MRQAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, Y=None, X=None):
        '''
        Initialization of variables
        :param Y: numpy array depended variable
        :param X: dictionary of numpy array independed variables
        :return:
        '''
        self.Y = Y
        self.X = X
        self.data = None
        self.model = None
        self.v = None
        self.n = 0 if X is None and Y is None else Y.shape[0]

    def init(self):
        yflatten = np.delete(self.Y.flatten(), [i*(self.n+1)for i in range(self.n)])
        self.v = {'y':yflatten}
        for k,x in self.X.items():
            self.v[k] = np.delete(x.flatten(), [i*(self.n+1)for i in range(self.n)])
        self.data = pandas.DataFrame(self.v)

    def fit(self):
        self.model = ols('y ~ {}'.format(' + '.join([k for k in self.v.keys() if k != 'y'])), self.data).fit()

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
        # Print the summary
        utils.printf('Summary:\n{}'.format(self.model.summary()))

        utils.printf("\nRetrieving manually the parameter estimates:\n{}".format(self.model._results.params))

        # Peform analysis of variance on fitted linear model
        anova_results = anova_lm(self.model)
        utils.printf('\nANOVA results:\n{}'.format(anova_results))

    def plot(self):
        return

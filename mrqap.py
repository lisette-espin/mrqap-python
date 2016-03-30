__author__ = 'espin'

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
import sys

#######################################################################
# MRQAP
#######################################################################
class MRQAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, X=None, Y=None):
        '''
        Initialization of variables
        :param X: dictionary of numpy array independed variables
        :param Y: numpy array depended variable
        :return:
        '''
        self.X = X          # independent variables: dictionary of numpy.array
        self.Y = Y          # dependent variable: numpy.array
        self.data = None    # Pandas DataFrame
        self.model = None   # OLS Model y ~ x1 + x2 + x3
        self.v = None       # vectorized matrices, flatten variables with no diagonal
        self.betas = {}     # betas distribution
        self.n = 0 if X is None and Y is None else Y.shape[0]               # number of nodes (rows/columns)
        self.permutations = list(itertools.permutations(range(self.n),2))   # all possible shuffleings

    def init(self):
        self.v = {'y':np.delete(self.Y.flatten(), [i*(self.n+1)for i in range(self.n)])} # vectorizing Y and removing diagonal
        for k,x in self.X.items():
            if k == 'y':
                utils.printf('ERROR: Idependent variable cannot be named \'y\'')
                sys.exit(0)
            self.v[k] = np.delete(x.flatten(), [i*(self.n+1)for i in range(self.n)])    # vectorizing X's and removing diagonal
        self.data = pandas.DataFrame(self.v)

    def fit(self):
        self.model = ols('y ~ {}'.format(' + '.join([k for k in self.v.keys() if k != 'y'])), self.data).fit()

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def mrqap(self,maxpermutations=None):
        '''
        MultipleRegression Quadratic Assignment Procedure
        :param maxpermutations: maximun number of shuffleings
        :return:
        '''
        npermutations = min(maxpermutations, math.factorial(self.n))
        self._shuffle(npermutations)

    def _shuffle(self, npermutations):
        self.Ymod = self.Y.copy()
        for t in range(npermutations):
            tuple = random.randint(0,len(self.permutations)-1)
            i = self.permutations[tuple][0]
            j = self.permutations[tuple][1]
            utils._swap_cols(self.Ymod, i, j)
            utils._swap_rows(self.Ymod, i, j)
            model = self._newfit()

            yflatten = np.delete(self.Ymod.flatten(), [i*(self.n+1)for i in range(self.n)])
            print np.corrcoef([x for k,x in self.v.items() if k!='y'],yflatten)
            raw_input('...')

            print model.summary()
            raw_input('...')

    def _newfit(self):
        newv = {'ymod':np.delete(self.Ymod.flatten(), [i*(self.n+1)for i in range(self.n)])}
        for k,x in self.v.items():
            if k != 'y':
                newv[k] = x
        newdata = pandas.DataFrame(newv)
        model = ols('ymod ~ {}'.format(' + '.join([k for k in newv.keys() if k != 'ymod'])), newdata).fit()
        return model

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

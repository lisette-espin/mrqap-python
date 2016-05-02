__author__ = 'espin'

#######################################################################
# Dependencies
#######################################################################
import sys
import itertools
import collections
import numpy as np
import pandas
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy import stats
from random import shuffle
from libs import utils
# from statsmodels.stats.anova import anova_lm
# import random
# from statsmodels.stats.weightstats import ztest

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
        self.X = X                              # independent variables: dictionary of numpy.array
        self.Y = Y                              # dependent variable: numpy.array
        self.data = None                        # Pandas DataFrame
        self.model = None                       # OLS Model y ~ x1 + x2 + x3
        self.v = None                           # vectorized matrices, flatten variables with no diagonal
        self.betas = collections.OrderedDict()  # betas distribution
        self.E = {}                             # betas (expected values)
        self.n = 0 if X is None and Y is None else Y.shape[0]               # number of nodes (rows/columns)
        self.permutations = list(itertools.permutations(range(self.n),2))   # all possible shuffleings

    def init(self, diagonal):
        self.v = {'y':self._getFlatten(self.Y, diagonal)}
        self.betas['INTERCEPT'] = []
        for k,x in self.X.items():
            if k == 'y':
                utils.printf('ERROR: Idependent variable cannot be named \'y\'')
                sys.exit(0)
            self.v[k] = self._getFlatten(x,diagonal)
            self.betas[k] = []
        self.data = pandas.DataFrame(self.v)

    def fit(self):
        self.model = ols('y ~ {}'.format(' + '.join([k for k in self.v.keys() if k != 'y'])), self.data).fit()
        # Storing beta coeficcients
        betas = self.model._results.params
        for idx,k in enumerate(self.betas.keys()):
            self.E[k] = round(betas[idx],6)

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def mrqap(self,npermutations=None, diagonal=False):
        '''
        MultipleRegression Quadratic Assignment Procedure
        :param maxpermutations: maximun number of shuffleings
        :return:
        '''
        self.init(diagonal)
        self.fit()
        self._shuffle(npermutations, diagonal)

    def _shuffle(self, npermutations, diagonal):
        self.Ymod = self.Y.copy()
        shuffle(self.permutations)
        for tuple in range(npermutations if npermutations < len(self.permutations) else len(self.permutations)):
            # tuple = random.randint(0,len(self.permutations)-1)
            i = self.permutations[tuple][0]
            j = self.permutations[tuple][1]
            utils._swap_cols(self.Ymod, i, j)
            utils._swap_rows(self.Ymod, i, j)
            model = self._newfit(diagonal)
            self._update_betas(model._results.params)

    def _newfit(self, diagonal):
        newv = {'ymod':self._getFlatten(self.Ymod, diagonal)}
        for k,x in self.v.items():
            if k != 'y':
                newv[k] = x
        newdata = pandas.DataFrame(newv)
        model = ols('ymod ~ {}'.format(' + '.join([k for k in newv.keys() if k != 'ymod'])), newdata).fit()
        return model

    def _update_betas(self, betas):
        for idx,k in enumerate(self.betas.keys()):
                self.betas[k].append(round(betas[idx],6))

    def _getFlatten(self, original, diagonal):
        tmp = original.flatten()
        if not diagonal:
            tmp = np.delete(tmp, [i*(self.n+1)for i in range(self.n)])
        return tmp

        # tmp = original.copy()
        # if not diagonal:
        #     np.fill_diagonal(tmp,0)
        # f = tmp.flatten()
        # del(tmp)
        # return f

    #####################################################################################
    # Plots & Prints
    #####################################################################################

    def summary(self):

        # Print the summary
        utils.printf('')
        utils.printf('=== Summary OLS (original) ===\n{}'.format(self.model.summary()))

        # Summary of beta coefficients
        utils.printf('')
        utils.printf('=== Summary beta coefficients ===')
        utils.printf('{:20s}{:>10s}{:>10s}{:>10s}{:>12s}{:>12s}{:>15s}{:>12s}'.format('INDEPENDENT VAR.','MIN','MEDIAN','MEAN','MAX','COEF.','T-TEST','P-VALUE'))
        for k,v in self.betas.items():
            tstats = stats.ttest_1samp(v,self.E[k])
            utils.printf('{:20s}{:10f}{:10f}{:10f}{:12f}{:>12f}{:15f}{:12f}'.format(k,min(v),sorted(v)[len(v)/2],sum(v)/len(v),max(v),self.E[k],round(float(tstats[0]),2),round(float(tstats[1]),2)))

        #tmp = [' - {}: {}'.format(k,self.model._results.params[idx]) for idx,k in enumerate(self.betas.keys())]
        #tmp.append(' - Intercept: {}'.format(self.model._results.params[0]))
        #utils.printf("Retrieving manually the parameter estimates (betas):\n{}".format('\n'.join(tmp)))
        # Peform analysis of variance on fitted linear model
        #anova_results = anova_lm(self.model)
        #utils.printf('ANOVA results:\n{}'.format(anova_results))
        #print '{}:({},{}) ttest:{} pvalue:{} rsquared:{}'.format(t,i,j,model._results.tvalues, model._results.pvalues, model._results.rsquared)

    def plot(self):
        '''
        Plots frequency of pearson's correlation values
        :return:
        '''
        plt.figure(1)
        ncols = round(len(self.betas.keys())/2.0)
        for idx,k in enumerate(self.betas.keys()):
            plt.subplot(2,ncols,idx+1)
            plt.hist(self.betas[k])
            plt.xlabel('regression coefficients')
            plt.ylabel('frequency')
            plt.title(k)
            plt.grid(True)
        plt.show()
        plt.close()
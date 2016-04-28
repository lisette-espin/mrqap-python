__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import math
import itertools
import random

import numpy as np
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from libs import utils


#######################################################################
# QAP
#######################################################################
class QAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, X=None, Y=None, npermutations=-1):
        '''
        Initialization of variables
        :param X: numpy array independed variable
        :param Y: numpy array depended variable
        :return:
        '''
        self.X = X
        self.Y = Y
        self.npermutations = npermutations
        self.beta = None
        self.n = 0 if X is None and Y is None else Y.shape[0]
        self.permutations = list(itertools.permutations(range(self.n),2))
        self.Ymod = None
        self.betas = {}

    def init(self):
        '''
        Shows the correlation of the initial/original variables (no shuffeling)
        :return:
        '''
        self.beta = self.correlation(self.X, self.Y)
        self.stats(self.X, self.Y)

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def qap(self):
        '''
        Quadratic Assignment Procedure
        :param npermutations:
        :return:
        '''
        self._shuffle()

    def _shuffle(self):
        self.Ymod = self.Y.copy()
        for t in range(self.npermutations if self.npermutations is not None else math.factorial(self.n)):
            tuple = random.randint(0,len(self.permutations)-1)
            i = self.permutations[tuple][0]
            j = self.permutations[tuple][1]
            utils._swap_cols(self.Ymod, i, j)
            utils._swap_rows(self.Ymod, i, j)
            self._addBeta(self.correlation(self.X, self.Ymod, False))

    def correlation(self, x, y, show=True):
        '''
        Computes Pearson's correlation value of variables x and y.
        Diagonal values are removed.
        :param x: numpy array independent variable
        :param y: numpu array dependent variable
        :param show: if True then shows pearson's correlation and p-value.
        :return:
        '''
        xflatten = np.delete(x, [i*(self.n+1)for i in range(self.n)])
        yflatten = np.delete(y, [i*(self.n+1)for i in range(self.n)])
        pc = pearsonr(xflatten, yflatten)
        if show:
            utils.printf('Pearson Correlation: {}'.format(pc[0]))
            utils.printf('p-value: {}'.format(pc[1]))
        return pc

    #####################################################################################
    # Handlers
    #####################################################################################

    def _addBeta(self, p):
        '''
        frequency dictionary of pearson's correlation values
        :param p: person's correlation value
        :return:
        '''
        p = round(p[0],6)
        if p not in self.betas:
            self.betas[p] = 0
        self.betas[p] += 1

    #####################################################################################
    # Plots & Prints
    #####################################################################################

    def summary(self):
        utils.printf('')
        utils.printf('# Permutations: {}'.format(self.npermutations))
        utils.printf('Correlation coefficients:\n{}'.format(self.betas))
        utils.printf('Percentages betas:\n{}'.format(['{}:{}'.format(k,round(v*100/float(sum(self.betas.values())),2)) for k,v in self.betas.items()]))
        utils.printf('Sum all betas: {}'.format(sum(self.betas.keys())))
        utils.printf('min betas: {}'.format(min(self.betas.keys())))
        utils.printf('max betas: {}'.format(max(self.betas.keys())))
        utils.printf('average betas: {}'.format(np.average(self.betas.keys())))
        utils.printf('std betas: {}'.format(np.std(self.betas.keys())))
        utils.printf('prop >= {}: {}'.format(self.beta[0], sum([v for k,v in self.betas.items() if k >= self.beta[0] ])/float(sum(self.betas.values()))))
        utils.printf('prop <= {}: {} (proportion of randomly generated correlations that were as {} as the observed)'.format(self.beta[0], sum([v for k,v in self.betas.items() if k <= self.beta[0] ])/float(sum(self.betas.values())),'large' if self.beta[0] >= 0 else 'small'))
        utils.printf('')
        self.ols(self.X, self.Ymod)

    def plot(self):
        '''
        Plots frequency of pearson's correlation values
        :return:
        '''
        plt.bar(self.betas.keys(), self.betas.values(),0.0005)
        plt.xlabel('regression coefficients')
        plt.ylabel('frequency')
        plt.title('MRQAP')
        plt.grid(True)
        #plt.savefig("test.png")
        plt.show()
        plt.close()

    #####################################################################################
    # Others
    #####################################################################################

    def stats(self, x, y):
        xflatten = np.delete(x, [i*(self.n+1)for i in range(self.n)])
        yflatten = np.delete(y, [i*(self.n+1)for i in range(self.n)])
        p = np.corrcoef(xflatten,yflatten)
        utils.printf('Pearson\'s correlation:\n{}'.format(p))
        utils.printf('Z-Test:{}'.format(ztest(xflatten, yflatten)))
        utils.printf('T-Test:{}'.format(ttest_ind(xflatten, yflatten)))

    def ols(self, x, y):
        xflatten = np.delete(x, [i*(self.n+1)for i in range(self.n)])
        yflatten = np.delete(y, [i*(self.n+1)for i in range(self.n)])
        xflatten = sm.add_constant(xflatten)
        model = sm.OLS(yflatten,xflatten)
        results = model.fit()
        print results.summary()

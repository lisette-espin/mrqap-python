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
# QAP
#######################################################################
class QAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, X=None, Y=None):
        '''
        Initialization of variables
        :param X: numpy array independed variable
        :param Y: numpy array depended variable
        :return:
        '''
        self.X = X
        self.Y = Y
        self.beta = None
        self.n = 0 if X is None and Y is None else Y.shape[0]
        self.permutations = list(itertools.permutations(range(self.n),2))
        self.Xmod = None
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

    def qap(self,npermutations=None):
        '''
        Quadratic Assignment Procedure
        :param npermutations:
        :return:
        '''
        self.Xmod = self.X.copy()
        for t in range(npermutations if npermutations is not None else math.factorial(self.n)):
            tuple = random.randint(0,len(self.permutations)-1)
            i = self.permutations[tuple][0]
            j = self.permutations[tuple][1]
            self._swap_cols(self.Xmod, i, j)
            self._swap_rows(self.Xmod, i, j)
            self._addBeta(self.correlation(self.Xmod, self.Y, False))

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

    def _swap_cols(self, arr, frm, to):
        arr[:,[frm, to]] = arr[:,[to, frm]]

    def _swap_rows(self, arr, frm, to):
        arr[[frm, to],:] = arr[[to, frm],:]

    #####################################################################################
    # Plots & Prints
    #####################################################################################

    def summary(self):
        utils.printf('Correlation coefficients:\n{}'.format(self.betas))
        utils.printf('Percentages betas:\n{}'.format(['{}:{}'.format(k,round(v*100/float(sum(self.betas.values())),2)) for k,v in self.betas.items()]))
        utils.printf('Sum all betas: {}'.format(sum(self.betas.keys())))
        utils.printf('min betas: {}'.format(min(self.betas.keys())))
        utils.printf('max betas: {}'.format(max(self.betas.keys())))
        utils.printf('average betas: {}'.format(np.average(self.betas.keys())))
        utils.printf('std betas: {}'.format(np.std(self.betas.keys())))
        utils.printf('prop >= {}: {}'.format(self.beta[0], sum([v for k,v in self.betas.items() if k >= self.beta[0] ])/float(sum(self.betas.values()))))
        utils.printf('prop <= {}: {} (proportion of randomly generated correlations that were as {} as the observed)'.format(self.beta[0], sum([v for k,v in self.betas.items() if k <= self.beta[0] ])/float(sum(self.betas.values())),'large' if self.beta[0] >= 0 else 'small'))

    def plot(self):
        '''
        Plots frequency of pearson's correlation values
        :return:
        '''
        #betas = utils.sortDictByKey(self.betas, False)
        #plt.bar([c[0] for c in betas], [c[1] for c in betas],0.2)
        plt.bar(self.betas.keys(), self.betas.values(),0.2)
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
        xflatten = x.flatten()
        yflatten = y.flatten()
        xflatten = sm.add_constant(xflatten)
        model = sm.OLS(yflatten,xflatten)
        results = model.fit()
        print results.params
        print results.tvalues
        print results.t_test([1, 0])
        print results.f_test(np.identity(2))

